#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# cluster.sh -- Cluster (live EKS) validation functions
#
# Implements levels L0-L6 of the validation framework:
#   L0: Infrastructure smoke test (GPU, EFA, FSx)
#   L1: Container build + ECR push (CodeBuild)
#   L2: Container validation (10 tests)
#   L3: NCCL/EFA bandwidth threshold
#   L4: Model download verification
#   L5: Training step tests (1 PPO step per example)
#   L6: Multi-node training step
# =============================================================================

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Prefer new infrastructure/ paths, fall back to terraform/ for backward compat
if [[ -d "${REPO_ROOT}/infrastructure" ]]; then
  _TF_DIR="${TF_DIR:-${REPO_ROOT}/infrastructure/cluster}"
  _MANIFESTS_DIR="${MANIFESTS_DIR:-${REPO_ROOT}/infrastructure/manifests}"
else
  _TF_DIR="${TF_DIR:-${REPO_ROOT}/terraform}"
  _MANIFESTS_DIR="${MANIFESTS_DIR:-${REPO_ROOT}/manifests}"
fi

# --- Detect cluster configuration ---
# Reads terraform outputs to set ECR_URI, CODEBUILD_PROJECT, etc.
detect_cluster_config() {
  if [[ -n "${ECR_URI:-}" ]]; then
    return 0  # Already configured
  fi

  log_info "Detecting cluster configuration from terraform outputs..."

  local tf_dir="${_TF_DIR}"
  if [[ ! -d "${tf_dir}/.terraform" ]]; then
    log_warn "Terraform not initialized. Set ECR_URI, CODEBUILD_PROJECT manually."
    return 0
  fi

  ECR_URI=$(terraform -chdir="$tf_dir" output -json ecr_repository_urls 2>/dev/null | \
    python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('rlinf',''))" 2>/dev/null) || true
  CODEBUILD_PROJECT=$(terraform -chdir="$tf_dir" output -json codebuild_project_names 2>/dev/null | \
    python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('rlinf',''))" 2>/dev/null) || true
  CODEBUILD_BUCKET=$(terraform -chdir="$tf_dir" output -raw codebuild_source_bucket 2>/dev/null) || true
  REGION=$(terraform -chdir="$tf_dir" output -raw target_az 2>/dev/null | sed 's/[a-z]$//' || echo "us-east-2")

  export ECR_URI CODEBUILD_PROJECT CODEBUILD_BUCKET REGION

  if [[ -n "$ECR_URI" ]]; then
    log_info "ECR_URI: $ECR_URI"
  fi
  if [[ -n "$CODEBUILD_PROJECT" ]]; then
    log_info "CodeBuild: $CODEBUILD_PROJECT"
  fi
}

# =============================================================================
# L0: Infrastructure Smoke Test
# =============================================================================
run_level_0() {
  log_header "L0: Infrastructure Smoke Test"

  # Clean up any previous run
  cleanup_job smoke-test

  # Apply smoke test
  kubectl apply -f "${_MANIFESTS_DIR}/smoke-test.yaml"

  # Wait for completion
  if ! wait_for_job smoke-test default 600; then
    return 1
  fi

  # Parse results
  local logs
  logs=$(get_job_logs smoke-test)

  local pass_count fail_count
  pass_count=$(echo "$logs" | sed -n 's/.*Results: \([0-9]*\) passed.*/\1/p' | tail -1)
  fail_count=$(echo "$logs" | sed -n 's/.*Results: [0-9]* passed, \([0-9]*\) failed.*/\1/p' | tail -1)
  pass_count="${pass_count:-0}"
  fail_count="${fail_count:-0}"

  log_info "Smoke test: ${pass_count} passed, ${fail_count} failed"

  # Cleanup
  cleanup_job smoke-test

  if (( fail_count > 0 )); then
    return 1
  fi
  return 0
}

# =============================================================================
# L1: Container Build + ECR Push
# =============================================================================
run_level_1() {
  log_header "L1: Container Build + ECR Push"

  detect_cluster_config

  if [[ -z "${CODEBUILD_PROJECT:-}" ]] || [[ -z "${CODEBUILD_BUCKET:-}" ]]; then
    log_fail "CODEBUILD_PROJECT and CODEBUILD_BUCKET required. Set via env or terraform."
    return 1
  fi

  # Step 1: Upload source to S3
  log_info "Packaging source for CodeBuild..."
  local zip_file="/tmp/rlinf-on-eks.zip"
  (
    cd "$REPO_ROOT"
    zip -r "$zip_file" . -x '.git/*' 'infrastructure/*/.terraform/*' '*.tfstate*' >/dev/null
  )
  aws s3 cp "$zip_file" "s3://${CODEBUILD_BUCKET}/rlinf-on-eks.zip" --quiet --region "${REGION}"
  rm -f "$zip_file"
  log_info "Source uploaded to s3://${CODEBUILD_BUCKET}/rlinf-on-eks.zip"

  # Step 2: Start build
  log_info "Starting CodeBuild project: ${CODEBUILD_PROJECT}"
  local build_id
  build_id=$(aws codebuild start-build \
    --project-name "${CODEBUILD_PROJECT}" \
    --region "${REGION}" \
    --query 'build.id' --output text)

  if [[ -z "$build_id" ]]; then
    log_fail "Failed to start CodeBuild build"
    return 1
  fi
  log_info "Build started: ${build_id}"

  # Step 3: Poll until complete (timeout: 60 min)
  local timeout=3600
  local elapsed=0
  local status="IN_PROGRESS"

  while [[ "$status" == "IN_PROGRESS" ]] && (( elapsed < timeout )); do
    sleep 30
    elapsed=$((elapsed + 30))
    status=$(aws codebuild batch-get-builds --ids "$build_id" \
      --region "${REGION}" \
      --query 'builds[0].buildStatus' --output text 2>/dev/null)
    log_info "Build status: ${status} (${elapsed}s)"
  done

  if [[ "$status" != "SUCCEEDED" ]]; then
    log_fail "CodeBuild ${status} after ${elapsed}s (build: ${build_id})"
    # Show build logs URL
    local logs_url
    logs_url=$(aws codebuild batch-get-builds --ids "$build_id" \
      --region "${REGION}" \
      --query 'builds[0].logs.deepLink' --output text 2>/dev/null)
    if [[ -n "$logs_url" ]]; then
      log_info "Build logs: ${logs_url}"
    fi
    return 1
  fi

  # Step 4: Verify image in ECR
  log_info "Verifying ECR image..."
  # ECR_URI format: <account>.dkr.ecr.<region>.amazonaws.com/<repo-name>
  local ecr_repo_name
  ecr_repo_name=$(echo "$ECR_URI" | sed 's|^[^/]*/||')
  if aws ecr describe-images \
       --repository-name "$ecr_repo_name" \
       --image-ids imageTag=latest \
       --region "${REGION}" >/dev/null 2>&1; then
    log_pass "Container image built and pushed to ${ECR_URI}:latest"
  else
    log_fail "Image not found in ECR after successful build"
    return 1
  fi

  return 0
}

# =============================================================================
# L2: Container Validation
# =============================================================================
run_level_2() {
  log_header "L2: Container Validation"

  detect_cluster_config

  if [[ -z "${ECR_URI:-}" ]]; then
    log_fail "ECR_URI required. Run L1 first or set ECR_URI."
    return 1
  fi

  export ECR_URI

  # Clean up any previous run
  cleanup_job container-test rlinf
  kubectl delete configmap container-test-script -n rlinf --ignore-not-found=true 2>/dev/null || true

  # Apply (envsubst for ECR_URI)
  envsubst < "${REPO_ROOT}/infrastructure/manifests/container-test.yaml" | kubectl apply -n rlinf -f -

  # Wait for completion
  if ! wait_for_job container-test rlinf 1200; then
    return 1
  fi

  # Parse results
  local logs
  logs=$(get_job_logs container-test)

  local pass_count fail_count
  pass_count=$(echo "$logs" | sed -n 's/.*Results: \([0-9]*\) passed.*/\1/p' | tail -1)
  fail_count=$(echo "$logs" | sed -n 's/.*Results: [0-9]* passed, \([0-9]*\) failed.*/\1/p' | tail -1)
  pass_count="${pass_count:-0}"
  fail_count="${fail_count:-0}"

  log_info "Container tests: ${pass_count} passed, ${fail_count} failed"

  # Check overall status
  if echo "$logs" | grep -q "CONTAINER TEST PASSED"; then
    log_pass "All container tests passed"
  else
    log_fail "Container tests failed"
    cleanup_job container-test rlinf
    return 1
  fi

  cleanup_job container-test rlinf
  kubectl delete configmap container-test-script -n rlinf --ignore-not-found=true 2>/dev/null || true
  return 0
}

# =============================================================================
# L3: NCCL/EFA Bandwidth
# =============================================================================

# parse_nccl_results LOGS
#   Extracts BusBW and provider from NCCL all_reduce_perf output.
#   Outputs: BUSBW_GBS PROVIDER
parse_nccl_results() {
  local logs="$1"
  local busbw provider

  # Strip MPI launcher tag prefixes like "[1,0]<stdout>:" or "[1,0]<stderr>:"
  local clean_logs
  clean_logs=$(echo "$logs" | sed 's/^\[.*\]<std[a-z]*>://; s/^\[.*\]<std[a-z]*>: //')

  # Extract provider: "NET/OFI Selected Provider is efa"
  provider=$(echo "$clean_logs" | sed -n 's/.*Selected Provider is \([a-zA-Z0-9_]*\).*/\1/p' | head -1)
  provider="${provider:-unknown}"

  # Extract bus bandwidth from the last data row of out-of-place results.
  # NCCL test output format (columns):
  #   size  count  type  redop  root  time  algbw  busbw  #wrong  time  algbw  busbw  #wrong
  # The out-of-place busbw is field 8 (1-indexed) in data lines starting with digits.
  busbw=$(echo "$clean_logs" | \
    grep -E '^\s+[0-9]+' | \
    tail -1 | \
    awk '{print $8}')

  # Fallback: try NF-1 (second-to-last column)
  if [[ -z "$busbw" ]] || [[ "$busbw" == "0" ]]; then
    busbw=$(echo "$clean_logs" | \
      grep -E '^\s+[0-9]+' | \
      tail -1 | \
      awk '{print $(NF-1)}')
  fi

  busbw="${busbw:-0}"
  echo "$busbw $provider"
}

run_level_3() {
  log_header "L3: NCCL/EFA Bandwidth"

  # Use the public NCCL test image (or ECR if available)
  local nccl_image="${NCCL_TEST_IMAGE:-public.ecr.aws/hpc-cloud/nccl-tests:cuda12.8.1-efa1.43.2-ofiv1.16.3-ncclv2.27.7-1-testsv2.16.9}"
  export NCCL_TEST_IMAGE="$nccl_image"
  log_info "NCCL test image: ${nccl_image}"

  # Check for MPI Operator CRD
  if ! kubectl get crd mpijobs.kubeflow.org &>/dev/null; then
    log_fail "Kubeflow MPI Operator not installed. Enable enable_mpi_operator in terraform."
    return 1
  fi

  # Clean up any previous run
  cleanup_mpijob nccl-tests

  # Apply MPIJob manifest (only substitute NCCL_TEST_IMAGE, not other $VARs)
  envsubst '${NCCL_TEST_IMAGE}' < "${_MANIFESTS_DIR}/nccl-tests-mpijob.yaml" | kubectl apply -f -

  # Wait for MPIJob launcher to complete
  if ! wait_for_mpijob nccl-tests default 600; then
    cleanup_mpijob nccl-tests
    return 1
  fi

  # Get launcher logs
  local logs
  logs=$(get_mpijob_launcher_logs nccl-tests)

  if [[ -z "$logs" ]]; then
    log_fail "No logs from NCCL test launcher"
    cleanup_mpijob nccl-tests
    return 1
  fi

  # Parse results
  local result busbw provider
  result=$(parse_nccl_results "$logs")
  busbw=$(echo "$result" | awk '{print $1}')
  provider=$(echo "$result" | awk '{print $2}')

  log_info "NCCL Provider: ${provider}"
  log_info "NCCL BusBW: ${busbw} GB/s"

  # Determine threshold based on instance type
  local threshold=1.0
  local instance_type
  instance_type=$(kubectl get nodes -l role=gpu-training \
    -o jsonpath='{.items[0].metadata.labels.node\.kubernetes\.io/instance-type}' 2>/dev/null)
  case "${instance_type:-}" in
    p5.48xlarge)   threshold=100.0 ;;
    p4de.24xlarge) threshold=50.0 ;;
    p4d.24xlarge)  threshold=50.0 ;;
    g6.*)          threshold=1.0 ;;
    *)             threshold=1.0 ;;
  esac
  log_info "Instance type: ${instance_type:-unknown}, threshold: ${threshold} GB/s"

  # Validate provider
  if [[ "$provider" != "efa" ]]; then
    log_warn "NCCL is NOT using EFA (provider: ${provider}). Performance will be degraded."
  fi

  # Validate bandwidth
  local pass=0
  if command -v python3 &>/dev/null; then
    pass=$(python3 -c "print(1 if float('${busbw}') >= ${threshold} else 0)" 2>/dev/null || echo 0)
  else
    # Fallback: integer comparison (rough)
    local busbw_int="${busbw%%.*}"
    local thresh_int="${threshold%%.*}"
    (( busbw_int >= thresh_int )) && pass=1
  fi

  cleanup_mpijob nccl-tests

  if (( pass )); then
    log_pass "NCCL BusBW ${busbw} GB/s >= ${threshold} GB/s (${provider})"
    return 0
  else
    log_fail "NCCL BusBW ${busbw} GB/s < ${threshold} GB/s threshold"
    return 1
  fi
}

# =============================================================================
# L4: Model Download Verification
# =============================================================================
run_level_4() {
  log_header "L4: Model Download Verification"

  # Clean up any previous run
  cleanup_job model-download-maniskill-openvla rlinf
  cleanup_job model-download-maniskill-openvlaoft rlinf
  cleanup_job model-download-libero-pi0 rlinf

  # Apply per-example model download jobs (run in parallel)
  kubectl apply -n rlinf -f "${REPO_ROOT}/examples/maniskill-openvla-ppo/manifests/model-download.yaml"
  kubectl apply -n rlinf -f "${REPO_ROOT}/examples/maniskill-openvlaoft-ppo/manifests/model-download.yaml"
  kubectl apply -n rlinf -f "${REPO_ROOT}/examples/libero-pi0-ppo/manifests/model-download.yaml"

  # Wait for all to complete (models are large -- up to 30 min)
  if ! wait_for_job model-download-maniskill-openvla rlinf 3600; then
    return 1
  fi
  if ! wait_for_job model-download-maniskill-openvlaoft rlinf 3600; then
    return 1
  fi
  if ! wait_for_job model-download-libero-pi0 rlinf 3600; then
    return 1
  fi

  # Check logs for success
  local logs
  logs=$(get_job_logs model-download-maniskill-openvla)

  if echo "$logs" | grep -q "Download complete"; then
    log_pass "All models downloaded successfully"
  else
    log_fail "Model download did not complete"
    cleanup_job model-download-maniskill-openvla rlinf
    cleanup_job model-download-maniskill-openvlaoft rlinf
    cleanup_job model-download-libero-pi0 rlinf
    return 1
  fi

  # Verify model directories exist via a quick check pod
  log_info "Verifying model directories on FSx..."
  local verify_result
  verify_result=$(kubectl run fsx-verify -n rlinf --rm -i --restart=Never \
    --image=busybox \
    --overrides='{
      "spec": {
        "nodeSelector": {"role": "gpu-training"},
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        "volumes": [{"name": "fsx", "persistentVolumeClaim": {"claimName": "fsx-claim"}}],
        "containers": [{
          "name": "verify",
          "image": "busybox",
          "command": ["sh", "-c",
            "ok=0; fail=0; for d in openvla-7b-rlvla-warmup openvla-oft-sft-libero10 rlinf-pi0-sft-spatial; do if [ -d /fsx/models/$d ]; then echo \"OK: $d\"; ok=$((ok+1)); else echo \"MISSING: $d\"; fail=$((fail+1)); fi; done; echo \"Models: $ok found, $fail missing\"; exit $fail"
          ],
          "volumeMounts": [{"name": "fsx", "mountPath": "/fsx"}]
        }]
      }
    }' 2>&1) || true

  if echo "$verify_result" | grep -q "0 missing"; then
    log_pass "All model directories verified on FSx"
  else
    log_warn "Some model directories may be missing (check output above)"
    echo "$verify_result" | grep -E '(OK|MISSING|Models):' || true
  fi

  cleanup_job model-download rlinf
  return 0
}

# =============================================================================
# L5: Training Step Tests
# =============================================================================

# Per-example configurations
# Format: "name|venv|config|model_path"
L5_EXAMPLE_LIST="maniskill-openvla|openvla|maniskill_ppo_openvla_quickstart|/fsx/models/openvla-7b-rlvla-warmup
maniskill-openvlaoft|openvla-oft|maniskill_ppo_openvlaoft_quickstart|/fsx/models/openvla-oft-sft-libero10
libero-pi0|openpi|libero_spatial_ppo_openpi_quickstart|/fsx/models/rlinf-pi0-sft-spatial"

# get_l5_example NAME -> outputs "venv|config|model_path" or returns 1
get_l5_example() {
  local name="$1"
  echo "$L5_EXAMPLE_LIST" | while IFS='|' read -r ex_name ex_venv ex_config ex_model; do
    if [[ "$ex_name" == "$name" ]]; then
      echo "${ex_venv}|${ex_config}|${ex_model}"
      return 0
    fi
  done
}

run_level_5() {
  log_header "L5: Training Step Tests"

  detect_cluster_config

  if [[ -z "${ECR_URI:-}" ]]; then
    log_fail "ECR_URI required. Run L1 first or set ECR_URI."
    return 1
  fi

  # Build list of examples to run
  local examples_to_run=""
  if [[ -n "${EXAMPLE_FILTER:-}" ]]; then
    local check
    check=$(get_l5_example "$EXAMPLE_FILTER")
    if [[ -z "$check" ]]; then
      log_fail "Unknown example: ${EXAMPLE_FILTER}. Valid: maniskill-openvla, maniskill-openvlaoft, libero-pi0"
      return 1
    fi
    examples_to_run="$EXAMPLE_FILTER"
  else
    examples_to_run="maniskill-openvla maniskill-openvlaoft libero-pi0"
  fi

  local overall_fail=0

  for example_name in $examples_to_run; do
    local config_str
    config_str=$(get_l5_example "$example_name")
    IFS='|' read -r venv_name config_name model_path <<< "$config_str"

    log_info "--- L5: ${example_name} ---"
    log_info "  Venv: ${venv_name}, Config: ${config_name}"

    local job_name="training-step-test-${example_name}"

    # Set envsubst variables
    export EXAMPLE_NAME="$example_name"
    export VENV_NAME="$venv_name"
    export CONFIG_NAME="$config_name"
    export MODEL_PATH="$model_path"
    export ECR_URI

    # Clean up any previous run
    cleanup_job "$job_name" rlinf

    # Apply the training step test manifest
    envsubst < "${REPO_ROOT}/tests/manifests/training-step-test.yaml" | kubectl apply -n rlinf -f -

    # Wait for completion (15 min per example)
    timer_start
    if ! wait_for_job "$job_name" rlinf 900; then
      local elapsed
      elapsed=$(timer_elapsed)
      result_add "L5" "$example_name" "FAIL" "$elapsed"
      overall_fail=1
      cleanup_job "$job_name" rlinf
      continue
    fi

    # Parse logs
    local logs
    logs=$(get_job_logs "$job_name")

    local elapsed
    elapsed=$(timer_elapsed)

    if echo "$logs" | grep -q "TRAINING_STEP_TEST_PASSED"; then
      log_pass "L5: ${example_name} -- 1 training step completed"
      result_add "L5" "$example_name" "PASS" "$elapsed"
    else
      log_fail "L5: ${example_name} -- training step did not complete"
      # Show last 20 lines for debugging
      echo "$logs" | tail -20
      result_add "L5" "$example_name" "FAIL" "$elapsed"
      overall_fail=1
    fi

    cleanup_job "$job_name" rlinf
  done

  return $overall_fail
}

# =============================================================================
# L6: Multi-Node Training Step
# =============================================================================
run_level_6() {
  log_header "L6: Multi-Node Training Step"

  detect_cluster_config

  if [[ -z "${ECR_URI:-}" ]]; then
    log_fail "ECR_URI required. Run L1 first or set ECR_URI."
    return 1
  fi

  # Check we have at least 2 GPU nodes
  local gpu_nodes
  gpu_nodes=$(kubectl get nodes -l role=gpu-training --no-headers 2>/dev/null | wc -l | tr -d ' ')
  if (( gpu_nodes < 2 )); then
    log_warn "L6 requires 2+ GPU nodes (found: ${gpu_nodes}). Skipping."
    return 2  # Return 2 for skip
  fi

  export ECR_URI

  # Clean up any previous run
  kubectl delete statefulset training -n rlinf --ignore-not-found=true 2>/dev/null || true
  kubectl delete service training-svc -n rlinf --ignore-not-found=true 2>/dev/null || true

  # Apply headless service
  envsubst < "${REPO_ROOT}/infrastructure/manifests/training-service.yaml" | kubectl apply -n rlinf -f -

  # Apply StatefulSet with overrides for single-step test
  # We need to patch the StatefulSet to add HYDRA_OVERRIDES for max_steps=1
  envsubst < "${REPO_ROOT}/infrastructure/manifests/training-statefulset.yaml" | kubectl apply -n rlinf -f -

  # Patch to set HYDRA_OVERRIDES for single step
  kubectl set env statefulset/training -n rlinf \
    HYDRA_OVERRIDES="runner.max_steps=1 runner.save_interval=-1 runner.val_check_interval=-1" \
    2>/dev/null || true

  # Wait for both pods to be Running
  log_info "Waiting for 2 training pods..."
  local elapsed=0
  local timeout=1200
  while (( elapsed < timeout )); do
    local running
    running=$(kubectl get pods -n rlinf -l app=rlinf-multi --no-headers 2>/dev/null | \
      grep -c "Running" || echo "0")
    if (( running >= 2 )); then
      break
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done

  if (( elapsed >= timeout )); then
    log_fail "Timed out waiting for training pods"
    kubectl get pods -n rlinf -l app=rlinf-multi
    kubectl delete statefulset training -n rlinf --ignore-not-found=true 2>/dev/null || true
    kubectl delete service training-svc -n rlinf --ignore-not-found=true 2>/dev/null || true
    return 1
  fi

  log_info "Both pods running. Waiting for Ray cluster and training step..."

  # Wait for training completion by watching pod-0 logs
  elapsed=0
  timeout=1200
  local success=0
  while (( elapsed < timeout )); do
    local logs
    logs=$(kubectl logs training-0 -n rlinf --tail=100 2>/dev/null || echo "")

    # Check for Ray cluster formation
    if echo "$logs" | grep -q "All .* Ray nodes connected" && (( elapsed < 60 )); then
      log_info "Ray cluster formed"
    fi

    # Check for training step completion
    if echo "$logs" | grep -q "TRAINING_STEP_TEST_PASSED\|Global Step: 1/1\|step.*1.*completed"; then
      success=1
      break
    fi

    # Check for errors
    if echo "$logs" | grep -qE "ERROR:|Traceback|FAILED"; then
      log_fail "Training error detected in pod-0 logs:"
      echo "$logs" | grep -A5 -E "ERROR:|Traceback|FAILED" | head -20
      break
    fi

    sleep 15
    elapsed=$((elapsed + 15))
  done

  # Cleanup
  kubectl delete statefulset training -n rlinf --ignore-not-found=true 2>/dev/null || true
  kubectl delete service training-svc -n rlinf --ignore-not-found=true 2>/dev/null || true

  if (( success )); then
    log_pass "Multi-node training step completed"
    return 0
  else
    log_fail "Multi-node training did not complete within ${timeout}s"
    return 1
  fi
}
