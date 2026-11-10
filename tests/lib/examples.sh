#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# examples.sh -- L7: run each example README's steps as a customer would
#
# Commands are extracted verbatim from the README bash blocks
# (tests/lib/readme_steps.py --harness: `kubectl logs -f` becomes wait + tail).
# The only change to a committed training manifest is the documented smoke edit
# (tests/lib/smoke_edit.py: runner.max_steps=2 runner.save_interval=-1).
# Results go to ${L7_RESULTS:-validate-results-L7.tsv}:
#   example  step  status  seconds  image  instance  k8s_version  step_time
# =============================================================================

L7_ALL="maniskill-openvlaoft-ppo libero-pi0-ppo wan-openvlaoft-grpo dreamzero isaaclab-gr00t-ppo"

# l7_run_steps EXAMPLE NAME README HEADING...  -> runs the extracted steps, records a row
l7_run_steps() {
  local ex="$1" name="$2" readme="$3"; shift 3
  local script log rc start from=()
  while [[ "${1:-}" == --from || "${1:-}" == --to ]]; do from+=("$1" "$2"); shift 2; done
  script=$(mktemp "${TMPDIR:-/tmp}/l7-${ex}-${name}.XXXXXX")
  log="${L7_LOG_DIR}/${ex}-${name}.log"
  python3 "${REPO_ROOT}/tests/lib/readme_steps.py" --harness ${from[@]+"${from[@]}"} "$readme" "$@" > "$script"
  if [[ ! -s "$script" ]] || ! grep -qv '^# ---' "$script"; then
    log_fail "L7 ${ex}/${name}: no bash block under '$*' in ${readme}"
    l7_record "$ex" "$name" FAIL 0 ""
    return 1
  fi
  log_info "L7 ${ex}/${name}: $(grep -cv '^\s*\(#\|$\)' "$script") commands from ${L7_README_LABEL:-${readme#"${REPO_ROOT}"/}}"
  start=$(date +%s)
  (cd "$REPO_ROOT" && bash -euo pipefail -x "$script") > "$log" 2>&1
  rc=$?
  rm -f "$script"
  local status=PASS; (( rc == 0 )) || status=FAIL
  l7_record "$ex" "$name" "$status" $(( $(date +%s) - start )) "$log"
  if [[ $status == PASS ]]; then
    log_pass "L7 ${ex}/${name}"
  else
    log_fail "L7 ${ex}/${name} (log: ${log})"; tail -15 "$log"
  fi
  return $rc
}

# l7_record EXAMPLE STEP STATUS SECONDS LOG -- one TSV row; instance type and step time
# come from the Jobs/RayJobs the step created (named in its `kubectl apply` output).
l7_record() {
  local ex="$1" name="$2" status="$3" secs="$4" log="$5" inst="" st="" img="" job pod node v
  if [[ -n "$log" && -f "$log" ]]; then
    while read -r job; do
      pod=$(kubectl get pods -n "$NAMESPACE" -l "job-name=${job}" -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null) || true
      if [[ -z "$pod" ]]; then  # RayJob: head pod of its RayCluster
        pod=$(kubectl get pods -n "$NAMESPACE" -l "ray.io/node-type=head" -o name 2>/dev/null | grep -m1 "/${job}-" | cut -d/ -f2) || true
      fi
      [[ -n "$pod" ]] || continue
      node=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}' 2>/dev/null) || true
      v=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].image}' 2>/dev/null) || true
      [[ -n "$v" ]] && img="$v"
      if [[ -n "$node" ]]; then
        v=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}' 2>/dev/null) || true
        [[ -n "$v" ]] && inst="$v"
      fi
      v=$(kubectl logs -n "$NAMESPACE" "$pod" 2>/dev/null | LC_ALL=C sed 's/\x1b\[[0-9;]*m//g' \
        | LC_ALL=C grep -aoE 'Step Time: [0-9.]+s' | tail -1 | awk '{print $3}' | tr -d s) || true
      [[ -n "$v" ]] && st="$v"
    done < <(grep -aoE '^(job\.batch|rayjob\.ray\.io)/[a-z0-9-]+ (created|configured)' "$log" | awk '{print $1}' | cut -d/ -f2)
  fi
  # image: what the step's Jobs actually ran (RayJob-only steps fall back to $IMAGE)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ex" "$name" "$status" "$secs" "${img:-${IMAGE:--}}" \
    "${inst:--}" "${L7_K8S_VERSION:--}" "${st:--}" >> "$L7_RESULTS"
  result_add "L7" "${ex}/${name}" "$status" "$secs"
}

# Delete the Jobs an example's README creates, so every step starts clean.
l7_cleanup() {
  kubectl delete job -n "$NAMESPACE" --ignore-not-found --wait=true "$@" >/dev/null 2>&1 || true
}

l7_smoke() {  # SMOKE_FILTER for the training-manifest apply
  export SMOKE_FILTER="python3 ${REPO_ROOT}/tests/lib/smoke_edit.py"
}

l7_maniskill-openvlaoft-ppo() {
  local d="${REPO_ROOT}/examples/maniskill-openvlaoft-ppo" rc=0
  l7_cleanup model-download-maniskill-openvlaoft rlinf-maniskill-openvlaoft
  l7_smoke
  l7_run_steps maniskill-openvlaoft-ppo train "$d/README.md" "1. Download" "2. Launch" "3. Monitor" || rc=1
  unset SMOKE_FILTER
  l7_cleanup rlinf-maniskill-openvlaoft
  return $rc
}

l7_libero-pi0-ppo() {
  local d="${REPO_ROOT}/examples/libero-pi0-ppo" rc=0
  l7_cleanup model-download-libero-pi0 rlinf-libero-pi0
  l7_smoke
  l7_run_steps libero-pi0-ppo train "$d/README.md" "1. Download" "2. Launch" "3. Monitor" || rc=1
  unset SMOKE_FILTER
  l7_cleanup rlinf-libero-pi0
  return $rc
}

l7_wan-openvlaoft-grpo() {
  local d="${REPO_ROOT}/examples/wan-openvlaoft-grpo" rc=0
  l7_cleanup model-download-wan-openvlaoft rlinf-wan-openvlaoft-grpo
  l7_smoke
  l7_run_steps wan-openvlaoft-grpo train "$d/README.md" "1. Download" "2. Launch" "3. Monitor" || rc=1
  unset SMOKE_FILTER
  l7_cleanup rlinf-wan-openvlaoft-grpo
  return $rc
}

# DreamZero: README steps 3 (SFT, RayJob on 2x p5en) -> 4 (convert) -> 5 (eval), as committed
# (the manifests already default to the short validation run). Step 2 (staging) runs only if
# L7_STAGE=1: its jobs re-download ~100 GB.
l7_dreamzero() {
  local d="${REPO_ROOT}/examples/dreamzero"
  kubectl delete rayjob dreamzero-sft -n "$NAMESPACE" --ignore-not-found --wait=true >/dev/null 2>&1 || true
  l7_cleanup dreamzero-sft dreamzero-convert dreamzero-eval
  if [[ "${L7_STAGE:-0}" == 1 ]]; then
    l7_cleanup model-download-dreamzero generate-metadata-dreamzero
    l7_run_steps dreamzero stage "$d/README.md" "2. Stage" || return 1
  fi
  l7_run_steps dreamzero sft "$d/README.md" "3. Multi-node SFT" || return 1
  local st
  st=$(kubectl get rayjob dreamzero-sft -n "$NAMESPACE" -o jsonpath='{.status.jobStatus}' 2>/dev/null)
  if [[ "$st" != SUCCEEDED ]]; then
    log_fail "L7 dreamzero/sft: RayJob status '${st}' (expected SUCCEEDED)"
    sed -i.bak '$ s/\tPASS\t/\tFAIL\t/' "$L7_RESULTS" && rm -f "${L7_RESULTS}.bak"
    return 1
  fi
  l7_run_steps dreamzero convert "$d/README.md" "4. Convert" || return 1
  l7_run_steps dreamzero eval "$d/README.md" "5. LIBERO simulator eval" || return 1
  grep -aq "eval/success_once" "${L7_LOG_DIR}/dreamzero-eval.log" || { log_fail "L7 dreamzero/eval: no eval/success_once"; return 1; }
}

# IsaacLab: README "Run it" steps 2-3 (stage + smoke, 1 GPU) and step 4 (train, 4 GPUs: g7e.24xlarge /
# g6e.24xlarge). Steps 0-1 are the deploy + image build (L1 with PROFILE=isaaclab-gr00t). Needs the
# gpu-sim pool and the isaaclab-gr00t-latest image. L7_ISAACLAB_TRAIN_TIMEOUT (default 90m) bounds
# the train step, including the wait for a 4-GPU node.
l7_isaaclab-gr00t-ppo() {
  local d="${REPO_ROOT}/examples/isaaclab-gr00t-ppo" rc=0
  l7_cleanup model-download-isaaclab-gr00t isaaclab-smoke rlinf-isaaclab-gr00t
  l7_run_steps isaaclab-gr00t-ppo smoke "$d/README.md" --from '# 2. Stage' --to '# 4. PPO' "## Run it" || rc=1
  if (( rc == 0 )) && ! grep -aq '"render_ok": true' "${L7_LOG_DIR}/isaaclab-gr00t-ppo-smoke.log"; then
    log_fail "L7 isaaclab-gr00t-ppo/smoke: render_ok is not true"
    sed -i.bak '$ s/\tPASS\t/\tFAIL\t/' "$L7_RESULTS" && rm -f "${L7_RESULTS}.bak"
    rc=1
  fi
  l7_cleanup isaaclab-smoke
  (( rc == 0 )) || return $rc
  l7_smoke
  STEP_TIMEOUT="${L7_ISAACLAB_TRAIN_TIMEOUT:-90m}" \
    l7_run_steps isaaclab-gr00t-ppo train "$d/README.md" --from '# 4. PPO' "## Run it" || rc=1
  unset SMOKE_FILTER
  if (( rc != 0 )) && kubectl get pods -n "$NAMESPACE" -l job-name=rlinf-isaaclab-gr00t --no-headers 2>/dev/null | grep -q Pending; then
    log_warn "L7 isaaclab-gr00t-ppo/train: pod still Pending (no 4-GPU g7e/g6e capacity)"
  fi
  l7_cleanup rlinf-isaaclab-gr00t
  return $rc
}

# Showcase video (README "Regenerating the Showcase Video", verbatim) for ManiSkill / pi0.
# Needs a checkpoint: L7_VIDEO=1 first trains 1 step with save_interval=1 (the smoke edit with
# a save), then runs the README block with <N>=1. The regenerated asset is restored afterwards
# (it came from a 1-step checkpoint).
l7_video() {
  local ex="$1" job="$2" d="${REPO_ROOT}/examples/$1" rc=0 readme tmpd
  l7_cleanup "$job"
  tmpd=$(mktemp -d "${TMPDIR:-/tmp}/l7-readme.XXXXXX")   # BSD mktemp keeps XXXXXX literal before a suffix
  readme="${tmpd}/README.md"
  sed 's/global_step_<N>/global_step_1/g' "$d/README.md" > "$readme"
  export SMOKE_EXTRA="runner.max_steps=1 runner.save_interval=1"
  l7_smoke
  L7_README_LABEL="examples/$ex/README.md (global_step_<N> = 1)" l7_run_steps "$ex" ckpt "$readme" "2. Launch" "3. Monitor" || rc=1
  unset SMOKE_FILTER SMOKE_EXTRA
  l7_cleanup "$job"
  (( rc == 0 )) && { L7_README_LABEL="examples/$ex/README.md (global_step_<N> = 1)" l7_run_steps "$ex" video "$readme" "Regenerating the Showcase Video" || rc=1; }
  rm -rf "$tmpd"
  git -C "$REPO_ROOT" checkout -- "examples/$ex/assets/" 2>/dev/null || true
  return $rc
}

# =============================================================================
# L7: Example READMEs (customer path)
#   L7_EXAMPLES  space-separated subset of: ${L7_ALL} (default: all; --example NAME also works)
#   L7_VIDEO=1   also regenerate the ManiSkill / pi0 showcase videos
#   L7_STAGE=1   also run DreamZero's staging jobs (README step 2)
#   IMAGE / IMAGE_TAG pick the image (IsaacLab defaults to isaaclab-gr00t-latest)
# =============================================================================
run_level_7() {
  log_header "L7: Example READMEs (customer path)"
  detect_cluster_config
  export NAMESPACE="${NAMESPACE:-rlinf}"
  L7_LOG_DIR="${L7_LOG_DIR:-${REPO_ROOT}/validate-logs/L7-$(date -u +%Y%m%dT%H%M%S)}"
  L7_RESULTS="${L7_RESULTS:-${L7_LOG_DIR}/results.tsv}"
  mkdir -p "$L7_LOG_DIR"
  L7_K8S_VERSION=$(kubectl version -o json 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["serverVersion"]["gitVersion"])' 2>/dev/null || echo -)
  export L7_LOG_DIR L7_RESULTS L7_K8S_VERSION
  [[ -f "$L7_RESULTS" ]] || printf 'example\tstep\tstatus\tseconds\timage\tinstance\tk8s\tstep_time_s\n' > "$L7_RESULTS"
  log_info "Logs + results: ${L7_LOG_DIR}"

  local list="${L7_EXAMPLES:-${EXAMPLE_FILTER:-$L7_ALL}}" ex fail=0 saved_image="${IMAGE:-}"
  for ex in $list; do
    if ! declare -F "l7_${ex}" >/dev/null; then
      log_fail "L7: unknown example '${ex}' (valid: ${L7_ALL})"; fail=1; continue
    fi
    if [[ $ex == isaaclab-gr00t-ppo && -z "${IMAGE_TAG:-}" && -n "${ECR_URI:-}" ]]; then
      export IMAGE="${ECR_URI}:isaaclab-gr00t-latest"
    else
      export IMAGE="$saved_image"
    fi
    "l7_${ex}" || fail=1
    if [[ "${L7_VIDEO:-0}" == 1 ]]; then
      case $ex in
        maniskill-openvlaoft-ppo) export LORA_PATH=/fsx/models/rlinf-openvlaoft-maniskill-base-lora
                                  l7_video "$ex" rlinf-maniskill-openvlaoft || fail=1 ;;
        libero-pi0-ppo)           l7_video "$ex" rlinf-libero-pi0 || fail=1 ;;
      esac
    fi
  done
  export IMAGE="$saved_image"
  log_info "L7 results (${L7_RESULTS}):"
  column -t -s $'\t' "$L7_RESULTS" 2>/dev/null || cat "$L7_RESULTS"
  return $fail
}
