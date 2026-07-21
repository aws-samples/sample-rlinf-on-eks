#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# common.sh -- Shared utilities for the RLinf-on-EKS validation harness
#
# Provides:
#   - Colored logging (log_pass, log_fail, log_info, log_warn)
#   - Timer functions (timer_start, timer_elapsed)
#   - Result tracking (result_add, result_summary)
#   - Kubernetes helpers (wait_for_job, parse_job_logs, cleanup_job)
# =============================================================================

# --- Color codes (disabled if NO_COLOR is set or stdout is not a tty) ---
if [[ -z "${NO_COLOR:-}" ]] && [[ -t 1 ]]; then
  _RED='\033[0;31m'
  _GREEN='\033[0;32m'
  _YELLOW='\033[0;33m'
  _CYAN='\033[0;36m'
  _BOLD='\033[1m'
  _RESET='\033[0m'
else
  _RED='' _GREEN='' _YELLOW='' _CYAN='' _BOLD='' _RESET=''
fi

# --- Logging ---
log_pass() { echo -e "${_GREEN}PASS${_RESET}: $*"; }
log_fail() { echo -e "${_RED}FAIL${_RESET}: $*"; }
log_info() { echo -e "${_CYAN}INFO${_RESET}: $*"; }
log_warn() { echo -e "${_YELLOW}WARN${_RESET}: $*"; }
log_header() {
  echo ""
  echo -e "${_BOLD}========================================${_RESET}"
  echo -e "${_BOLD}  $*${_RESET}"
  echo -e "${_BOLD}========================================${_RESET}"
}

# --- Timers ---
# Usage: timer_start; ... ; elapsed=$(timer_elapsed)
_TIMER_START=0

timer_start() {
  _TIMER_START=$(date +%s)
}

timer_elapsed() {
  local now
  now=$(date +%s)
  echo $(( now - _TIMER_START ))
}

timer_format() {
  local secs=$1
  if (( secs >= 3600 )); then
    printf '%dh%02dm%02ds' $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
  elif (( secs >= 60 )); then
    printf '%dm%02ds' $((secs/60)) $((secs%60))
  else
    printf '%ds' "$secs"
  fi
}

# --- Result Tracking ---
# Stores results as lines in a temp file: "level|name|status|time"
_RESULTS_FILE=""

results_init() {
  _RESULTS_FILE=$(mktemp /tmp/validate-results.XXXXXX)
}

result_add() {
  local level="$1" name="$2" status="$3" elapsed="$4"
  echo "${level}|${name}|${status}|${elapsed}" >> "$_RESULTS_FILE"
}

results_summary() {
  local total=0 passed=0 failed=0 total_time=0

  log_header "VALIDATION RESULTS"
  printf "  ${_BOLD}%-7s %-24s %-8s %s${_RESET}\n" "Level" "Name" "Status" "Time"
  printf "  %-7s %-24s %-8s %s\n" "-----" "------------------------" "------" "----"

  while IFS='|' read -r level name status elapsed; do
    total=$((total + 1))
    total_time=$((total_time + elapsed))
    local formatted
    formatted=$(timer_format "$elapsed")
    if [[ "$status" == "PASS" ]]; then
      passed=$((passed + 1))
      printf "  %-7s %-24s ${_GREEN}%-8s${_RESET} %s\n" "$level" "$name" "$status" "$formatted"
    elif [[ "$status" == "SKIP" ]]; then
      printf "  %-7s %-24s ${_YELLOW}%-8s${_RESET} %s\n" "$level" "$name" "$status" "$formatted"
    else
      failed=$((failed + 1))
      printf "  %-7s %-24s ${_RED}%-8s${_RESET} %s\n" "$level" "$name" "$status" "$formatted"
    fi
  done < "$_RESULTS_FILE"

  echo -e "  ${_BOLD}========================================${_RESET}"
  local total_formatted
  total_formatted=$(timer_format "$total_time")
  if (( failed > 0 )); then
    echo -e "  ${_RED}${passed}/${total} passed${_RESET} in ${total_formatted}"
  else
    echo -e "  ${_GREEN}${passed}/${total} passed${_RESET} in ${total_formatted}"
  fi
  echo ""

  # Cleanup
  rm -f "$_RESULTS_FILE"

  return "$failed"
}

# --- Kubernetes Helpers ---

# wait_for_job JOB_NAME NAMESPACE TIMEOUT_SECONDS
#   Waits for a K8s Job to complete (succeed or fail).
#   Returns 0 on success, 1 on failure/timeout.
wait_for_job() {
  local job_name="$1"
  local namespace="${2:-default}"
  local timeout="${3:-300}"

  log_info "Waiting for job/${job_name} (timeout: ${timeout}s)..."

  # First wait for the pod to be created
  local elapsed=0
  while (( elapsed < timeout )); do
    if kubectl get pods -n "$namespace" -l "job-name=${job_name}" --no-headers 2>/dev/null | grep -q .; then
      break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done

  if (( elapsed >= timeout )); then
    log_fail "Timed out waiting for pod creation for job/${job_name}"
    return 1
  fi

  # Now wait for completion
  if kubectl wait --for=condition=complete "job/${job_name}" \
       -n "$namespace" --timeout="${timeout}s" 2>/dev/null; then
    log_info "job/${job_name} completed successfully"
    return 0
  fi

  # Check if it failed (not just timed out)
  local status
  status=$(kubectl get job "$job_name" -n "$namespace" \
    -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)
  if [[ "$status" == "True" ]]; then
    log_fail "job/${job_name} failed"
    # Capture logs for debugging
    local pod
    pod=$(kubectl get pods -n "$namespace" -l "job-name=${job_name}" \
      --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null)
    if [[ -n "$pod" ]]; then
      log_info "--- Last 50 lines of pod/${pod} ---"
      kubectl logs "$pod" -n "$namespace" --tail=50 2>/dev/null || true
      echo "---"
    fi
    return 1
  fi

  log_fail "job/${job_name} timed out after ${timeout}s"
  return 1
}

# wait_for_mpijob MPIJOB_NAME NAMESPACE TIMEOUT_SECONDS
#   Waits for a Kubeflow MPIJob's launcher pod to complete.
wait_for_mpijob() {
  local name="$1"
  local namespace="${2:-default}"
  local timeout="${3:-600}"

  log_info "Waiting for MPIJob/${name} launcher (timeout: ${timeout}s)..."

  local elapsed=0
  local launcher_pod=""

  # Wait for launcher pod to appear
  while (( elapsed < timeout )); do
    launcher_pod=$(kubectl get pods -n "$namespace" \
      -l "training.kubeflow.org/job-name=${name},training.kubeflow.org/job-role=launcher" \
      --no-headers -o name 2>/dev/null | head -1)
    if [[ -n "$launcher_pod" ]]; then
      break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done

  if [[ -z "$launcher_pod" ]]; then
    log_fail "Timed out waiting for MPIJob/${name} launcher pod"
    return 1
  fi

  # Wait for launcher to finish by polling phase
  while (( elapsed < timeout )); do
    local phase
    phase=$(kubectl get "$launcher_pod" -n "$namespace" -o jsonpath='{.status.phase}' 2>/dev/null)
    if [[ "$phase" == "Succeeded" ]]; then
      log_info "MPIJob/${name} launcher completed successfully"
      return 0
    elif [[ "$phase" == "Failed" ]]; then
      local exit_code
      exit_code=$(kubectl get "$launcher_pod" -n "$namespace" \
        -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null)
      log_fail "MPIJob/${name} launcher failed (exit code: ${exit_code:-unknown})"
      return 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done

  log_fail "Timed out waiting for MPIJob/${name} launcher to complete"
  return 1
}

# get_job_logs JOB_NAME [NAMESPACE]
#   Returns the full logs from a job's pod.
get_job_logs() {
  local job_name="$1"
  local namespace="${2:-default}"
  local pod
  pod=$(kubectl get pods -n "$namespace" -l "job-name=${job_name}" \
    --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null)
  if [[ -n "$pod" ]]; then
    kubectl logs "$pod" -n "$namespace" 2>/dev/null
  fi
}

# get_mpijob_launcher_logs MPIJOB_NAME [NAMESPACE]
#   Returns the full logs from an MPIJob's launcher pod.
get_mpijob_launcher_logs() {
  local name="$1"
  local namespace="${2:-default}"
  local pod
  pod=$(kubectl get pods -n "$namespace" \
    -l "training.kubeflow.org/job-name=${name},training.kubeflow.org/job-role=launcher" \
    --no-headers -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [[ -n "$pod" ]]; then
    kubectl logs "$pod" -n "$namespace" 2>/dev/null
  fi
}

# cleanup_job JOB_NAME [NAMESPACE]
#   Deletes a job and waits for pod cleanup.
cleanup_job() {
  local job_name="$1"
  local namespace="${2:-default}"
  kubectl delete job "$job_name" -n "$namespace" --ignore-not-found=true --wait=true 2>/dev/null || true
}

# cleanup_mpijob MPIJOB_NAME [NAMESPACE]
cleanup_mpijob() {
  local name="$1"
  local namespace="${2:-default}"
  kubectl delete mpijob "$name" -n "$namespace" --ignore-not-found=true --wait=true 2>/dev/null || true
}

# require_cmd CMD
#   Checks that a command is available; logs fail and returns 1 if not.
require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" &>/dev/null; then
    log_fail "Required command not found: $cmd"
    return 1
  fi
  return 0
}

# require_cluster
#   Checks kubectl connectivity to the cluster.
require_cluster() {
  if ! kubectl cluster-info &>/dev/null; then
    log_fail "Cannot reach Kubernetes cluster. Run: aws eks update-kubeconfig ..."
    return 1
  fi
  return 0
}
