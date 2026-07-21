#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# validate.sh -- RLinf-on-EKS Validation Harness
#
# End-to-end validation from static checks to training step tests.
# Two modes:
#   local   - No cluster required. Validates code, formatting, manifests.
#   cluster - Requires live EKS cluster. Runs levels L0-L6.
#
# Usage:
#   ./validate.sh --mode local
#   ./validate.sh --mode cluster --level 3
#   ./validate.sh --mode cluster --level 5 --skip-to 2 --example maniskill-openvla
#   ./validate.sh --mode cluster --continue-on-error
#
# Levels (cluster mode):
#   L0: Infrastructure smoke test (GPU, EFA, FSx)
#   L1: Container build + ECR push
#   L2: Container validation (10 tests)
#   L3: NCCL/EFA bandwidth threshold
#   L4: Model download verification
#   L5: Training step tests (1 PPO step per example)
#   L6: Multi-node training step
# =============================================================================
set -euo pipefail

# Resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$SCRIPT_DIR"

# Infrastructure paths
export INFRA_DIR="${REPO_ROOT}/infrastructure"
export TF_DIR="${REPO_ROOT}/infrastructure/cluster"
export MANIFESTS_DIR="${REPO_ROOT}/infrastructure/manifests"

# Source libraries
# shellcheck source=tests/lib/common.sh
source "${REPO_ROOT}/tests/lib/common.sh"
# shellcheck source=tests/lib/local.sh
source "${REPO_ROOT}/tests/lib/local.sh"
# shellcheck source=tests/lib/cluster.sh
source "${REPO_ROOT}/tests/lib/cluster.sh"

# --- Defaults ---
MODE=""
MAX_LEVEL=6
SKIP_TO=0
EXAMPLE_FILTER=""
CONTINUE_ON_ERROR=0

# --- Usage ---
usage() {
  cat <<EOF
Usage: $0 --mode <local|cluster> [OPTIONS]

Modes:
  local     Run offline checks (terraform, kubeconform, shellcheck, integrity)
  cluster   Run live cluster validation levels L0-L6

Options:
  --level N            Run up to level N (default: 6, cluster mode only)
  --skip-to N          Start from level N (skip earlier levels)
  --example NAME       Test a specific example in L5 (maniskill-openvla, maniskill-openvlaoft, libero-pi0)
  --continue-on-error  Don't stop on first failure
  -h, --help           Show this help

Environment Variables:
  ECR_URI              ECR image URI (auto-detected from terraform if not set)
  CODEBUILD_PROJECT    CodeBuild project name (auto-detected)
  CODEBUILD_BUCKET     S3 bucket for CodeBuild source (auto-detected)
  NCCL_TEST_IMAGE      NCCL test container image (default: public ECR)
  NO_COLOR             Disable colored output
EOF
  exit 0
}

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)         MODE="$2"; shift 2 ;;
    --level)        MAX_LEVEL="$2"; shift 2 ;;
    --skip-to)      SKIP_TO="$2"; shift 2 ;;
    --example)      EXAMPLE_FILTER="$2"; shift 2 ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    -h|--help)      usage ;;
    *)              echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Error: --mode is required"
  usage
fi

# Export for cluster.sh L5
export EXAMPLE_FILTER

# =============================================================================
# Local Mode
# =============================================================================
run_mode_local() {
  log_header "RLinf-on-EKS Local Validation"
  results_init

  timer_start
  if run_terraform_checks; then
    result_add "Local" "Terraform" "PASS" "$(timer_elapsed)"
  else
    result_add "Local" "Terraform" "FAIL" "$(timer_elapsed)"
  fi

  timer_start
  if run_manifest_checks; then
    result_add "Local" "K8s Manifests" "PASS" "$(timer_elapsed)"
  else
    result_add "Local" "K8s Manifests" "FAIL" "$(timer_elapsed)"
  fi

  timer_start
  if run_shellcheck; then
    result_add "Local" "Shell Scripts" "PASS" "$(timer_elapsed)"
  else
    result_add "Local" "Shell Scripts" "FAIL" "$(timer_elapsed)"
  fi

  timer_start
  if run_reference_integrity; then
    result_add "Local" "Reference Integrity" "PASS" "$(timer_elapsed)"
  else
    result_add "Local" "Reference Integrity" "FAIL" "$(timer_elapsed)"
  fi

  results_summary
  return $?
}

# =============================================================================
# Cluster Mode
# =============================================================================
run_mode_cluster() {
  log_header "RLinf-on-EKS Cluster Validation (L0-L${MAX_LEVEL})"

  # Verify cluster connectivity
  if ! require_cluster; then
    exit 1
  fi

  results_init
  local failed=0

  # Level definitions: number, name, function
  local -a levels=(
    "0|Infrastructure|run_level_0"
    "1|Container Build|run_level_1"
    "2|Container Test|run_level_2"
    "3|NCCL/EFA|run_level_3"
    "4|Model Download|run_level_4"
    "5|Training Steps|run_level_5"
    "6|Multi-Node|run_level_6"
  )

  for level_def in "${levels[@]}"; do
    IFS='|' read -r level_num level_name level_func <<< "$level_def"

    # Skip if below skip-to
    if (( level_num < SKIP_TO )); then
      log_info "Skipping L${level_num} (--skip-to ${SKIP_TO})"
      continue
    fi

    # Stop if above max level
    if (( level_num > MAX_LEVEL )); then
      break
    fi

    # L5 adds sub-results internally; we still track overall L5
    timer_start
    local rc=0
    "$level_func" || rc=$?

    local elapsed
    elapsed=$(timer_elapsed)

    if (( rc == 0 )); then
      # L5 adds its own per-example results, but also record the overall
      if [[ "$level_func" != "run_level_5" ]]; then
        result_add "L${level_num}" "$level_name" "PASS" "$elapsed"
      fi
    elif (( rc == 2 )); then
      # Skip (e.g., L6 with < 2 nodes)
      result_add "L${level_num}" "$level_name" "SKIP" "$elapsed"
    else
      if [[ "$level_func" != "run_level_5" ]]; then
        result_add "L${level_num}" "$level_name" "FAIL" "$elapsed"
      fi
      failed=1
      if (( ! CONTINUE_ON_ERROR )); then
        log_warn "Stopping at L${level_num}. Use --continue-on-error to continue."
        break
      fi
    fi
  done

  results_summary
  return $?
}

# =============================================================================
# Main
# =============================================================================
case "$MODE" in
  local)   run_mode_local ;;
  cluster) run_mode_cluster ;;
  *)       echo "Unknown mode: $MODE"; usage ;;
esac
