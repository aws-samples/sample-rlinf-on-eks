#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# local.sh -- Local (offline) validation functions
#
# Runs without a cluster. Validates code quality, formatting, and consistency.
# =============================================================================

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Infrastructure paths
_TF_DIR="${TF_DIR:-${REPO_ROOT}/infrastructure/cluster}"
_INFRA_DIR="${INFRA_DIR:-${REPO_ROOT}/infrastructure}"

# --- Terraform Checks ---
run_terraform_checks() {
  log_header "Local: Terraform Checks"
  local fail=0

  # terraform fmt
  log_info "Checking terraform fmt..."
  if require_cmd terraform; then
    # Check both infrastructure/ and terraform/ if they exist
    local fmt_fail=0
    if ! terraform fmt -check -recursive "${REPO_ROOT}/infrastructure" 2>/dev/null; then
      fmt_fail=1
    fi

    if (( fmt_fail == 0 )); then
      log_pass "terraform fmt -- all files formatted"
    else
      log_fail "terraform fmt -- files need formatting. Run: terraform fmt -recursive"
      fail=1
    fi
  else
    log_warn "Skipping terraform fmt (terraform not installed)"
  fi

  # terraform validate (requires init, best-effort)
  log_info "Checking terraform validate..."
  if [[ -d "${_TF_DIR}/.terraform" ]]; then
    if terraform -chdir="${_TF_DIR}" validate -no-color 2>/dev/null; then
      log_pass "terraform validate -- configuration is valid"
    else
      log_fail "terraform validate -- configuration has errors"
      fail=1
    fi
  else
    log_warn "Skipping terraform validate (.terraform not initialized)"
  fi

  return $fail
}

# --- Kubernetes Manifest Checks ---
run_manifest_checks() {
  log_header "Local: Kubernetes Manifest Checks"
  local fail=0

  if ! require_cmd kubeconform; then
    log_warn "Skipping manifest checks (kubeconform not installed). Install: brew install kubeconform"
    return 0
  fi

  local dirs=(
    "manifests"
    "infrastructure/manifests"
    "tests/manifests"
  )

  local checked=0
  local failed_files=()

  for dir in "${dirs[@]}"; do
    local full_path="${REPO_ROOT}/${dir}"
    if [[ ! -d "$full_path" ]]; then
      continue
    fi

    while IFS= read -r -d '' yaml_file; do
      checked=$((checked + 1))
      # kubeconform: -strict for unknown fields, -ignore-missing-schemas for CRDs (MPIJob)
      if ! kubeconform -strict -ignore-missing-schemas \
           -kubernetes-version 1.31.0 \
           "$yaml_file" 2>/dev/null; then
        failed_files+=("$yaml_file")
      fi
    done < <(find "$full_path" -name '*.yaml' -o -name '*.yml' | tr '\n' '\0')
  done

  if (( ${#failed_files[@]} > 0 )); then
    log_fail "kubeconform -- ${#failed_files[@]}/${checked} files failed:"
    for f in "${failed_files[@]}"; do
      echo "  - ${f#"$REPO_ROOT"/}"
    done
    fail=1
  else
    log_pass "kubeconform -- ${checked} manifests valid"
  fi

  return $fail
}

# --- Shell Script Checks ---
run_shellcheck() {
  log_header "Local: Shell Script Checks"
  local fail=0

  if ! require_cmd shellcheck; then
    log_warn "Skipping shellcheck (not installed). Install: brew install shellcheck"
    return 0
  fi

  local dirs=(
    "examples/scripts"
    "examples/dreamzero/scripts"
    "infrastructure/cluster/scripts"
    "tests/lib"
  )

  local checked=0
  local failed_files=()

  for dir in "${dirs[@]}"; do
    local full_path="${REPO_ROOT}/${dir}"
    if [[ ! -d "$full_path" ]]; then
      continue
    fi

    while IFS= read -r -d '' sh_file; do
      checked=$((checked + 1))
      # Only report errors, not warnings/info
      if ! shellcheck -S error "$sh_file" 2>/dev/null; then
        failed_files+=("$sh_file")
      fi
    done < <(find "$full_path" -name '*.sh' | tr '\n' '\0')
  done

  # Also check validate.sh itself
  if [[ -f "${REPO_ROOT}/validate.sh" ]]; then
    checked=$((checked + 1))
    if ! shellcheck -S error "${REPO_ROOT}/validate.sh" 2>/dev/null; then
      failed_files+=("validate.sh")
    fi
  fi

  if (( ${#failed_files[@]} > 0 )); then
    log_fail "shellcheck -- ${#failed_files[@]}/${checked} files have errors:"
    for f in "${failed_files[@]}"; do
      echo "  - ${f#"$REPO_ROOT"/}"
    done
    fail=1
  else
    log_pass "shellcheck -- ${checked} scripts clean"
  fi

  return $fail
}

# --- Reference Integrity ---
# Checks that file paths mentioned in AGENTS.md actually exist.
run_reference_integrity() {
  log_header "Local: Reference Integrity"
  local fail=0

  local agents_md="${REPO_ROOT}/AGENTS.md"
  if [[ ! -f "$agents_md" ]]; then
    log_warn "AGENTS.md not found, skipping"
    return 0
  fi

  local missing=()
  local checked=0

  # Extract file paths from AGENTS.md that look like repo-relative paths.
  # Only match paths with at least one directory separator (to avoid bare filenames
  # like `pyproject.toml` that refer to external repos, not this one).
  while IFS= read -r path; do
    # Skip URLs, anchors, and patterns with variables
    [[ "$path" =~ ^https?:// ]] && continue
    [[ "$path" =~ \$ ]] && continue
    [[ "$path" =~ \* ]] && continue
    [[ "$path" == "#"* ]] && continue

    # Strip leading ./ or /
    path="${path#./}"
    path="${path#/}"

    # Skip empty or paths without directory separators (bare filenames are likely
    # references to external files, not paths in this repo)
    [[ -z "$path" ]] && continue
    [[ "$path" != */* ]] && continue

    checked=$((checked + 1))
    if [[ ! -e "${REPO_ROOT}/${path}" ]]; then
      missing+=("$path")
    fi
  done < <(grep -oE '`[a-zA-Z][a-zA-Z0-9_./-]+\.(tf|sh|yaml|yml|py|md|json|toml|txt|j2)`' "$agents_md" | tr -d '`' | sort -u)

  # Also check directory references like `references/rlinf/`
  # Only match paths starting with lowercase (exclude external project names like LIBERO/)
  while IFS= read -r path; do
    path="${path%/}"  # strip trailing slash
    [[ -z "$path" ]] && continue
    [[ "$path" =~ \$ ]] && continue
    # Skip paths starting with uppercase (likely external project references)
    [[ "$path" =~ ^[A-Z] ]] && continue
    checked=$((checked + 1))
    if [[ ! -d "${REPO_ROOT}/${path}" ]]; then
      missing+=("${path}/")
    fi
  done < <(grep -oE '`[a-zA-Z][a-zA-Z0-9_/-]+/`' "$agents_md" | tr -d '`' | sort -u)

  if (( ${#missing[@]} > 0 )); then
    log_fail "Reference integrity -- ${#missing[@]} broken paths in AGENTS.md:"
    for m in "${missing[@]}"; do
      echo "  - $m"
    done
    fail=1
  else
    log_pass "Reference integrity -- ${checked} paths verified"
  fi

  return $fail
}

# --- Run All Local Checks ---
run_local_validation() {
  local fail=0

  run_terraform_checks || fail=1
  run_manifest_checks || fail=1
  run_shellcheck || fail=1
  run_reference_integrity || fail=1

  return $fail
}
