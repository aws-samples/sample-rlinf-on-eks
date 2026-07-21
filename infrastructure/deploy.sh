#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

# -------------------------------------------------------------------
# deploy.sh — Orchestrate layered Terraform infrastructure
# -------------------------------------------------------------------
# Usage:
#   ./deploy.sh --action plan --layer all
#   ./deploy.sh --action apply --layer cluster --profile g6-validation
#   ./deploy.sh --action apply --layer all --profile p5-training
#   ./deploy.sh --action destroy --layer all  (destroys in reverse order)
# -------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYERS=("cluster" "storage" "addons" "workloads" "build")

# Defaults
ACTION="plan"
LAYER="all"
PROFILE=""
AUTO_APPROVE=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --action ACTION    Terraform action: plan, apply, destroy (default: plan)
  --layer LAYER      Layer to target: cluster, storage, addons, workloads, build, all (default: all)
  --profile PROFILE  Tfvars profile from cluster/profiles/ (e.g., g6-validation)
  --auto-approve     Skip confirmation prompts
  -h, --help         Show this help

Examples:
  $(basename "$0") --action plan --layer all
  $(basename "$0") --action apply --layer cluster --profile g6-validation
  $(basename "$0") --action destroy --layer all
EOF
  exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --action) ACTION="$2"; shift 2 ;;
    --layer) LAYER="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --auto-approve) AUTO_APPROVE="-auto-approve"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# Validate action
if [[ ! "$ACTION" =~ ^(plan|apply|destroy)$ ]]; then
  echo "ERROR: Invalid action '$ACTION'. Must be: plan, apply, destroy"
  exit 1
fi

# Validate layer
# shellcheck disable=SC2076
if [[ "$LAYER" != "all" ]] && [[ ! " ${LAYERS[*]} " =~ " ${LAYER} " ]]; then
  echo "ERROR: Invalid layer '$LAYER'. Must be: ${LAYERS[*]} all"
  exit 1
fi

log() { echo "==> $*"; }
err() { echo "ERROR: $*" >&2; exit 1; }

# -------------------------------------------------------------------
# Get outputs from a layer's state
# -------------------------------------------------------------------
get_output() {
  local layer="$1" key="$2"
  terraform -chdir="${SCRIPT_DIR}/${layer}" output -raw "$key" 2>/dev/null || echo ""
}

# -------------------------------------------------------------------
# Initialize a layer
# -------------------------------------------------------------------
init_layer() {
  local layer="$1"
  log "Initializing ${layer}..."
  terraform -chdir="${SCRIPT_DIR}/${layer}" init -input=false
}

# -------------------------------------------------------------------
# Build -var flags for a layer based on upstream outputs
# -------------------------------------------------------------------
build_vars() {
  local layer="$1"
  local vars=()

  # Profile vars (only for cluster layer)
  if [[ "$layer" == "cluster" ]] && [[ -n "$PROFILE" ]]; then
    local profile_file="${SCRIPT_DIR}/cluster/profiles/${PROFILE}.tfvars"
    if [[ -f "$profile_file" ]]; then
      vars+=("-var-file=$profile_file")
    else
      err "Profile not found: $profile_file"
    fi
  fi

  # Cross-layer variable passing
  case "$layer" in
    storage)
      local cluster_name cluster_endpoint cluster_ca vpc_id node_sg subnet_ids region
      cluster_name=$(get_output cluster cluster_name)
      cluster_endpoint=$(get_output cluster cluster_endpoint)
      cluster_ca=$(get_output cluster cluster_certificate_authority_data)
      vpc_id=$(get_output cluster vpc_id)
      node_sg=$(get_output cluster node_security_group_id)
      subnet_ids=$(get_output cluster target_subnet_ids)
      region=$(get_output cluster region)

      [[ -n "$cluster_name" ]] && vars+=("-var=cluster_name=$cluster_name")
      [[ -n "$cluster_endpoint" ]] && vars+=("-var=cluster_endpoint=$cluster_endpoint")
      [[ -n "$cluster_ca" ]] && vars+=("-var=cluster_certificate_authority_data=$cluster_ca")
      [[ -n "$vpc_id" ]] && vars+=("-var=vpc_id=$vpc_id")
      [[ -n "$node_sg" ]] && vars+=("-var=node_security_group_id=$node_sg")
      [[ -n "$subnet_ids" ]] && vars+=("-var=target_subnet_ids=$subnet_ids")
      [[ -n "$region" ]] && vars+=("-var=region=$region")
      ;;
    addons)
      local cluster_name cluster_endpoint cluster_ca node_sg region mlflow_iam_role_arn mlflow_s3_bucket
      cluster_name=$(get_output cluster cluster_name)
      cluster_endpoint=$(get_output cluster cluster_endpoint)
      cluster_ca=$(get_output cluster cluster_certificate_authority_data)
      node_sg=$(get_output cluster node_security_group_id)
      region=$(get_output cluster region)
      mlflow_iam_role_arn=$(get_output cluster mlflow_iam_role_arn)
      mlflow_s3_bucket=$(get_output cluster mlflow_s3_bucket)

      [[ -n "$cluster_name" ]] && vars+=("-var=cluster_name=$cluster_name")
      [[ -n "$cluster_endpoint" ]] && vars+=("-var=cluster_endpoint=$cluster_endpoint")
      [[ -n "$cluster_ca" ]] && vars+=("-var=cluster_certificate_authority_data=$cluster_ca")
      [[ -n "$node_sg" ]] && vars+=("-var=node_security_group_id=$node_sg")
      [[ -n "$region" ]] && vars+=("-var=region=$region")
      [[ -n "$mlflow_iam_role_arn" ]] && vars+=("-var=mlflow_iam_role_arn=$mlflow_iam_role_arn")
      [[ -n "$mlflow_s3_bucket" ]] && vars+=("-var=mlflow_s3_bucket=$mlflow_s3_bucket")
      ;;
    workloads)
      local cluster_name cluster_endpoint cluster_ca node_sg target_az region
      cluster_name=$(get_output cluster cluster_name)
      cluster_endpoint=$(get_output cluster cluster_endpoint)
      cluster_ca=$(get_output cluster cluster_certificate_authority_data)
      node_sg=$(get_output cluster node_security_group_id)
      target_az=$(get_output cluster target_az)
      region=$(get_output cluster region)

      [[ -n "$cluster_name" ]] && vars+=("-var=cluster_name=$cluster_name")
      [[ -n "$cluster_endpoint" ]] && vars+=("-var=cluster_endpoint=$cluster_endpoint")
      [[ -n "$cluster_ca" ]] && vars+=("-var=cluster_certificate_authority_data=$cluster_ca")
      [[ -n "$node_sg" ]] && vars+=("-var=node_security_group_id=$node_sg")
      [[ -n "$target_az" ]] && vars+=("-var=target_az=$target_az")
      [[ -n "$region" ]] && vars+=("-var=region=$region")
      ;;
    build)
      local region
      region=$(get_output cluster region)
      [[ -n "$region" ]] && vars+=("-var=region=$region")
      ;;
  esac

  echo "${vars[*]:-}"
}

# -------------------------------------------------------------------
# Run terraform action on a layer
# -------------------------------------------------------------------
run_layer() {
  local layer="$1"
  local vars

  log "--- Layer: ${layer} (${ACTION}) ---"
  init_layer "$layer"

  vars=$(build_vars "$layer")

  case "$ACTION" in
    plan)
      # shellcheck disable=SC2086
      terraform -chdir="${SCRIPT_DIR}/${layer}" plan -input=false $vars
      ;;
    apply)
      # shellcheck disable=SC2086
      terraform -chdir="${SCRIPT_DIR}/${layer}" apply -input=false $AUTO_APPROVE $vars
      ;;
    destroy)
      # shellcheck disable=SC2086
      terraform -chdir="${SCRIPT_DIR}/${layer}" destroy -input=false $AUTO_APPROVE $vars
      ;;
  esac
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
main() {
  if [[ "$LAYER" == "all" ]]; then
    if [[ "$ACTION" == "destroy" ]]; then
      # Destroy in reverse order
      for ((i=${#LAYERS[@]}-1; i>=0; i--)); do
        run_layer "${LAYERS[i]}"
      done
    else
      # Apply in forward order
      for layer in "${LAYERS[@]}"; do
        run_layer "$layer"
      done
    fi
  else
    run_layer "$LAYER"
  fi

  log "Done."
}

main
