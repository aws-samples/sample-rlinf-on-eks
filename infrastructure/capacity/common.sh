#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Shared settings for the capacity scripts (catch -> keep -> run -> release).
# Everything account- or cluster-specific comes from the environment; nothing is hard-coded.
#
#   KUBECONFIG     kubeconfig for the cluster (default: kubectl's default)
#   NAMESPACE      namespace for placeholders, keeper and runs (default: rlinf)
#   IMAGE          training image <ecr repo>:<tag> (required by run-on-held-node.sh)
#   WORK_DIR       rendered manifests, logs and state (default: $TMPDIR/rlinf-capacity)
#   SIM_TYPES      comma-separated instance types for the sim net (default: g6e/g7e sizes)
#   REPO_ROOT      repository root (default: two levels up from this file)

NAMESPACE=${NAMESPACE:-rlinf}
WORK_DIR=${WORK_DIR:-${TMPDIR:-/tmp}/rlinf-capacity}
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SIM_TYPES=${SIM_TYPES:-g6e.12xlarge,g6e.24xlarge,g6e.48xlarge,g7e.24xlarge,g7e.48xlarge}
CAP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CAP_DIR
mkdir -p "$WORK_DIR"

# kubectl with a hard 100 s cap per call (a hung API call must not stall an unattended loop)
k() { perl -e 'alarm 100; exec @ARGV' kubectl --request-timeout=60s "$@"; }
note() { echo "$(date -u +%H:%M:%S) $*" >> "$WORK_DIR/pipeline.log"; }
