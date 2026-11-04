#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Keep the node hold-sim landed on: put a CPU-only keeper pod (keep-sim, no GPU,
# do-not-disrupt) on that node, then delete hold-sim so all GPUs are free for runs.
# The keeper stops Karpenter's empty-node consolidation (gpu-sim consolidateAfter: 10m)
# between runs. Release the node with: keep-node.sh release
#
# usage: keep-node.sh [keep|release]
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"
if [ "${1:-keep}" = release ]; then
  k delete pod -n "$NAMESPACE" keep-sim --ignore-not-found; exit 0
fi
NODE=$(k get pod -n "$NAMESPACE" hold-sim -o jsonpath='{.spec.nodeName}' 2>/dev/null || true)
[ -n "$NODE" ] || { echo "hold-sim not scheduled yet" >&2; exit 2; }
cat <<EOF | k apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: keep-sim
  namespace: ${NAMESPACE}
  labels: {app: rlinf-hold}
  annotations: {karpenter.sh/do-not-disrupt: "true"}
spec:
  restartPolicy: Never
  nodeSelector: {kubernetes.io/hostname: ${NODE}}
  tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]
  containers:
    - name: keep
      image: public.ecr.aws/docker/library/busybox:1.36
      command: ["sleep", "604800"]
      resources: {requests: {cpu: 50m, memory: 32Mi}}
EOF
k wait --for=condition=Ready pod/keep-sim -n "$NAMESPACE" --timeout=120s
k delete pod -n "$NAMESPACE" hold-sim
echo "holding ${NODE} ($(k get node "$NODE" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type} {.metadata.labels.topology\.kubernetes\.io/zone}'))"
