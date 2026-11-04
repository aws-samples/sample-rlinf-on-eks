#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Cast a wide net for an RT-core sim node: a 1-GPU placeholder pod (hold-sim) that
# Karpenter can satisfy with any instance type in SIM_TYPES, in any AZ the gpu-sim
# NodePool allows. Nothing runs on it; it only makes Karpenter keep retrying on-demand
# launches until one succeeds. Pair with keep-node.sh once it lands.
#
# usage: hold-sim.sh [apply|delete]
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"
case "${1:-apply}" in
  delete) k delete pod -n "$NAMESPACE" hold-sim --ignore-not-found; exit 0;;
esac
types="[${SIM_TYPES//,/, }]"
cat <<EOF | k apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: hold-sim
  namespace: ${NAMESPACE}
  labels: {app: rlinf-hold}
  annotations: {karpenter.sh/do-not-disrupt: "true"}
spec:
  restartPolicy: Never
  activeDeadlineSeconds: 86400
  nodeSelector: {role: gpu-sim}
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - {key: node.kubernetes.io/instance-type, operator: In, values: ${types}}
  tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]
  containers:
    - name: hold
      image: public.ecr.aws/docker/library/busybox:1.36
      command: ["sleep", "86400"]
      resources:
        requests: {cpu: 100m, memory: 64Mi, nvidia.com/gpu: 1}
        limits: {nvidia.com/gpu: 1}
EOF
