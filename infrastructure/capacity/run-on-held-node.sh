#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Render the IsaacLab GR00T training Job for the node keep-sim is holding, sized to
# that node: hostname selector; GPU count from the node (or FORCE_GPUS); CPU/memory =
# 85% of allocatable, scaled by GPUs used. Placement: 4 GPUs = env/rollout/actor on
# 0-3; 8 GPUs = env 0-3 / rollout 4-7 / actor 0-7. Prints the manifest.
#
# usage: run-on-held-node.sh <run-name> <max_steps> <val_check_interval> ["<extra hydra overrides>"]
# env:   IMAGE (required), FORCE_GPUS, GR00T_ATTN=sdpa (mount the SDPA fallback patch)
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"
RUN=$1; STEPS=$2; VAL=$3; EXTRA=${4:-}
: "${IMAGE:?set IMAGE=<ecr repo>:<tag>}"
NODE=$(k get pod -n "$NAMESPACE" keep-sim -o jsonpath='{.spec.nodeName}')
[ -n "$NODE" ] || { echo "no held node (keep-sim not scheduled)" >&2; exit 1; }
INFO=$(k get node "$NODE" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type} {.status.allocatable.nvidia\.com/gpu} {.status.allocatable.cpu} {.status.allocatable.memory}')
read -r ITYPE GPUS CPU MEM <<< "$INFO" || true
case "$CPU" in *m) CPU_M=${CPU%m};; *) CPU_M=$(( CPU * 1000 ));; esac
MEM_KI=${MEM%Ki}
[ -n "${GPUS:-}" ] && [ "$GPUS" -gt 0 ] || { echo "node $NODE has no allocatable GPUs yet: [$INFO]" >&2; exit 1; }
USE=${FORCE_GPUS:-$GPUS}
CPU_REQ=$(( CPU_M * 85 * USE / (100 * 1000 * GPUS) ))
MEM_REQ=$(( MEM_KI * 85 * USE / (100 * 1024 * 1024 * GPUS) ))
if [ "$USE" -ge 8 ]; then
  PLACE="cluster.component_placement.env=0-3 cluster.component_placement.rollout.placement=4-7 cluster.component_placement.actor=0-7"; NG=8
else
  PLACE="cluster.component_placement.env=0-3 cluster.component_placement.rollout.placement=0-3 cluster.component_placement.actor=0-3"; NG=4
fi
echo "# node=$NODE type=$ITYPE gpus=$GPUS -> using $NG GPU, ${CPU_REQ} CPU, ${MEM_REQ}Gi" >&2
export IMAGE NAMESPACE
# shellcheck disable=SC2016  # envsubst takes the literal variable names
envsubst '${IMAGE} ${NAMESPACE}' < "$REPO_ROOT/examples/isaaclab-gr00t-ppo/manifests/isaaclab-gr00t-ppo.yaml" | \
python3 -c '
import os, sys
run, node, ng, cpu, mem, place, steps, val, extra = sys.argv[1:10]
s = sys.stdin.read()
ov = f"{place} runner.max_steps={steps} runner.val_check_interval={val} runner.save_interval=-1 runner.logger.experiment_name={run} {extra}".strip()
rep = [
    ("name: rlinf-isaaclab-gr00t\n", f"name: {run}\n"),
    # pin to the held node instead of the instance-type list
    ("      affinity:\n        nodeAffinity:\n          requiredDuringSchedulingIgnoredDuringExecution:\n            nodeSelectorTerms:\n              - matchExpressions:\n                  - key: node.kubernetes.io/instance-type\n                    operator: In\n                    values: [\"g7e.24xlarge\", \"g6e.24xlarge\"]\n", ""),
    ("        role: gpu-sim\n", f"        role: gpu-sim\n        kubernetes.io/hostname: {node}\n"),
    ("\"cluster.component_placement.env=0-3 cluster.component_placement.rollout.placement=0-3 cluster.component_placement.actor=0-3 actor.micro_batch_size=64\"", f"\"{ov}\""),
    ("              value: \"4\"\n            - name: NUM_NODES", f"              value: \"{ng}\"\n            - name: NUM_NODES"),
    ("nvidia.com/gpu: 4", f"nvidia.com/gpu: {ng}"),
    ("cpu: \"80\"", f"cpu: \"{cpu}\""), ("memory: \"600Gi\"", f"memory: \"{mem}Gi\""),
    ("  backoffLimit: 0\n", "  backoffLimit: 0\n  activeDeadlineSeconds: 14400\n"),
]
if os.environ.get("GR00T_ATTN") == "sdpa":
    rep += [
        ("              bash /workspace/eks/scripts/run_training_eks.sh 2>&1 \\\n",
         "              GR00T_ATTN=sdpa bash /opt/patch/gr00t_attn_patch.sh\n              bash /workspace/eks/scripts/run_training_eks.sh 2>&1 \\\n"),
        ("            - name: dshm\n              mountPath: /dev/shm\n",
         "            - name: dshm\n              mountPath: /dev/shm\n            - name: patch\n              mountPath: /opt/patch\n"),
        ("      volumes:\n", "      volumes:\n        - name: patch\n          configMap:\n            name: gr00t-attn-patch\n"),
    ]
for a, b in rep:
    assert a in s, f"manifest anchor not found: {a!r}"
    s = s.replace(a, b)
print(s, end="")
' "$RUN" "$NODE" "$NG" "$CPU_REQ" "$MEM_REQ" "$PLACE" "$STEPS" "$VAL" "$EXTRA"
