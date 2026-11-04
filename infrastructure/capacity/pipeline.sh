#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Unattended capacity pipeline for on-demand GPU fishing:
#   1. catch    hold-sim.sh: 1-GPU placeholder for any type in SIM_TYPES, any allowed AZ
#   2. keep     keep-node.sh: CPU-only keeper on the caught node (stops consolidation)
#   3. FSx      fsx-move.sh if the node is not in FSx's AZ (needs AUTO_FSX_MOVE=1; unset = release such nodes)
#   4. queue    every queued run whose type regex matches the node, sized to the node
#   5. release  delete the keeper when nothing left matches; catch again if runs remain
# Run it detached (nohup ... &) so it survives the launching shell.
#
# Queue file (QUEUE_FILE, one run per line, '#' comments):
#   name|gpus|max_steps|val_check_interval|families (comma-separated, e.g. g6e,g7e)|extra hydra overrides
#   g6e-4gpu-b64|4|2|-1|g6e|actor.micro_batch_size=64
#   smoke|1|0|0|g6e,g7e|manifest=examples/isaaclab-gr00t-ppo/manifests/isaaclab-smoke.yaml
#   (extra=manifest=<path>: render that Job for the held node; gpus/steps/val are ignored)
# Finished runs are appended to $WORK_DIR/pipeline.done and skipped on restart.
#
# env: IMAGE (required), QUEUE_FILE (required), MAX_HOURS (default 20), plus common.sh
#      and fsx-move.sh settings. Log: $WORK_DIR/pipeline.log
set -uo pipefail
# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"
: "${IMAGE:?set IMAGE}" "${QUEUE_FILE:?set QUEUE_FILE}"
DONE=$WORK_DIR/pipeline.done; touch "$DONE"
QUEUE=()   # bash 3.2 (macOS) has no mapfile
while IFS= read -r line; do QUEUE+=("$line"); done < <(grep -vE '^[[:space:]]*(#|$)' "$QUEUE_FILE")
[ "${#QUEUE[@]}" -gt 0 ] || { echo "empty queue: $QUEUE_FILE" >&2; exit 1; }
pending() { local n=0 q; for q in "${QUEUE[@]}"; do grep -qx "${q%%|*}" "$DONE" || n=$((n+1)); done; echo $n; }
jobstate() {
  local s; s=$(k get job -n "$NAMESPACE" "$1" -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type}{" "}{end}' 2>&1) \
    || { case "$s" in *NotFound*) echo Missing;; *) echo Error;; esac; return; }
  case "$s" in *Complete*) echo Complete;; *Failed*) echo Failed;; *) echo Running;; esac
}
end=$(( $(date +%s) + ${MAX_HOURS:-20} * 3600 ))
note "pipeline start, $(pending) runs pending (queue: $QUEUE_FILE)"
while [ "$(pending)" -gt 0 ] && [ "$(date +%s)" -lt "$end" ]; do
  # 1-2. catch and keep
  if ! k get pod -n "$NAMESPACE" keep-sim >/dev/null 2>&1; then
    k get pod -n "$NAMESPACE" hold-sim >/dev/null 2>&1 || { bash "$CAP_DIR/hold-sim.sh" >/dev/null && note "hold-sim applied ($SIM_TYPES)"; }
    [ -n "$(k get pod -n "$NAMESPACE" hold-sim -o jsonpath='{.spec.nodeName}' 2>/dev/null)" ] || { sleep 120; continue; }
    bash "$CAP_DIR/keep-node.sh" >> "$WORK_DIR/pipeline.log" 2>&1 || { sleep 60; continue; }
  fi
  NODE=$(k get pod -n "$NAMESPACE" keep-sim -o jsonpath='{.spec.nodeName}')
  ITYPE=$(k get node "$NODE" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}')
  ZONE=$(k get node "$NODE" -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}')
  note "holding $NODE $ITYPE $ZONE"
  # 3. FSx follows the node. The storage output fsx_az says where FSx is; also check
  # that the claim exists (a failed storage apply can leave FSx re-created but no PVC).
  FSZ=$(terraform -chdir="$REPO_ROOT/infrastructure/storage" output -raw fsx_az 2>/dev/null)
  if ! k get pvc -n "$NAMESPACE" fsx-claim >/dev/null 2>&1; then
    note "fsx-claim missing (FSx not usable); stopping for manual repair"
    bash "$CAP_DIR/keep-node.sh" release >/dev/null; exit 1
  fi
  if [ -n "$FSZ" ] && [ "$ZONE" != "$FSZ" ]; then
    if [ "${AUTO_FSX_MOVE:-0}" != 1 ]; then
      # unattended AZ moves disabled: never run on a node outside FSx's AZ (cross-AZ FSx)
      note "node in $ZONE but FSx in $FSZ and AUTO_FSX_MOVE unset; releasing node"
      bash "$CAP_DIR/keep-node.sh" release >/dev/null; sleep 300; continue
    fi
    note "FSx in $FSZ, node in $ZONE: moving"
    # A failed move can leave Terraform state, the PVC or FSx half-replaced: stop for a
    # human instead of looping (2026-11-01: an expired token mid storage apply left FSx
    # created but out of state and no PVC; the next runs hit their 4 h deadline).
    bash "$CAP_DIR/fsx-move.sh" "$ZONE" || { note "FSx move failed; releasing node and stopping for manual repair"; bash "$CAP_DIR/keep-node.sh" release >/dev/null; exit 1; }
  fi
  if k get job -n "$NAMESPACE" fsx-hydrate >/dev/null 2>&1; then
    k wait --for=condition=complete job/fsx-hydrate -n "$NAMESPACE" --timeout=2400s >/dev/null 2>&1 && note "FSx prefetch complete"
  fi
  # 4. queue
  ran=0
  for q in "${QUEUE[@]}"; do
    IFS='|' read -r name g steps val types extra <<< "$q"
    grep -qx "$name" "$DONE" && continue
    [[ "$ITYPE" =~ ^(${types//,/|})\. ]] || continue   # types: comma-separated families
    if [[ "$extra" == manifest=* ]]; then   # any single-pod Job manifest (smoke test, L2)
      # The gpu-sim pool attaches no EFA: drop EFA requests there (NO_EFA=1)
      nodeefa=$(k get node "$NODE" -o jsonpath='{.status.allocatable.vpc\.amazonaws\.com/efa}' 2>/dev/null)
      NO_EFA=$([ -z "$nodeefa" ] || [ "$nodeefa" = 0 ] && echo 1 || echo 0) \
      bash "$CAP_DIR/manifest-on-held-node.sh" "$name" "$REPO_ROOT/${extra#manifest=}" > "$WORK_DIR/$name.yaml" 2>>"$WORK_DIR/pipeline.log" || continue
    else
      FORCE_GPUS=$g bash "$CAP_DIR/run-on-held-node.sh" "$name" "$steps" "$val" "$extra" > "$WORK_DIR/$name.yaml" 2>>"$WORK_DIR/pipeline.log" || continue
    fi
    [ "$(jobstate "$name")" = Missing ] && k apply -f "$WORK_DIR/$name.yaml" >/dev/null && note "$name applied on $ITYPE"
    while :; do
      st=$(jobstate "$name"); case "$st" in Complete|Failed) break;; esac
      # Unschedulable for 15 min (e.g. missing PVC) means something is broken: stop.
      ps=$(k get pod -n "$NAMESPACE" -l job-name="$name" -o jsonpath='{.items[0].status.phase} {.items[0].metadata.creationTimestamp}' 2>/dev/null)
      if [ "${ps%% *}" = Pending ] && [ $(( $(date +%s) - $(date -j -u -f %Y-%m-%dT%H:%M:%SZ "${ps#* }" +%s 2>/dev/null || date -u -d "${ps#* }" +%s) )) -gt 900 ]; then
        note "$name pending > 15 min on a held node; deleting it and stopping"; k delete job -n "$NAMESPACE" "$name" --wait=false >/dev/null
        bash "$CAP_DIR/keep-node.sh" release >/dev/null; exit 1
      fi
      sleep 60
    done
    P=$(k get pod -n "$NAMESPACE" -l job-name="$name" -o jsonpath='{.items[0].metadata.name}')
    k logs -n "$NAMESPACE" "$P" > "$WORK_DIR/$name.log" 2>&1
    note "$name $st"; echo "$name" >> "$DONE"; ran=1
  done
  # 5. release
  note "queue done for $ITYPE (ran=$ran); releasing $NODE"; bash "$CAP_DIR/keep-node.sh" release >/dev/null
  [ "$ran" = 0 ] && sleep 600
done
note "pipeline end, $(pending) runs pending"
