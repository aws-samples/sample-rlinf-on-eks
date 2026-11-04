#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Move FSx to $1, where a GPU node was caught: check nothing uses fsx-claim, then
# storage layer (FSx + PV + PVC re-created in fsx_az; it refills from S3 via the
# data-repository association), workloads layer (GPU NodePools pinned to fsx_az), then
# prefetch FSx so the first Isaac Sim boot is not a 20-min lazy load. The cluster layer
# is not touched: system nodes stay in target_az (they do not mount FSx), and no tracked
# file is edited.
#
# usage: fsx-move.sh <az>
# env:   FSX_CAPACITY (GiB, default 1200),
#        WORKLOADS_TF_VARS (space-separated TF_VAR_x=value for the workloads layer)
set -uo pipefail
# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"
AZ=$1
cd "$REPO_ROOT" || exit 1
inuse=$(k get pods -n "$NAMESPACE" -o json | python3 -c '
import json, sys
for p in json.load(sys.stdin)["items"]:
    vols = p["spec"].get("volumes") or []
    if p["status"]["phase"] in ("Running", "Pending") and any((v.get("persistentVolumeClaim") or {}).get("claimName") == "fsx-claim" for v in vols):
        print(p["metadata"]["name"])')
[ -z "$inuse" ] || { note "[fsx-move] fsx-claim in use by: $inuse; not moving"; exit 1; }
# A move runs two Terraform applies (~35-45 min). If the AWS session expires mid-apply,
# Terraform cannot save state and leaves FSx out of state with no PVC. Require enough
# credential lifetime first. MIN_CRED_MINUTES (default 90) is checked when the credential
# process reports an expiry; a refreshable credential_process is re-run by the SDK per call.
exp=$(aws configure export-credentials --format process 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("Expiration") or "")' 2>/dev/null)
if [ -n "$exp" ]; then
  left=$(( ( $(date -j -u -f %Y-%m-%dT%H:%M:%SZ "${exp%%.*}Z" +%s 2>/dev/null || date -u -d "$exp" +%s) - $(date +%s) ) / 60 ))
  [ "$left" -ge "${MIN_CRED_MINUTES:-90}" ] || { note "[fsx-move] AWS credentials expire in ${left} min (< ${MIN_CRED_MINUTES:-90}); not starting"; exit 1; }
fi
note "[fsx-move] fsx_az -> $AZ; storage apply (FSx re-create)"
export TF_VAR_fsx_az=$AZ
TF_VAR_fsx_storage_capacity=${FSX_CAPACITY:-1200} ./infrastructure/deploy.sh --action apply --layer storage --auto-approve \
  > "$WORK_DIR/move-storage.log" 2>&1 || { note "[fsx-move] storage apply FAILED"; exit 1; }
note "[fsx-move] workloads apply"
# shellcheck disable=SC2086
env ${WORKLOADS_TF_VARS:-} ./infrastructure/deploy.sh --action apply --layer workloads --auto-approve \
  > "$WORK_DIR/move-workloads.log" 2>&1 || { note "[fsx-move] workloads apply FAILED"; exit 1; }
note "[fsx-move] prefetch FSx from S3"
k delete job -n "$NAMESPACE" fsx-hydrate --ignore-not-found >/dev/null
cat <<YAML | k apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata: {name: fsx-hydrate, namespace: ${NAMESPACE}}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      # Read FSx from its own AZ (system nodes stay in target_az): run next to the caught GPU node.
      nodeSelector: {topology.kubernetes.io/zone: ${AZ}}
      tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]
      containers:
        - name: h
          image: public.ecr.aws/docker/library/busybox:1.36
          command: ["sh", "-c", "for d in /fsx/isaacsim/5.1.0 /fsx/models/RLinf-Gr00t-SFT-Stack-cube; do [ -d \$d ] || continue; find \$d -type f -size +0 -print0 | xargs -0 -n 64 -P 48 cat > /dev/null; done; echo HYDRATED"]
          resources: {requests: {cpu: "6", memory: 2Gi}}
          volumeMounts: [{name: fsx, mountPath: /fsx, readOnly: true}]
      volumes: [{name: fsx, persistentVolumeClaim: {claimName: fsx-claim}}]
YAML
k wait --for=condition=complete job/fsx-hydrate -n "$NAMESPACE" --timeout=2400s >/dev/null || { note "[fsx-move] prefetch did not complete"; exit 1; }
note "[fsx-move] done: FSx in $AZ, prefetched"
