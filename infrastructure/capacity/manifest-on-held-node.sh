#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Render any single-pod Job manifest for the node keep-sim is holding: envsubst
# ${IMAGE} ${NAMESPACE} (plus REQUIRED_VENVS when set), set metadata.name, and replace
# the pod's nodeSelector with the held node's hostname (tolerating nvidia.com/gpu).
# Use for smoke tests and container tests (L2); training Jobs use run-on-held-node.sh.
#
# usage: manifest-on-held-node.sh <job-name> <manifest.yaml>
# env:   NO_EFA=1 drops vpc.amazonaws.com/efa requests and sets EXPECT_EFA=0 (nodes without EFA)
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"
NAME=$1; SRC=$2
: "${IMAGE:?set IMAGE}"
NODE=$(k get pod -n "$NAMESPACE" keep-sim -o jsonpath='{.spec.nodeName}')
[ -n "$NODE" ] || { echo "no held node (keep-sim not scheduled)" >&2; exit 1; }
export IMAGE NAMESPACE REQUIRED_VENVS="${REQUIRED_VENVS:-}"
# shellcheck disable=SC2016  # envsubst takes the literal variable names
envsubst '${IMAGE} ${NAMESPACE} ${REQUIRED_VENVS}' < "$SRC" | python3 -c '
import re, sys
name, node, ns = sys.argv[1:4]
s = sys.stdin.read()
# Job name: the first metadata.name in the document
s = re.sub(r"(^metadata:\n(?:  .*\n)*?  name: )\S+", lambda m: m.group(1) + name, s, count=1, flags=re.M)
# Job namespace (container-test.yaml has none)
if not re.search(r"^metadata:\n(?:  .*\n)*?  namespace: ", s, flags=re.M):
    s = re.sub(r"^metadata:\n", f"metadata:\n  namespace: {ns}\n", s, count=1, flags=re.M)
# pod nodeSelector -> held node (drop the existing selector block, any indentation)
s, n = re.subn(r"^( +)nodeSelector:\n(?:\1  .*\n)+", lambda m: f"{m.group(1)}nodeSelector:\n{m.group(1)}  kubernetes.io/hostname: {node}\n", s, count=1, flags=re.M)
assert n == 1, "no nodeSelector block found"
# Node without EFA (gpu-sim pool): drop EFA resource requests and tell the test not to expect it
if __import__("os").environ.get("NO_EFA") == "1":
    s = re.sub(r"^ +vpc\.amazonaws\.com/efa: \S+\n", "", s, flags=re.M)
    s = re.sub(r"(- name: EXPECT_EFA\n +value: )\"1\"", lambda m: m.group(1) + "\"0\"", s)
print(s, end="")
' "$NAME" "$NODE" "$NAMESPACE"
