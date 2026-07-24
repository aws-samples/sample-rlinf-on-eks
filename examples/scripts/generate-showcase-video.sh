#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# generate-showcase-video.sh -- End-to-end showcase video generation
#
# Deploys an eval-video Job for a given example, waits for completion, downloads
# the raw video, strips duplicate frames (if needed), and places the result in
# the example's assets/ directory.
#
# Usage:
#   ./examples/scripts/generate-showcase-video.sh <example-name>
#
# Examples:
#   ./examples/scripts/generate-showcase-video.sh maniskill-openvlaoft-ppo
#   ./examples/scripts/generate-showcase-video.sh libero-pi0-ppo
#
# Prerequisites:
#   - kubectl configured for the target cluster
#   - ECR_URI, NAMESPACE, S3_BUCKET, and CKPT_PATH environment variables set
#   - LORA_PATH also required for openvlaoft-ppo example
#   - aws CLI configured with S3 access to the FSx data bucket
#   - ffmpeg installed (only needed if num_action_chunks > 1)
#   - Model weights and checkpoint already on FSx (run model-download first)
#
# The script reads metadata from the eval-video.yaml Job annotations:
#   rlinf.io/num-action-chunks  -- frame decimation factor (1 = no stripping)
#   rlinf.io/video-output-path  -- FSx subpath where video is written
#   rlinf.io/asset-filename     -- target filename in assets/
#   rlinf.io/s3-bucket          -- S3 bucket backing FSx (for download)
# =============================================================================
set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB_TIMEOUT="${JOB_TIMEOUT:-3600}"  # 60 min default (model loading + eval)

# --- Color logging (same style as tests/lib/common.sh) ---
if [[ -z "${NO_COLOR:-}" ]] && [[ -t 1 ]]; then
  _RED='\033[0;31m'; _GREEN='\033[0;32m'; _YELLOW='\033[0;33m'
  _CYAN='\033[0;36m'; _BOLD='\033[1m'; _RESET='\033[0m'
else
  _RED='' _GREEN='' _YELLOW='' _CYAN='' _BOLD='' _RESET=''
fi
log_info() { echo -e "${_CYAN}INFO${_RESET}: $*"; }
log_pass() { echo -e "${_GREEN}PASS${_RESET}: $*"; }
log_fail() { echo -e "${_RED}FAIL${_RESET}: $*"; }
log_warn() { echo -e "${_YELLOW}WARN${_RESET}: $*"; }

# --- Argument parsing ---
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <example-name>"
  echo ""
  echo "Available examples:"
  for dir in "$REPO_ROOT"/examples/*/manifests/eval-video.yaml; do
    example="$(basename "$(dirname "$(dirname "$dir")")")"
    echo "  $example"
  done
  exit 1
fi

EXAMPLE="$1"
EXAMPLE_DIR="$REPO_ROOT/examples/$EXAMPLE"
MANIFEST="$EXAMPLE_DIR/manifests/eval-video.yaml"

if [[ ! -f "$MANIFEST" ]]; then
  log_fail "No eval-video.yaml found at: $MANIFEST"
  exit 1
fi

# --- Validate prerequisites ---
for cmd in kubectl aws envsubst; do
  if ! command -v "$cmd" &>/dev/null; then
    log_fail "Required command not found: $cmd"
    exit 1
  fi
done

if [[ -z "${ECR_URI:-}" ]]; then
  log_fail "ECR_URI environment variable not set"
  echo "  Set it with: export ECR_URI=\$(terraform -chdir=infrastructure/build output -raw ecr_repository_url)"
  exit 1
fi

if [[ -z "${NAMESPACE:-}" ]]; then
  log_fail "NAMESPACE environment variable not set"
  echo "  Set it with: export NAMESPACE=rlinf"
  exit 1
fi

if [[ -z "${S3_BUCKET:-}" ]]; then
  log_fail "S3_BUCKET environment variable not set"
  echo "  Set it with: export S3_BUCKET=\$(terraform -chdir=infrastructure/storage output -raw fsx_s3_bucket)"
  exit 1
fi

if [[ -z "${CKPT_PATH:-}" ]]; then
  log_fail "CKPT_PATH environment variable not set"
  echo "  Set it to the full_weights.pt path on FSx, e.g.:"
  echo "  export CKPT_PATH=/fsx/checkpoints/<run>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt"
  exit 1
fi

# --- Extract metadata from manifest annotations ---
# We parse the YAML directly (after envsubst) to avoid needing the Job deployed first.
rendered=$(envsubst < "$MANIFEST")
get_annotation() {
  local key="$1"
  echo "$rendered" | grep "rlinf.io/$key" | head -1 | sed 's/.*: *"\(.*\)"/\1/'
}

NUM_ACTION_CHUNKS=$(get_annotation "num-action-chunks")
VIDEO_OUTPUT_PATH=$(get_annotation "video-output-path")
ASSET_FILENAME=$(get_annotation "asset-filename")
S3_BUCKET=$(get_annotation "s3-bucket")

if [[ -z "$NUM_ACTION_CHUNKS" || -z "$VIDEO_OUTPUT_PATH" || -z "$ASSET_FILENAME" || -z "$S3_BUCKET" ]]; then
  log_fail "Missing required annotations in $MANIFEST"
  echo "  Required: rlinf.io/num-action-chunks, rlinf.io/video-output-path,"
  echo "            rlinf.io/asset-filename, rlinf.io/s3-bucket"
  exit 1
fi

# Check ffmpeg if frame stripping is needed
if [[ "$NUM_ACTION_CHUNKS" -gt 1 ]]; then
  if ! command -v ffmpeg &>/dev/null; then
    log_fail "ffmpeg required for frame stripping (num_action_chunks=$NUM_ACTION_CHUNKS) but not found"
    echo "  Install with: brew install ffmpeg"
    exit 1
  fi
fi

JOB_NAME=$(echo "$rendered" | grep "name: rlinf-" | head -1 | sed 's/.*name: *//')

log_info "Example:          $EXAMPLE"
log_info "Job:              $JOB_NAME"
log_info "Action chunks:    $NUM_ACTION_CHUNKS"
log_info "Video output:     s3://$S3_BUCKET/checkpoints/$VIDEO_OUTPUT_PATH/"
log_info "Asset filename:   $ASSET_FILENAME"
echo ""

# --- Step 1: Deploy the eval job ---
log_info "Deploying eval-video job..."

# Clean up any existing job with the same name
kubectl delete job "$JOB_NAME" -n "$NAMESPACE" 2>/dev/null || true
sleep 2

echo "$rendered" | kubectl apply -f -
log_pass "Job deployed: $JOB_NAME"

# --- Step 2: Wait for job completion ---
log_info "Waiting for job completion (timeout: ${JOB_TIMEOUT}s)..."

elapsed=0
# Wait for pod creation
while (( elapsed < JOB_TIMEOUT )); do
  if kubectl get pods -n "$NAMESPACE" -l "job-name=${JOB_NAME}" --no-headers 2>/dev/null | grep -q .; then
    break
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

if (( elapsed >= JOB_TIMEOUT )); then
  log_fail "Timed out waiting for pod creation"
  exit 1
fi

log_info "Pod created, waiting for job to complete..."

# Wait for completion or failure
if kubectl wait --for=condition=complete "job/${JOB_NAME}" \
     -n "$NAMESPACE" --timeout="${JOB_TIMEOUT}s" 2>/dev/null; then
  log_pass "Job completed successfully"
else
  # Check if failed
  status=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" \
    -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)
  if [[ "$status" == "True" ]]; then
    log_fail "Job failed. Last 30 lines of logs:"
    pod=$(kubectl get pods -n "$NAMESPACE" -l "job-name=${JOB_NAME}" \
      --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null)
    kubectl logs "$pod" -n "$NAMESPACE" --tail=30 2>/dev/null || true
    exit 1
  fi
  log_fail "Job timed out after ${JOB_TIMEOUT}s"
  exit 1
fi

# --- Step 3: Download the best video ---
log_info "Finding video output..."

VIDEO_S3_PREFIX="s3://$S3_BUCKET/checkpoints/$VIDEO_OUTPUT_PATH/video/eval/"

# List all mp4 files and pick the largest (most complete episode)
best_video=$(aws s3 ls "$VIDEO_S3_PREFIX" --recursive 2>/dev/null \
  | grep '\.mp4$' \
  | sort -k3 -n -r \
  | head -1 \
  | sed 's/^[0-9 -]*[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\} *[0-9]* //')

if [[ -z "$best_video" ]]; then
  log_fail "No .mp4 files found at $VIDEO_S3_PREFIX"
  exit 1
fi

TMPDIR="${TMPDIR:-/tmp}"
RAW_VIDEO="$TMPDIR/rlinf-eval-video-raw-$$.mp4"
FINAL_VIDEO="$EXAMPLE_DIR/assets/$ASSET_FILENAME"

log_info "Downloading: s3://$S3_BUCKET/$best_video"
aws s3 cp "s3://$S3_BUCKET/$best_video" "$RAW_VIDEO"
log_pass "Downloaded raw video ($(du -h "$RAW_VIDEO" | cut -f1))"

# --- Step 4: Post-process (strip duplicate frames if needed) ---
if [[ "$NUM_ACTION_CHUNKS" -gt 1 ]]; then
  log_info "Stripping duplicate frames (keeping every ${NUM_ACTION_CHUNKS}th frame)..."

  ffmpeg -i "$RAW_VIDEO" \
    -vf "select='not(mod(n\\,${NUM_ACTION_CHUNKS}))',setpts=N/FRAME_RATE/TB" \
    -r 5 -c:v libx264 -pix_fmt yuv420p -crf 28 -preset slow \
    -y "$FINAL_VIDEO" 2>/dev/null

  log_pass "Frame stripping complete"
  rm -f "$RAW_VIDEO"
elif command -v ffmpeg &>/dev/null; then
  # No stripping needed — re-encode to consistent format for uniform file sizes
  log_info "Re-encoding to standard format..."
  ffmpeg -i "$RAW_VIDEO" \
    -c:v libx264 -pix_fmt yuv420p -crf 28 -preset slow \
    -y "$FINAL_VIDEO" 2>/dev/null

  rm -f "$RAW_VIDEO"
  log_pass "Re-encoding complete"
else
  # No ffmpeg available but no stripping needed — use raw video directly
  mv "$RAW_VIDEO" "$FINAL_VIDEO"
  log_warn "ffmpeg not available, using raw video (file size may be larger than optimal)"
fi

# --- Step 5: Report results ---
echo ""
log_pass "Showcase video generated:"
echo "  File: $FINAL_VIDEO"
echo "  Size: $(du -h "$FINAL_VIDEO" | cut -f1)"

# Quick metadata check via ffprobe if available
if command -v ffprobe &>/dev/null; then
  frames=$(ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=nb_read_frames -of csv=p=0 "$FINAL_VIDEO" 2>/dev/null || echo "?")
  duration=$(ffprobe -v error -show_entries format=duration \
    -of csv=p=0 "$FINAL_VIDEO" 2>/dev/null || echo "?")
  resolution=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0:s=x "$FINAL_VIDEO" 2>/dev/null || echo "?")
  echo "  Resolution: $resolution"
  echo "  Frames: $frames"
  echo "  Duration: ${duration}s"
fi

# --- Step 6: Cleanup ---
log_info "Cleaning up job..."
kubectl delete job "$JOB_NAME" -n "$NAMESPACE" 2>/dev/null || true
log_pass "Done. Review the video, then commit: git add $FINAL_VIDEO"
