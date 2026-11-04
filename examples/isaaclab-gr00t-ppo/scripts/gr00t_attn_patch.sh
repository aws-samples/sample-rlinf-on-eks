#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Runtime patch: run GR00T N1.5's Eagle backbone (Qwen3 LLM + SigLIP vision) with
# PyTorch SDPA attention instead of flash-attn. For GPUs the image's flash-attn has
# no kernels for (Blackwell RTX sm_120, e.g. g7e RTX PRO 6000): the prebuilt
# flash-attn 2.8.3 wheel ships only sm_80 + sm_90.
#
# Two hard-coded flash_attention_2 settings in the gr00t package:
#   - eagle2_hg_model/config.json            "_attn_implementation": "flash_attention_2"
#   - eagle2_hg_model/modeling_eagle2_5_vl.py config.vision_config._attn_implementation = "flash_attention_2"
# Edits the venv copy in the container only (the image is unchanged). Idempotent.
# Enable with GR00T_ATTN=sdpa before run_training_eks.sh.
set -euo pipefail
[ "${GR00T_ATTN:-}" = sdpa ] || exit 0
D=/opt/venv/gr00t/gr00t/gr00t/model/backbone/eagle2_hg_model
sed -i 's/"_attn_implementation": "flash_attention_2"/"_attn_implementation": "sdpa"/' "$D/config.json"
sed -i 's/config.vision_config._attn_implementation = "flash_attention_2"/config.vision_config._attn_implementation = "sdpa"/' "$D/modeling_eagle2_5_vl.py"
grep -q '"_attn_implementation": "sdpa"' "$D/config.json"
grep -q 'vision_config._attn_implementation = "sdpa"' "$D/modeling_eagle2_5_vl.py"
echo "gr00t_attn_patch: Eagle backbone set to sdpa"
