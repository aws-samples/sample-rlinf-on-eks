<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# RLinf Reference Implementation

RLinf v0.3 -- Macro-to-micro flow transformation for Embodied & Agentic RL. Deploys PPO training of VLA models on ManiSkill3, LIBERO, and RoboTwin simulators.

**Paper**: [RLinf System](https://arxiv.org/abs/2509.15965) | [RLinf-VLA](https://arxiv.org/abs/2510.06710) | **Code**: [github.com/RLinf/RLinf](https://github.com/RLinf/RLinf) | **Docs**: [rlinf.readthedocs.io](https://rlinf.readthedocs.io/en/latest/)

## Namespace

All RLinf training workloads deploy into the `rlinf` namespace. This is created by Terraform (`kubernetes_namespace.rlinf` in `infrastructure/cluster/main.tf`).

- Manifests in `manifests/` and `examples/` hardcode `namespace: rlinf`
- The `training-sa` ServiceAccount and `fsx-claim` PVC live in `rlinf`
- MLflow deploys into the `rlinf` namespace (accessed via `mlflow.rlinf.svc.cluster.local`)
- Infrastructure tests (L0 smoke, L3 NCCL) remain in `default`

Deploy with: `envsubst < examples/<example>/manifests/<example>.yaml | kubectl apply -f -`
(No `-n rlinf` needed -- namespace is in the manifest.)

## Related Skills

Load these via the `skill` tool for RLinf-specific procedural workflows:

| Skill | Scope |
|-------|-------|
| `rlinf-pytorch-training-scripts` | Hydra configs, veRL entrypoint, FSDP settings, hyperparameters |
| `rlinf-dataset-preparation` | HuggingFace model IDs, LIBERO/ManiSkill/RoboTwin asset staging |
| `rlinf-benchmarking` | LIBERO/RoboTwin evaluation, reference results from paper |
| `rlinf-open-source-plugins` | Ray worker architecture, veRL extensions, KubeRay CRDs |
| `rlinf-functional-testing` | L5 per-example test configs, expected outputs |

## Examples

All four examples run from ONE unified container image (`BUILD_TARGET=embodied-maniskill_libero` + a `dreamzero`-venv overlay in `examples/Dockerfile`), differing only by env vars (`CONFIG_NAME`/`VENV_NAME`/`MODEL_PATH`). The base target ships the PPO venvs (`openvla`/`openvla-oft`/`openpi`/gr00t/...) + ManiSkill & openpi assets; the overlay adds the `dreamzero` venv (DreamZero's `groot` is external via `DREAMZERO_PATH`). venvs are isolated `uv` envs, so one image serves all examples — see the DreamZero section below.

| Example | `CONFIG_NAME` | `VENV_NAME` | `MODEL_PATH` |
|---------|---------------|-------------|--------------|
| ManiSkill+OpenVLA-OFT PPO | `maniskill_ppo_openvlaoft_quickstart` | `openvla-oft` | `/fsx/models/openvla-oft-sft-libero10` |
| Wan World Model GRPO | `wan_libero_spatial_grpo_openvlaoft` | `openvla-oft` | `/fsx/models/Openvla-oft-SFT-libero-spatial-traj1` (+ Wan world model `/fsx/models/RLinf-Wan-LIBERO-Spatial`) |
| LIBERO+pi0 PPO | `libero_spatial_ppo_openpi_quickstart` | `openpi` | `/fsx/models/rlinf-pi0-sft-spatial` |
| DreamZero LIBERO 14B SFT (multi-node) | `libero_sft_dreamzero_14b` | `dreamzero` | `/fsx/models/DreamZero-DROID` |

LIBERO PPO examples additionally need: `LIBERO_DATASET_DIR=/fsx/datasets/libero` and `ROBOT_PLATFORM=LIBERO`.

## Critical Gotchas

### Container Build
- **Two-stage build is required**: upstream RLinf Dockerfile handles 15 BUILD_TARGETs with multi-venv (`uv`). Don't replicate -- build upstream first, layer EFA on top. See `examples/Dockerfile` and `examples/buildspec.yml`.
- **`NO_MIRROR=1` build arg required** for upstream RLinf Dockerfile on CodeBuild (otherwise defaults to unreachable Chinese mirrors).
- **Multi-venv pattern**: each model (openvla, openvla-oft, openpi, gr00t, dexbotic) gets its own venv via `uv`. Activated by `VENV_NAME` env var in the launch script (`examples/scripts/run_training_eks.sh`).

### Training
- **`runner.max_steps=1`** limits training to 1 PPO step (for validation). Use with `runner.save_interval=-1 runner.val_check_interval=-1`.
- **`run_embodiment.sh` does NOT forward Hydra overrides**. Call `train_embodied_agent.py` directly. The `run_training_eks.sh` launcher supports `HYDRA_OVERRIDES` env var.
- **MuJoCo headless rendering**: requires `MUJOCO_GL=osmesa` + `PYOPENGL_PLATFORM=osmesa` (not EGL). Set in Dockerfile.
- **LIBERO's `__init__.py` calls `input()` at import time** -- hangs in containers. Must pre-create `~/.libero/config.yaml` and set `LIBERO_DATASET_DIR`.
- **flash-attn required** for DiT attention layers. Must build with `--no-build-isolation` in container.

### DreamZero LIBERO 14B SFT (validated end-to-end on EKS, 2026-06)

This example demonstrates the canonical customer workflow: **continue SFT from the
released DreamZero-DROID 14B checkpoint onto a new embodiment's data (LIBERO),
then eval in the LIBERO simulator.** It is FSDP2 + KubeRay RayJob multi-node (NOT the
old DROID/torchrun/DeepSpeed/StatefulSet design). Full pipeline: build → stage → SFT (2x p5en) →
DCP→.pt convert → LIBERO sim eval → in-sim rollout video.

- **Unified image: `BUILD_TARGET=embodied-maniskill_libero` + `dreamzero`-venv overlay.** The base target ships openvla/openvla-oft/openpi/gr00t/... venvs + ManiSkill/openpi assets; `examples/Dockerfile` adds the `dreamzero` venv when the base lacks it (RLinf's `install.sh --venv dreamzero --model dreamzero --env libero`, run from `${UV_PATH}`). `VENV_NAME=dreamzero`. Pins in `examples/buildspec.yml`: `UPSTREAM_REF=0505431899574619da86f551bad70b71e0ea2177` (v0.3), `DREAMZERO_REF=ab790c198fbc`. The `groot` package is external (`github.com/RLinf/dreamzero.git`), on PYTHONPATH via `DREAMZERO_PATH=/workspace/DreamZero`; the buildspec clones it.
- **`groot` package is external** (`github.com/RLinf/dreamzero.git`), provided via
  `DREAMZERO_PATH=/workspace/DreamZero` on PYTHONPATH. The integration glue
  (`rlinf/models/embodiment/dreamzero/`, world-model env) is in-tree. Buildspec clones it.
- **Multi-node uses a KubeRay `RayJob`** (embedded RayCluster: head 1 + worker 1), NOT torchrun and NOT a StatefulSet. The KubeRay operator starts Ray and runs the launcher (`run_dreamzero_sft_eks.sh`, Ray-agnostic) as the head `entrypoint`; there is no manual head election. `shutdownAfterJobFinishes: true` tears the cluster down on completion.
- **Apply manifests with RESTRICTED envsubst**: `envsubst '${ECR_URI} ${NAMESPACE}' < ...`. The Ray head-election bash is gone (the KubeRay operator handles it), but the RayJob `entrypoint` still embeds an inline `bash -c`; unrestricted envsubst would mangle it.
- **PYTHONPATH unbound-variable trap**: every venv-activating context (launcher, Ray
  bootstrap, dreamzero-deps Dockerfile layer) must `export PYTHONPATH="${PYTHONPATH:-}"`
  BEFORE sourcing the venv; the activate script references PYTHONPATH under `set -u`.
- **Hydra `+` prefix for commented/absent keys**: `metadata_json_path` and
  `fsdp_config.save_full_model_weights` are NOT in the config struct → must be ADDED with
  `+actor.model.metadata_json_path=...` (plain override fails: "Key not in struct").
- **Metadata embodiment mismatch**: the DreamZero-DROID checkpoint bundles
  `experiment_cfg/metadata.json` for `oxe_droid` ONLY. LIBERO SFT (`embodiment_tag:
  libero_sim`) fails with `KeyError: embodiment_tag 'libero_sim' not found` unless you
  generate LIBERO stats: `toolkits/lerobot/generate_dreamzero_metadata.py --preset
  libero_sim --dataset-root /fsx/datasets/libero` (see `generate-metadata.yaml`) and pass
  `+actor.model.metadata_json_path=/fsx/models/metadata-libero.json` (launcher default).
- **Temporal alignment**: `libero_sft_dreamzero_14b.yaml` sets `action_horizon=16` but
  inherits `num_action_per_block=24` from the DROID model default → forward-pass
  assertion (`actions 64 / (noise-1)=8 != 24//2=12`). Override
  `actor.model.num_action_per_block=16` (launcher does this; matches the 5B LIBERO config).
- **Dataset MUST be `physical-intelligence/libero`** (state/actions column schema matching
  the `libero_sim` preset), NOT `lerobot/libero` (different `observation.state`/`action`
  schema → breaks metadata gen + transforms).
- **Checkpoint format**: SFT writes a sharded FSDP DCP (`.distcp` + `.metadata`) under
  `.../global_step_<N>/actor/dcp_checkpoint/`. Eval needs a single `.pt`. Convert offline
  on CPU with `convert-checkpoint.yaml` (`convert_dcp_to_pt.py`).
- **DCP finalization crash (REPRODUCED + ROOT-CAUSED + FIXED, 2026-06)**: on torch 2.6,
  `dcp.save`'s post-write finalization (`_DistWrapper.all_reduce` → `broadcast_object` →
  `broadcast_object_list`) broadcasts a multi-MB pickled result object over the **default
  (NCCL) PG on CUDA**. At the end of a long (~209GB / ~20min) checkpoint write this races
  with NCCL comm teardown, so non-coordinator ranks read an all-zero buffer →
  `_pickle.UnpicklingError: invalid load key '\x00'` — AFTER all 16 shards + `.metadata`
  are already on disk. **Fix**: `dcp-save-gloo-coordinator.patch` passes a CPU/gloo process
  group to `dcp.save(..., process_group=gloo_pg)` so the object broadcast runs over gloo
  (CPU), immune to the CUDA/NCCL teardown race (this is what torch 2.7+ does upstream).
  Validated end-to-end on 2x p5en: RayJob `SUCCEEDED`, 207GB DCP, zero `UnpicklingError`.
  (The older `dcp-save-finalize-besteffort.patch` symptom-guard is removed in favor of this.)
- **`save_full_model_weights` MUST be forced false on the 16B model**: `libero_sft_dreamzero_14b.yaml`
  omits `fsdp_config.save_full_model_weights`, so it falls through to the code default of
  `True` (`fsdp_model_manager.py`). On the 16B model that runs a full-state-dict gather →
  `RuntimeError: Backend nccl does not support allgather_into_tensor_coalesced` (and a rank-0
  gather that stalls). The launcher (`run_dreamzero_sft_eks.sh`) passes
  `+actor.fsdp_config.save_full_model_weights=false` by default; DCP-only + offline convert
  is the supported path. Do NOT set it true.
- **FSx free space is critical**: a 14B DCP checkpoint is ~140-206GB (full optimizer
  state). A full filesystem truncates `torch.save` mid-write → `inline_container.cc
  unexpected pos` → corrupt unreadable shards (EOFError on convert). Ensure >=250GB free
  before an SFT run.

### Dataset Download
- **`snapshot_download` is too slow for large video datasets** (138K+ files). It paginates the full repo tree via REST API before downloading, achieving only ~5 files/s. Use `examples/dreamzero/scripts/fast_hf_download.py` instead (direct URL construction + 64 concurrent async downloads = 65+ files/s, 13x speedup).
- **HuggingFace rate-limits the tree/metadata API, NOT the file CDN**. Multiple parallel `snapshot_download` calls with the same token trigger 429 errors. Direct file downloads do not.
- **Do NOT use `hf_transfer` for many small files** -- it's optimized for large file multi-part downloads, not thousands of 1-2MB videos where per-file HTTP overhead dominates.
- **DROID video structure is deterministic**: `videos/chunk-{NNN:03d}/{view_name}/episode_{NNNNNN:06d}.mp4`, 58 chunks, 1000 episodes/chunk (last chunk: 774), 3 camera views. Use this to construct URLs directly.

### Python/Dependency Pins
- **Never install `numpy>=2`** in NGC-derived containers -- breaks all compiled C-extensions.
- **`transformers` must be `>=4.40,<4.46`** (newer versions reject NGC pre-release PyTorch version strings).
- **`timm` must be `>=0.9.10,<1.0`** (breaking API in 1.0+).
- **`robosuite` must be 1.4.1** (LIBERO needs removed APIs from 1.5+).

### GPU Sizing
- **L4 (24 GB) CANNOT fit 7B VLA models** for RL training. FSDP on a single GPU cannot shard -- full model must fit in VRAM.
- **Minimum for 7B+ VLA**: A10G with FSDP across 2+ GPUs, or single A100/H100 (40-80 GB).
- **DreamZero LIBERO 14B requires 2x p5en.48xlarge** (8x H200 141GB each). FSDP2 `full_shard`
  across 16 GPUs; the 16.48B diffusion model is sharded, not replicated.
- **DreamZero 14B DCP checkpoint = ~140-206GB** (16 FSDP shards incl. full optimizer state).
  Ensure FSx has >=250GB free; a full FS truncates `torch.save` → corrupt shards. (This is
  FSDP DCP, NOT the old DeepSpeed ZeRO-2 ~282GB figure.)
- **DreamZero eval OOMs at upstream's `total_num_envs=128`** for the 14B model (sim + 16B
  model co-located on 8x H200). Use `eval.total_num_envs=16` (our config default).

### DreamZero Eval & Showcase (LIBERO simulator)
- **Eval needs a 14B eval config**: upstream ships only `libero_spatial_eval_dreamzero.yaml`
  (5B). Our `examples/dreamzero/configs/libero_spatial_eval_dreamzero_14b.yaml` uses
  `model/dreamzero_14b` (resolved via a Hydra `searchpath` entry to
  `examples/sft/config/`, since the 14B model config lives only there) with null component
  paths (the full checkpoint provides weights), `target_video 352x640`, `num_action_per_block=16`.
- **Eval `runner.ckpt_path` is the converted `.pt`** (`full_weights.pt`), NOT the DCP dir.
- **In-sim rollout video** is produced by the eval itself (`env.eval.video_cfg.save_video=True`)
  → `{log_path}/video/eval/seed_*/0.mp4`. This replaced the old offline DROID
  prediction/compose scripts (now removed).
- **Eval config is mounted via ConfigMap + copied** into `examples/embodiment/config/` by the
  launcher (mounting over the dir would hide upstream config groups).
- **1-step checkpoint → `success_once: 0.0`** is expected; the eval gate validates the
  machinery, not accuracy. Real accuracy needs a multi-step SFT run.

## Key Files

| File | Purpose |
|------|---------|
| `examples/Dockerfile` | Stage 2: EFA overlay on upstream RLinf image |
| `examples/buildspec.yml` | CodeBuild: two-stage build (upstream + EFA) |
| `examples/scripts/run_training_eks.sh` | Generic training launcher (any config/venv) |
| `examples/scripts/install_extras.sh` | Conditional package installer for EXTRAS build arg |
| `infrastructure/manifests/container-test.yaml` | Container validation (10 tests) |
| `infrastructure/manifests/training-statefulset.yaml` | Multi-node StatefulSet for L6 validation (Ray) |
| `infrastructure/manifests/training-service.yaml` | Headless Service for L6 validation |
| `examples/maniskill-openvlaoft-ppo/manifests/model-download.yaml` | Downloads OpenVLA-OFT model |
| `examples/maniskill-openvlaoft-ppo/manifests/maniskill-openvlaoft-ppo.yaml` | ManiSkill + OpenVLA-OFT PPO training Job |
| `examples/wan-openvlaoft-grpo/manifests/model-download.yaml` | Downloads OpenVLA-OFT model + Wan world model |
| `examples/wan-openvlaoft-grpo/manifests/wan-openvlaoft-grpo.yaml` | Wan world model + OpenVLA-OFT GRPO training Job |
| `examples/libero-pi0-ppo/manifests/model-download.yaml` | Downloads pi0 SFT model |
| `examples/libero-pi0-ppo/manifests/libero-pi0-ppo.yaml` | LIBERO + pi0 PPO training Job |
| `examples/dreamzero/manifests/model-download.yaml` | Downloads DreamZero-DROID ckpt + umt5-xxl + physical-intelligence/libero dataset |
| `examples/dreamzero/manifests/generate-metadata.yaml` | CPU Job: generate libero_sim metadata.json (training image) |
| `examples/dreamzero/manifests/dreamzero-sft.yaml` | KubeRay RayJob (embedded RayCluster) for 2-node FSDP2 SFT |
| `examples/dreamzero/manifests/convert-checkpoint.yaml` | CPU Job: FSDP DCP shards → full_weights.pt |
| `examples/dreamzero/manifests/dreamzero-eval.yaml` | Single-node GPU Job: LIBERO simulator eval + in-sim video |
| `examples/dreamzero/configs/libero_spatial_eval_dreamzero_14b.yaml` | 14B LIBERO eval config (our variant; upstream ships 5B only) |
| `examples/dreamzero/scripts/run_dreamzero_sft_eks.sh` | Multi-node DreamZero SFT launcher (Ray/FSDP, Ray-agnostic) |
| `examples/dreamzero/scripts/convert_checkpoint.sh` | DCP→.pt conversion launcher |
| `examples/dreamzero/scripts/run_dreamzero_eval_eks.sh` | LIBERO simulator eval launcher |
| `examples/dreamzero/scripts/fast_hf_download.py` | Fast parallel HF downloader (bypasses tree API; for large datasets) |
| `examples/dreamzero/scripts/plot_dreamzero_loss.py` | Plot SFT loss curve from tensorboard |

## SFT Model Weights

| Model | HuggingFace Path | Used By |
|-------|------------------|---------|
| OpenVLA-OFT LIBERO-10 SFT | `RLinf/Openvla-oft-SFT-libero10-trajall` | ManiSkill+OpenVLA-OFT |
| OpenVLA-OFT LIBERO-Spatial SFT | `Haozhan72/Openvla-oft-SFT-libero-spatial-traj1` | Wan World Model GRPO (policy) |
| Wan world model (LIBERO-Spatial) | `RLinf/RLinf-Wan-LIBERO-Spatial` | Wan World Model GRPO (env) |
| Pi0 Spatial SFT | `RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal` | LIBERO+pi0 |
| DreamZero-DROID | `GEAR-Dreams/DreamZero-DROID` | DreamZero LIBERO 14B SFT (14B warm-start backbone) |
| umt5-xxl | `google/umt5-xxl` | DreamZero SFT (tokenizer) |

DreamZero LIBERO dataset: `physical-intelligence/libero` (HF dataset; NOT `lerobot/libero`).

Full collection: https://huggingface.co/RLinf
