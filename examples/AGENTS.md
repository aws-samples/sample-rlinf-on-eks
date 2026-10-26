<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# RLinf Reference Implementation

RLinf main@d844e648 (post-v0.4, 2026-10-22) -- Macro-to-micro flow transformation for Embodied & Agentic RL. Deploys PPO training of VLA models on ManiSkill3, LIBERO, and RoboTwin simulators.

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

All four examples run from ONE unified container image (`BUILD_TARGET=embodied-maniskill_libero`, which ships the `dreamzero` venv upstream; `examples/Dockerfile` requires `BUILD_TARGET=embodied-maniskill_libero`; the build fails otherwise), differing only by env vars (`CONFIG_NAME`/`VENV_NAME`/`MODEL_PATH`). The base target ships the PPO venvs (`openvla`/`openvla-oft`/`openpi`/gr00t/...) + ManiSkill & openpi assets; DreamZero's `groot` is external via `DREAMZERO_PATH`. venvs are isolated `uv` envs, so one image serves all examples — see the DreamZero section below.

| Example | `CONFIG_NAME` | `VENV_NAME` | `MODEL_PATH` |
|---------|---------------|-------------|--------------|
| ManiSkill+OpenVLA-OFT PPO | `maniskill_ppo_openvlaoft_quickstart` | `openvla-oft` | `/fsx/models/openvla-oft-sft-libero10` |
| Wan World Model GRPO | `wan_libero_spatial_grpo_openvlaoft` | `openvla-oft` | `/fsx/models/Openvla-oft-SFT-libero-spatial-traj1` (+ Wan world model `/fsx/models/RLinf-Wan-LIBERO-Spatial`) |
| LIBERO+pi0 PPO | `libero_spatial_ppo_openpi_quickstart` | `openpi` | `/fsx/models/rlinf-pi0-sft-spatial` |
| DreamZero LIBERO 14B SFT (multi-node) | `libero_sft_dreamzero_14b` | `dreamzero` | `/fsx/models/DreamZero-DROID` |

LIBERO PPO examples additionally need: `LIBERO_DATASET_DIR=/fsx/datasets/libero` and `ROBOT_PLATFORM=LIBERO`.

## Critical Gotchas

### Container Build
- **Image profiles (`PROFILE` CodeBuild env var; `examples/profiles.env`)**:
  - `full` (default) = upstream `embodied-maniskill_libero` (13 venvs), tagged `<BUILD_TARGET>-<ts>` + `latest`. It is the **only** profile that pushes `:latest`.
  - Other profiles build upstream `--target embodied-common-image` (still with `BUILD_TARGET=embodied-maniskill_libero`, which selects its base), then `examples/Dockerfile.profile` (one `install.sh` per `venv:model:env` tuple), then the same stage 2. They are tagged `<profile>-<ts>` + `<profile>-latest`, and write `/opt/venv/.profile` (one venv per line). The stage-2 DreamZero guard and L2 read that file.
  - Profile images have **no default venv in `.bashrc`** (that came from upstream `final-image`), so set `VENV_NAME` explicitly.
  - `ASSET_VENV` must be a profile venv with `mani_skill` + `hf`. `resolve_profile.sh` checks this.
  - Profiles use `--env maniskill_libero` (matches the validated full image), **except the LIBERO-only `libero-pi0` and `dreamzero`**, which use `--env libero` (no ManiSkill/SAPIEN, ~1.5 GB smaller; upstream documents `install.sh embodied --model dreamzero --env libero` for DreamZero SFT + LIBERO eval). Both are revalidated (pi0 PPO 5-step; DreamZero 2-node SFT + convert + LIBERO eval smoke).
  - Custom profile: `PROFILE=<name> INSTALLS="v:m:e ..."`.
  - Re-check `embodied-common-image` and the `install.sh` options on every `UPSTREAM_REF` bump.
- **Manifests take `${IMAGE}`** (full `<ecr repo>:<tag>`), not `${ECR_URI}:latest`. envsubst has no defaults, so an unset `IMAGE` renders `image: ` and the pod fails. `tests/lib/cluster.sh` derives `IMAGE=${ECR_URI}:${IMAGE_TAG:-latest}`.
- **Two-stage build is required**: upstream RLinf Dockerfile handles 20 BUILD_TARGETs (reason + 19 embodied) with multi-venv (`uv`). Don't replicate -- build upstream first, layer EFA on top. See `examples/Dockerfile` and `examples/buildspec.yml`.
- **`NO_MIRROR=1` build arg required** for upstream RLinf Dockerfile on CodeBuild (otherwise defaults to unreachable Chinese mirrors).
- **Multi-venv pattern**: each model (openvla, openvla-oft, openpi, gr00t, dexbotic, dreamzero, ...; 13 venvs in `embodied-maniskill_libero`) gets its own venv via `uv`. Activated by `VENV_NAME` env var in the launch script (`examples/scripts/run_training_eks.sh`).
- **Stage-1 build patches (`examples/scripts/upstream-build-patches/*.patch`)** are `git apply`'d to the upstream clone by buildspec pre_build (fail-fast). **None are currently needed** (the directory holds only `.gitkeep`). Patches are line-anchored to upstream files; re-check each one on every `UPSTREAM_REF` bump. History: `dexbotic-hub-cap.patch` (main@c70606f0) capped `huggingface-hub<1.0` in the dexbotic venv because unpinned gradio 6.29 → hub 2.x made the resolver backtrack to a tokenizers 0.10.3 sdist that needs Rust. Upstream #1663 (557ccda5) now installs dexbotic with `transformers==4.53.2` on the same line (which requires `huggingface-hub<1.0` and `tokenizers<0.22`), so the patch was dropped at main@d844e648.
- **Wan overlay pins the openvla-oft venv's numpy** (`examples/Dockerfile`, `pip install -c` with `numpy==<installed>`). The overlay's deps are unpinned, and pandas 3.1.0 (2026-10-22) requires numpy>=2. Unconstrained, pip upgraded numpy 1.26.4 → 2.4.6, which broke tensorflow 2.15 / ml_dtypes (`ImportError: numpy.core.umath failed to import` in RolloutWorker init) for ManiSkill PPO and Wan GRPO. The build now fails if numpy changes.
- **nccl-tests tag must match the NCCL version**: nccl-tests v2.21.0 (2026-10) calls `ncclCommWindowRegister(..., NCCL_WIN_GIN_ONLY)` in `src/device_api/gin/utils/gin_context.cc`; that flag first exists in NCCL v2.32.3, so with NCCL v2.31.2-1 stage 2 fails with `error: 'NCCL_WIN_GIN_ONLY' was not declared in this scope`. Pin nccl-tests v2.20.0 until NCCL is bumped to 2.32+.
- **Full-node L3 (`nccl-allreduce-fullnode.yaml`) first-run behaviour**: the launcher log stays empty for ~1-2 min after all pods are Running -- the launcher's `ssh` (ConnectionAttempts=10) retries until the workers' headless-Service DNS resolves; `orted` then appears on both workers. Not a hang. The MPIJob takes ~3-4 min total with a cached image. aws-ofi-nccl 1.21 logs `Selected provider is efa` (lowercase `provider`); log parsers must match both cases.
- **NCCL v2.31.2-1 on the CUDA 12.8.1 base**: aws-ofi-nccl 1.21.1 (EFA 1.50.0) is tested with NCCL v2.31.2-1; re-check the plugin's release notes before changing either. Known issue: NCCL 2.31.2 PAT+NVLS has a performance regression on H100; PAT is off by default, so do not set `NCCL_ALGO=PAT`.

### Training
- **`runner.max_steps=1`** limits training to 1 PPO step (for validation). Use with `runner.save_interval=-1 runner.val_check_interval=-1`.
- **`run_embodiment.sh` does NOT forward Hydra overrides**. Call `train_embodied_agent.py` directly. The `run_training_eks.sh` launcher supports `HYDRA_OVERRIDES` env var.
- **MuJoCo headless rendering**: requires `MUJOCO_GL=osmesa` + `PYOPENGL_PLATFORM=osmesa` (not EGL). Set in Dockerfile.
- **LIBERO's `__init__.py` calls `input()` at import time** -- hangs in containers. Must pre-create `~/.libero/config.yaml` and set `LIBERO_DATASET_DIR`.
- **L5 harness (`tests/manifests/training-step-test.yaml`) runs from `/workspace/RLinf`** with `--config-path config` and `EMBODIED_PATH`; ManiSkill needs `MANISKILL_ASSET_DIR=/fsx/assets/maniskill` (the in-tree default assets dir is not in the image), and the OFT quickstart needs `actor.model.lora_path=/fsx/models/rlinf-openvlaoft-maniskill-base-lora` (`is_lora: True`, placeholder path). Per-example extras go in the 5th field of `L5_EXAMPLE_LIST` (`EXTRA_OVERRIDES`).
- **Enabling eval on pi0 training** (`runner.val_check_interval>0`): the config's `env.eval.total_num_envs: 500` is not divisible by 8 env workers → assertion. Add `env.eval.total_num_envs=128`.
- **`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`** in Ray worker logs (Rollout/Actor/Env ranks, ~10x per run) is non-fatal TensorFlow/protobuf import noise in the openvla-oft venv; jobs complete normally. Don't grep `Error` as a failure signal; use `Traceback` + Job status.
- **No step-0 eval upstream** (no `val_before_train`): the first `eval/*` comes after step `val_check_interval`. The SFT-start success is the step-1 train `env/success_once`.
- **Eval entrypoint moved**: `examples/embodiment/eval_embodied_agent.py` was removed upstream (2e1c99f7). The eval-video manifests (`maniskill-openvlaoft-ppo`, `libero-pi0-ppo`) run `python3 evaluations/eval_embodied_agent.py --config-path /workspace/RLinf/examples/embodiment/config` (ABSOLUTE path) with the TRAINING config. The eval reads only `rollout.model`, so pass `'rollout.model=${actor.model}'` (single-quoted; you cannot override `rollout.model.*` on top of the interpolation) and use `env.eval.rollout_epoch=1` (`algorithm.eval_rollout_epoch` is gone).
- **pi0 eval keeps `openpi.task=rl`** (Pi0RL with value head): the rollout worker strict-loads the PPO checkpoint, which has `value_head.*`. Upstream eval's `task: eval` (Pi0Eval) would not load it.
- **ManiSkill eval video dirs are named `seed_<function ManiskillOffloadEnv.__getattr__.<locals>.method_proxy at 0x...>`** (upstream cosmetic bug; the offload env proxies `seed`). `generate-showcase-video.sh` still works, but quote these paths (spaces, `<>`).
- **Wan 2-step run: ~40 min container time + image pull** (measured cold and warm on one fresh p5en, 2026-10: 39.8 min both). Breakdown: ~6.5 min container start → first rollout (Ray/worker init ~1 min; actor FSDP + model load + LIBERO env init ~5.5 min), step 1 ~916 s, step 2 ~1105 s (incl. 195 s eval). Cold adds only the image pull (18.6 GB, ~5 min) before the container starts. The v0.3 "~35 min" had no breakdown; an earlier 51-min run (~15 min silent init gap) did not reproduce. Silent log gaps of ~265 s/149 s during steps are the actor update and eval (no per-batch logging) — not hangs.
- **flash-attn required** for DiT attention layers. Must build with `--no-build-isolation` in container.

### DreamZero LIBERO 14B SFT (validated end-to-end on EKS, 2026-06; revalidated on main@c70606f0, 2026-10)

This example demonstrates the canonical customer workflow: **continue SFT from the
released DreamZero-DROID 14B checkpoint onto a new embodiment's data (LIBERO),
then eval in the LIBERO simulator.** It is FSDP2 + KubeRay RayJob multi-node (NOT the
old DROID/torchrun/DeepSpeed/StatefulSet design). Full pipeline: build → stage → SFT (2x p5en) →
DCP→.pt convert → LIBERO sim eval → in-sim rollout video.

- **Unified image: `BUILD_TARGET=embodied-maniskill_libero`.** At main@d844e648 the upstream target ships 13 venvs (openvla, openvla-oft, openpi, gr00t, gr00t_n1d6, gr00t_n1d7, dexbotic, starvla, abot_m0, dreamzero, molmoact2, evo1, fastwam) + ManiSkill/openpi assets. `examples/Dockerfile` requires `BUILD_TARGET=embodied-maniskill_libero` (which ships the `dreamzero` venv); the build fails otherwise (a one-line `test -d ${UV_PATH}/dreamzero` guard; no install logic). `VENV_NAME=dreamzero`. Pins in `examples/buildspec.yml`: `UPSTREAM_REF=d844e6485d9c00da497b1e1980996fe36da6fb1c` (main 2026-10-22, post-v0.4), `DREAMZERO_REF=ab790c198fbc`. No stage-1 build patches are needed at this pin. The `groot` package is external (`github.com/RLinf/dreamzero.git`), on PYTHONPATH via `DREAMZERO_PATH=/workspace/DreamZero`; the buildspec clones it.
- **`groot` package is external** (`github.com/RLinf/dreamzero.git`), provided via
  `DREAMZERO_PATH=/workspace/DreamZero` on PYTHONPATH. The integration glue
  (`rlinf/models/embodiment/dreamzero/`, world-model env) is in-tree. Buildspec clones it.
- **Multi-node uses a KubeRay `RayJob`** (embedded RayCluster: head 1 + worker 1), NOT torchrun and NOT a StatefulSet. The KubeRay operator starts Ray and runs the launcher (`run_dreamzero_sft_eks.sh`, Ray-agnostic) as the head `entrypoint`; there is no manual head election. `shutdownAfterJobFinishes: true` tears the cluster down on completion.
- **Apply manifests with RESTRICTED envsubst**: `envsubst '${IMAGE} ${NAMESPACE}' < ...`. The Ray head-election bash is gone (the KubeRay operator handles it), but the RayJob `entrypoint` still embeds an inline `bash -c`; unrestricted envsubst would mangle it.
- **PYTHONPATH unbound-variable trap**: every venv-activating context (launcher, Ray
  bootstrap, dreamzero-deps Dockerfile layer) must `export PYTHONPATH="${PYTHONPATH:-}"`
  BEFORE sourcing the venv; the activate script references PYTHONPATH under `set -u`.
- **Hydra `+` prefix for commented/absent keys**: `metadata_json_path` and
  `fsdp_config.save_full_model_weights` are NOT in the config struct → must be ADDED with
  `+actor.model.metadata_json_path=...` (plain override fails: "Key not in struct").
  This applies to the SFT launcher only; eval sets `rollout.model.metadata_json_path`
  with no `+` (in-struct there).
- **Hydra config group `training_backend/` was renamed `hybrid_engines/`** upstream (34cacb3f): `hybrid_engines/fsdp@actor.fsdp_config`. Upstream's `libero_sft_dreamzero_14b.yaml` got the same rename (the scalar `actor.training_backend: "fsdp"` is unchanged), so custom SFT configs that use `training_backend/fsdp@...` must be updated.
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
- **DCP finalization crash (HISTORY: torch 2.6 / RLinf v0.3, 2026-06; patch now removed)**: on torch 2.6,
  `dcp.save`'s post-write finalization (`_DistWrapper.all_reduce` → `broadcast_object` →
  `broadcast_object_list`) broadcast a multi-MB pickled result object over the **default
  (NCCL) PG on CUDA**. At the end of a long (~209GB / ~20min) checkpoint write this raced
  with NCCL comm teardown, so non-coordinator ranks read an all-zero buffer →
  `_pickle.UnpicklingError: invalid load key '\x00'` — AFTER all 16 shards + `.metadata`
  were on disk. At the time it was worked around with an in-image patch that passed a gloo PG
  to `dcp.save`. **Current state**: the torch 2.6 NCCL finalization race (`UnpicklingError`)
  does not reproduce on torch 2.11 (main@c70606f0); the patch was removed (validated: 2x p5en,
  16-shard 223.7 GB DCP save, 0 `UnpicklingError`, 2026-10). See the main@c70606f0 bullet below.
- **`save_full_model_weights` MUST be forced false on the 16B model**: `libero_sft_dreamzero_14b.yaml`
  omits `fsdp_config.save_full_model_weights`, so it falls through to the code default of
  `True` (`fsdp_model_manager.py`). On the 16B model that runs a full-state-dict gather →
  `RuntimeError: Backend nccl does not support allgather_into_tensor_coalesced` (and a rank-0
  gather that stalls). The launcher (`run_dreamzero_sft_eks.sh`) passes
  `+actor.fsdp_config.save_full_model_weights=false` by default; DCP-only + offline convert
  is the supported path. Do NOT set it true.
- **FSx free space is critical**: a 14B DCP checkpoint is ~207 GB on disk (223.7 GB apparent) (full optimizer
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
- **DreamZero 14B DCP checkpoint = ~207 GB on disk (223.7 GB apparent)** (16 FSDP shards incl. full optimizer state).
  Ensure FSx has >=250GB free; a full FS truncates `torch.save` → corrupt shards. (This is
  FSDP DCP, NOT the old DeepSpeed ZeRO-2 ~282GB figure.)
- **DreamZero eval OOMs at upstream's `total_num_envs=128`** for the 14B model (sim + 16B
  model co-located on 8x H200). Use `env.eval.total_num_envs=16` (our config default).

### DreamZero Eval & Showcase (LIBERO simulator)
- **Eval needs a 14B eval config**: upstream ships only `evaluations/libero/libero_spatial_dreamzero_eval.yaml`
  (5B). Our `examples/dreamzero/configs/libero_spatial_eval_dreamzero_14b.yaml` is its 14B
  variant: `model/dreamzero_14b@rollout.model` (resolved via a Hydra `searchpath` entry to
  `examples/sft/config/`, since the 14B model config lives only there) with null component
  paths (the full checkpoint provides weights), `target_video 352x640`, `num_action_per_block=16`.
- **Eval is rollout-only**: `evaluations/eval_embodied_agent.py` builds just rollout + env
  workers and reads `rollout.model.*` (no `actor`). The SFT model file interpolates
  `${actor.model.*}`, so our eval config re-binds those keys under
  `rollout.model.action_head_cfg` (otherwise: `InterpolationKeyError 'actor.model.action_dim'`).
  `metadata_json_path` IS in the eval struct, so the launcher sets it with no `+`. Pass an
  ABSOLUTE `--config-path` (hydra resolves relative paths against the script dir).
- **Eval `runner.ckpt_path` is the converted `.pt`** (`full_weights.pt`), NOT the DCP dir.
- **In-sim rollout video** is produced by the eval itself (`env.eval.video_cfg.save_video=True`)
  → `{log_path}/video/eval/seed_*/0.mp4`. This replaced the old offline DROID
  prediction/compose scripts (now removed).
- **Eval config is mounted via ConfigMap + copied** into `evaluations/libero/` (entrypoint `evaluations/eval_embodied_agent.py`) by the
  launcher (mounting over the dir would hide upstream config groups).
- **1-step checkpoint → `success_once: 0.0`** is expected; the eval gate validates the
  machinery, not accuracy. Real accuracy needs a multi-step SFT run.
- **Eval host RAM (main@c70606f0, first run 2026-10)**: each of the 8 rollout workers
  `torch.load`s the 91.7 GB `full_weights.pt` concurrently onto CPU, so the pod peaks at
  ~713 GB RSS for ~5 min before the weights move to GPU. The manifest's 1000Gi request /
  1100Gi limit covers it. Do NOT lower the eval pod's memory, and expect ~15 min of silent
  model load before the first `Time taken:` line. With `max_steps_per_rollout_epoch=960`, a
  smoke eval took 23 min end to end (32 trajectories); the 15360 default runs the full 500-episode suite.
- **DCP gloo patch is no longer needed (main@c70606f0, torch 2.11)**: the torch 2.6 NCCL
  finalization race (`UnpicklingError`) does not reproduce on torch 2.11: a 20-step SFT with the
  patch OFF on 2x p5en wrote the full 16-shard 223.7 GB DCP (+ `.metadata`), RayJob `SUCCEEDED`, 0
  `UnpicklingError`. (Mechanism unverified: torch 2.11 `dcp.save` still defaults to the default PG.) The patch file, `APPLY_DCP_GLOO_PATCH`, and the then-empty runtime
  patch loop in the Dockerfile were removed (stage-1 `upstream-build-patches/` is unaffected).

## Key Files

| File | Purpose |
|------|---------|
| `examples/Dockerfile` | Stage 2: EFA overlay on upstream RLinf image |
| `examples/buildspec.yml` | CodeBuild: two-stage build (upstream + EFA) |
| `examples/scripts/run_training_eks.sh` | Generic training launcher (any config/venv) |
| `examples/scripts/install_extras.sh` | Conditional package installer for EXTRAS build arg |
| `infrastructure/manifests/container-test.yaml` | Container validation (12 tests) |
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
