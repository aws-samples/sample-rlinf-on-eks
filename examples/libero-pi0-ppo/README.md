<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# LIBERO Spatial + pi0 PPO on EKS

Single-node RL (reinforcement learning) fine-tuning of [pi0](https://www.physicalintelligence.company/blog/pi0) (3B flow-matching VLA (Vision-Language-Action model)) on LIBERO (Lifelong Embodied Robot Operation) spatial manipulation tasks using Proximal Policy Optimization (PPO).

> **Validated on EKS.** This deployment was run end-to-end on Amazon EKS; PPO fine-tuning converged to an **81.6% peak success rate**. Use it as a starting point for your own training runs (see [Results](#results) for the reference run).

<video src="assets/libero_pi0_success.mp4" width="720" autoplay loop muted></video>

*32 parallel LIBERO-Spatial evaluation episodes from the trained pi0 policy (agentview camera). Each tile shows the robot arm completing a different spatial manipulation task — picking objects from specific locations and placing them at target positions.*

## Overview

| Property | Value |
|----------|-------|
| **Model** | pi0 (3B params: PaliGemma VLM (Vision-Language Model) + flow-matching action expert) |
| **Algorithm** | PPO (Proximal Policy Optimization) |
| **Environment** | LIBERO Spatial (10 tasks, MuJoCo physics) |
| **Hardware** | 1x g7e.12xlarge (2x RTX PRO 6000 96GB) by default; 1x p5en.48xlarge (8x H200 141GB) for the reference run |
| **EFA** | Not used (single node). The `gpu-training` pool supports EFA (Elastic Fabric Adapter) for multi-node runs |
| **Config** | `libero_spatial_ppo_openpi_quickstart` |
| **SFT Checkpoint** | `RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal` |

## How It Works

### pi0: Flow-Matching Action Generation

pi0 is architecturally distinct from the OpenVLA family. Instead of generating actions as discrete tokens through an autoregressive language model, pi0 uses a **flow-matching action expert** — a small denoising transformer that iteratively refines a noise vector into a precise action chunk. The VLM backbone (PaliGemma 2B) processes images and task language into a rich context embedding, which conditions the action expert's denoising process.

The key advantage is that flow matching naturally outputs continuous actions without binning artifacts, and predicts entire **action chunks** (H future timesteps simultaneously) rather than single-step actions. This temporal coherence improves manipulation quality for tasks requiring smooth, coordinated motions.

### PPO with Flow Log-Probabilities

Applying PPO to a flow-matching policy requires computing log-probabilities differently than with discrete token models. Instead of summing token log-probs, the policy's log-probability is derived from the flow matching score function — the learned denoising direction at each noise level. The clipped PPO surrogate objective operates on these flow log-probs, with advantages computed via GAE (Generalized Advantage Estimation) adapted for the action-chunk temporal structure.

### LIBERO-Spatial Tasks

LIBERO provides 10 tabletop manipulation tasks requiring spatial reasoning — picking objects from specific locations, placing them in target positions, and navigating around obstacles. Each task specifies a natural language goal (e.g., "pick up the red mug and place it on the shelf"). The environment uses robosuite 1.4.1 with MuJoCo physics, providing sparse rewards (success/failure) that make RL challenging but closer to real-world evaluation.

## Architecture

### Infrastructure

![RLinf PPO Infrastructure](../diagrams/infra-rlinf-ppo.drawio.svg)

The infrastructure is shared with the other single-node examples (ManiSkill, π₀, Wan). A single GPU node (g7e.12xlarge by default, p5en.48xlarge for the reference run) is provisioned by Karpenter when the training Job is submitted. The pod runs all RL components colocated on the node's GPUs (2 RTX PRO 6000 on g7e.12xlarge, 8 H200 on p5en), with FSx for Lustre providing shared access to model weights, LIBERO dataset, and checkpoints. Unlike the ManiSkill examples which use GPU-accelerated simulation, LIBERO uses **CPU-based MuJoCo rendering** (`MUJOCO_GL=osmesa`), so this example uses **combined GPU placement** — actor, environment, and rollout all share every GPU on the node rather than splitting across subsets. The 64 parallel environments run entirely on CPU cores, making rollout throughput CPU-bound rather than GPU-bound. This is why the step time is about the same as ManiSkill's despite half as many environments (v0.3 reference runs: ~228s here with 64 envs vs ~230s for the 128-env ManiSkill OFT example). FSx stores additional LIBERO-specific data: the full dataset directory (`/fsx/datasets/libero`) and a pre-created config file that prevents LIBERO's `__init__.py` from hanging on an interactive `input()` call at import time. The pod sets `ROBOT_PLATFORM=LIBERO` and `LIBERO_DATASET_DIR` environment variables that are not needed by the ManiSkill examples.

### Model & Algorithm

![LIBERO pi0 PPO Model](diagrams/model-libero-pi0-ppo.drawio.svg)

The model diagram shows the flow-matching PPO loop specific to pi0. SigLIP encodes multi-view RGB observations (agentview + eye-in-hand cameras) into vision tokens, which PaliGemma 2B fuses with task language to produce a context embedding. The flow-matching action expert — a separate Gemma 300M transformer — iteratively denoises a random vector into an action chunk (H future timesteps) over 10 integration steps (δ=0.1), conditioned on this embedding. LIBERO returns observations and sparse rewards that populate the on-policy rollout buffer (trajectories are discarded after each PPO update). The central challenge is computing log-probabilities for a flow model's continuous outputs: RLinf uses the **Flow-SDE (Stochastic Differential Equation)** method, which converts the deterministic ODE (Ordinary Differential Equation) denoising trajectory into a stochastic differential equation, enabling both tractable log-likelihood estimation and built-in exploration. PPO then computes GAE advantages adapted for the action-chunk temporal structure and updates the action expert's parameters through the clipped surrogate loss — with `train_expert_only: True`, the PaliGemma VLM backbone remains frozen during RL, and only the 300M-parameter action expert receives gradient updates.

## Running on g-class (default)

The training manifest runs the quickstart config (32 envs, actor/env/rollout on all GPUs) at `actor.micro_batch_size=32` on **one g7e.12xlarge** (2x RTX PRO 6000 96 GB) in the `gpu-sim` pool (`enable_sim_pool = true`). Measured on EKS 1.37 (2026-11), 2 PPO steps each:

| Instance | GPUs | Step (s) | Training (s) | Rollouts (s) | $/hr (on-demand, us-west-2) | $/step |
|---|---|---|---|---|---|---|
| **g7e.12xlarge** (default) | 2x RTX PRO 6000 96 GB | **429–445** | 153–162 | 259–280 | 8.29 | **~1.0** |
| p5en.48xlarge | 8x H200 141 GB | ~192–224 | — | — | 63.30 | ~3.7 |

The g7e step is ~2x slower but ~3.7x cheaper per step, and it needs no Capacity Block. Train `success_once` was 0.625 and 0.547 for the two steps (the SFT-start reference is 0.578). To use p5en instead, see the comment at the top of `manifests/libero-pi0-ppo.yaml`. The convergence run below was produced on p5en.

## Results

> Reference run produced on RLinf v0.3. Revalidated on RLinf main@c70606f0 (2026-10) with a short 5-step end-to-end run (SFT-start success 57.8% — step-1 train-rollout `success_once`, since upstream has no step-0 eval; step time ~192–224 s vs ~228 s on v0.3 — within 10%); full convergence was not re-run. Re-checked on main@d844e648 (2026-10-22) with the same 5-step run: step-1 train `success_once` 65.6%, step times 222.4 / 189.9 / 192.8 / 193.5 s, eval@5 (128 envs) `success_once` 66.4%. The curve and numbers below are from the v0.3 run. If you enable eval, pass `env.eval.total_num_envs=128` (see `examples/AGENTS.md`).

![Training Curve](assets/training_curve.png)

| Metric | Value |
|--------|-------|
| **Peak success_once** | 81.6% (step 189) |
| **Final success_once** | 77.8% (step 299) |
| **Steps trained** | 300 |
| **Step time** | ~228s |
| **Total training time** | ~19 hours |
| **Placement** | Combined: actor, env, rollout all on 0-7 |
| **Environments** | 64 parallel (MuJoCo CPU-based) |
| **Global batch size** | 320 |

Pi0's training curve shows more variance than the ManiSkill examples because LIBERO tasks are harder — 10 distinct spatial manipulation scenarios with different objects, target positions, and required strategies. The flow-matching policy starts at approximately 59% success from SFT and improves to a peak of 81.6% with PPO. The variance (±10%) reflects the diversity of the task suite: some tasks reach near-perfect success while others remain challenging.

To regenerate this chart from your own training run:

```bash
python examples/scripts/plot_training_curve.py \
    --logdir /fsx/checkpoints/<experiment>/tensorboard \
    --metric success_once \
    --output examples/libero-pi0-ppo/assets/training_curve.png \
    --title "LIBERO + pi0 PPO — Training Progress"
```

## Reproduce

### Prerequisites

- EKS (Elastic Kubernetes Service) cluster with the `gpu-sim` pool enabled (`enable_sim_pool = true`, g7e.12xlarge), or the `gpu-training` pool with p5en.48xlarge
- FSx for Lustre with pre-staged model weights (`rlinf-pi0-sft-spatial`) and LIBERO dataset
- Container image built and pushed to ECR (Elastic Container Registry)

### 1. Download Model Weights

Downloads the pi0 SFT checkpoint to FSx. The LIBERO simulator dataset is **not**
downloaded here — LIBERO auto-downloads its assets to `LIBERO_DATASET_DIR`
(`/fsx/datasets/libero`) on the first training run. To pre-stage it for faster
startup, see the `rlinf-dataset-preparation` skill.

```bash
export NAMESPACE=rlinf  # or your namespace
envsubst '${NAMESPACE}' < examples/libero-pi0-ppo/manifests/model-download.yaml | kubectl apply -f -
kubectl logs -f job/model-download-libero-pi0 -n $NAMESPACE
```

### 2. Launch Training

```bash
export IMAGE=$(terraform -chdir=infrastructure/build output -raw ecr_repository_url):latest   # or :<profile>-latest
export NAMESPACE=rlinf
envsubst '${IMAGE} ${NAMESPACE}' < examples/libero-pi0-ppo/manifests/libero-pi0-ppo.yaml | kubectl apply -f -
```

For a short smoke run, append `runner.max_steps=2 runner.save_interval=-1` to the `HYDRA_OVERRIDES` value in the training manifest before applying it (this is what `./validate.sh --mode cluster --level 7` does).

### 3. Monitor

```bash
kubectl logs -f job/rlinf-libero-pi0 -n $NAMESPACE
```

## Regenerating the Showcase Video

The showcase video at the top of this README can be regenerated from any training checkpoint using the shared script:

```bash
export IMAGE=$(terraform -chdir=infrastructure/build output -raw ecr_repository_url):latest   # or :<profile>-latest
export NAMESPACE=rlinf
export S3_BUCKET=$(terraform -chdir=infrastructure/storage output -raw s3_bucket_name)
export CKPT_PATH=/fsx/checkpoints/rlinf-eks/libero_spatial_ppo_openpi/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt   # <CKPT_PATH>/<EXPERIMENT_NAME>/<runner.logger.experiment_name> from the training Job

./examples/scripts/generate-showcase-video.sh libero-pi0-ppo
```

The script deploys `manifests/eval-video.yaml` as a Kubernetes Job, waits for completion, downloads the raw video from S3, re-encodes it, and places the result in `assets/libero_pi0_success.mp4`. No frame stripping is needed for this example (action chunks = 1). The video is rendered at 256x256 (LIBERO's native resolution) at 30fps.

**Requirements**: `kubectl`, `aws` CLI, `envsubst`, and optionally `ffmpeg` (for re-encoding). The cluster must have a GPU node available and model weights + LIBERO dataset pre-staged on FSx.

## File Layout

```
examples/libero-pi0-ppo/
├── README.md               # This file
├── assets/                 # Training curves and showcase videos
│   ├── training_curve.png
│   └── libero_pi0_success.mp4
├── diagrams/
│   ├── model-libero-pi0-ppo.drawio     # Model/algorithm diagram
│   └── model-libero-pi0-ppo.drawio.svg
├── manifests/
│   ├── eval-video.yaml              # Showcase video eval Job
│   ├── model-download.yaml              # Model weight staging
│   └── libero-pi0-ppo.yaml             # Training Job
└── video_artifacts/        # Raw eval dumps (gitignored)
```
