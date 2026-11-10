<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# LIBERO-10 π₀-FAST GRPO on EKS (2x p5en, 16 H200)

Multi-node RL fine-tuning of [LeRobot π₀-FAST](https://huggingface.co/lerobot/pi0fast-libero) on **LIBERO-10 (Long)** with **token-level GRPO**, using RLinf's two-node, 16-GPU reference configuration ([upstream docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0_fast.html)). Upstream reports `success_once` around 96% after 330 steps (one seed; base policy 85.8%).

> **Validated on EKS 1.37 (2026-11):** 2x p5en.48xlarge, README steps 2–4 as written (`./validate.sh --mode cluster --skip-to 7 --level 7 --example libero-pi0fast-grpo`, `pi0-fast` image built with step 1). One GRPO step takes **~50 min** at the reference config (1,024 trajectories): rollouts 2,500 s (4 epochs × ~620 s), actor update 472 s. Step 1: train `success_once` 0.876, replay log-probabilities finite (fraction 1.000), ratio 0.971, `approx_kl` 0.0037. Step 2 (3,063 s incl. the final eval): train `success_once` 0.865, ratio 1.003; **eval `success_once` 0.852** over 256 fixed episodes (upstream base policy: 85.8% over 500), RayJob `SUCCEEDED` in 1 h 54 min.

## Overview

| Property | Value |
|----------|-------|
| **Model** | `lerobot/pi0fast-libero` (PaliGemma-based, autoregressive FAST action tokens), all-linear LoRA rank 16 |
| **Algorithm** | GRPO with token-level PPO clipping, `group_size` 8 |
| **Environment** | LIBERO-10 (Long), 256 parallel envs × 4 rollout epochs = 1,024 trajectories per update |
| **Hardware** | 2x p5en.48xlarge (16x H200 141GB), 16 EFA per node |
| **Config** | `libero_10_grpo_pi0_fast` (RLinf main@d844e648) |
| **Image** | `pi0-fast` profile: Python 3.12.12, torch 2.11.0, no flash-attn (upstream's validated runtime) |

## How It Works

π₀-FAST generates its whole action chunk as a token sequence (FAST action tokenizer, `jadechoghari/tokenizer-lib-mean`), so RL treats it like a language model: rollouts sample tokens at temperature 0.3, and the actor replays them with teacher forcing to get per-token log-probabilities. Every token of a trajectory shares the trajectory's GRPO advantage (its success relative to the other 7 trajectories from the same initial state), with PPO clipping per token. Malformed token sequences are not resampled: they execute a zero action and keep their (failing) reward, which keeps sampling on-policy.

All RL components (LIBERO env workers, rollout, actor) are colocated on all 16 GPUs (`actor,env,rollout: all`). The actor is FSDP `NO_SHARD` with FP32 master weights: data parallel over 16 GPUs, gradients all-reduced over NCCL on EFA between the two nodes.

## Cost and duration

At ~50 min per step, upstream's 330-step schedule is ~11.5 days on 2x p5en.48xlarge. A Capacity Block for ML is the practical way to hold the pair for that long. For pipeline checks use the smoke edit: 2 steps plus the final eval, 1 h 54 min end to end including ~10 min startup.

| Phase (step 1) | Seconds |
|---|---|
| Rollouts (`generate_rollouts`, 4 epochs × 256 envs × 520 steps) | 2,501 |
| of which policy inference (`rollout/predict`) | 2,221 |
| Actor update (`actor_training`) | 472 |
| Weight sync actor → rollout | 14 |
| **Step** | **2,986** |

Rollouts are inference-bound (autoregressive FAST token generation, `generation_backend: huggingface`).

## Architecture

KubeRay `RayJob`: a Ray head and one Ray worker, one per p5en.48xlarge (`podAntiAffinity`), each requesting 8 GPUs and 16 EFA. RLinf on the head connects to the running Ray cluster and places workers on both nodes. Same pattern as the [DreamZero](../dreamzero/) SFT example.

## Reproduce

### Prerequisites

- 2x p5en.48xlarge reachable by the `gpu-training` NodePool (see "GPU Capacity" in the main [README](../../README.md); a Capacity Block is the usual way to get them), FSx `fsx-claim`, and `training-sa`, all created by `infrastructure/deploy.sh`.
- A Hugging Face token in the `hf-token` Secret whose account has accepted the [PaliGemma license](https://huggingface.co/google/paligemma-3b-pt-224) (gated; without it the download fails with `GatedRepoError: 403`).

```bash
export NAMESPACE=rlinf
export IMAGE=$(terraform -chdir=infrastructure/build output -raw ecr_repository_url):pi0-fast-latest
```

### 1. Build the image

```bash
zip -r rlinf-on-eks.zip examples/Dockerfile examples/Dockerfile.profile examples/profiles.env examples/buildspec.yml examples/scripts/
aws s3 cp rlinf-on-eks.zip s3://$(terraform -chdir=infrastructure/build output -raw codebuild_source_bucket)/rlinf-on-eks.zip
aws codebuild start-build --project-name $(terraform -chdir=infrastructure/build output -raw codebuild_project_name) \
  --environment-variables-override name=PROFILE,value=pi0-fast,type=PLAINTEXT
```

### 2. Download model weights

```bash
envsubst '${NAMESPACE}' < examples/libero-pi0fast-grpo/manifests/model-download.yaml | kubectl apply -f -
kubectl wait -n "$NAMESPACE" job/model-download-libero-pi0fast --for=condition=complete --timeout=60m
```

Stages `/fsx/models/pi0fast-libero`, `/fsx/models/paligemma-3b-pt-224` (11 GB) and `/fsx/models/tokenizer-lib-mean`.

### 3. Launch training

```bash
kubectl -n "$NAMESPACE" create configmap libero-pi0fast-grpo-launcher \
  --from-file=run_pi0fast_grpo_eks.sh=examples/libero-pi0fast-grpo/scripts/run_pi0fast_grpo_eks.sh \
  --dry-run=client -o yaml | kubectl apply -f -

envsubst '${IMAGE} ${NAMESPACE}' \
  < examples/libero-pi0fast-grpo/manifests/libero-pi0fast-grpo.yaml | kubectl apply -f -
```

The manifest runs upstream's full schedule (330 steps, eval and checkpoint every 10). For a short smoke run, append `runner.max_steps=2 runner.save_interval=-1` to the `HYDRA_OVERRIDES` value in the manifest before applying it (this is what `./validate.sh --mode cluster --level 7` does).

### 4. Monitor

```bash
kubectl wait -n "$NAMESPACE" rayjob/libero-pi0fast-grpo --for=jsonpath='{.status.jobDeploymentStatus}'=Complete --timeout=24h
kubectl get -n "$NAMESPACE" rayjob/libero-pi0fast-grpo -o jsonpath='{.status.jobStatus}{"\n"}'   # expect: SUCCEEDED
# Training log, also tee'd to /fsx/checkpoints/libero-pi0fast-grpo/train-<ts>.log:
#   kubectl logs -f -n "$NAMESPACE" job/libero-pi0fast-grpo
```

Watch `env/success_once`, `eval/success_once`, the group-success histogram, token entropy, `approx_kl` and the log-ratio statistics. The first actor update should have finite replay log-probabilities and a ratio close to one.

## File Layout

```
examples/libero-pi0fast-grpo/
├── README.md
├── manifests/
│   ├── model-download.yaml        # pi0fast-libero, paligemma-3b-pt-224 (gated), tokenizer-lib-mean
│   └── libero-pi0fast-grpo.yaml   # RayJob: 2x p5en.48xlarge, 16 GPUs, 16 EFA each
└── scripts/
    └── run_pi0fast_grpo_eks.sh    # RayJob entrypoint (mirrors upstream run_embodiment.sh)
```
