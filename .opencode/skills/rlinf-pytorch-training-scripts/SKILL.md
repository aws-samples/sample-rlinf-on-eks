---
name: rlinf-pytorch-training-scripts
description: "RLinf: Adapt RLinf training scripts for distributed execution on Amazon EKS with veRL, Ray, and FSDP"
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 08: PyTorch Training Scripts

## Purpose

Adapt Physical AI RL training scripts for execution on Amazon EKS. This involves configuring distributed training, storage paths, environment variables, and framework settings to work within the Kubernetes pod environment.

> **RLinf**: The working launch script is at [`examples/scripts/run_training_eks.sh`](/examples/scripts/run_training_eks.sh). The content below explains the patterns and adaptation decisions.

## RLinf Entrypoint

The training pipeline is launched via:

```bash
python -u -m verl.trainer.main_ppo <hydra overrides...>
```

Key files:
- **Entrypoint**: `verl/trainer/main_ppo.py` -- Ray init, worker creation, `RayTrainer.fit()`
- **Training loop**: `verl/trainer/ppo/ray_trainer.py` -- epoch loop, rollout, advantage computation, policy update
- **FSDP workers**: `verl/workers/fsdp_workers.py` -- model init, generate_sequences, update_actor, checkpointing
- **Rollout**: `verl/workers/rollout/rob_rollout.py` -- environment interaction, trajectory collection
- **Config**: `verl/trainer/config/ppo_trainer.yaml` -- Hydra base configuration

## Adaptations for EKS

### 0. Known Training Runtime Issues (Validated on 8x H100)

These issues affect RLinf specifically but the patterns are common across Physical AI RL stacks:

1. **flash_attn `cross_entropy_loss` returns wrong tensor shape** (Discovery #26): The Triton kernel returns a size-1 tensor for VLA action token logit slices. **Fix**: Set `FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False` in `verl/utils/torch_functional.py` at runtime.

2. **`filter_accuracy=True` with `accuracy_lower_bound=0.1` causes infinite loop** (Discovery #27): When model has 0% success rate (baseline), all tasks filtered out. **Fix**: Set `accuracy_lower_bound=0.0`.

3. **`filter_accuracy=False` causes `UnboundLocalError`** (Discovery #28): Bug in RLinf's training code. **Workaround**: Use `filter_accuracy=True` with bounds `[0.0, 1.0]`.

4. **`overwrite_vla_ckpt_utils.sh` destroys `processing_prismatic.py`** (Discovery #29): Copies ALL Python files, overwriting the HuggingFace `PrismaticProcessor` class. **Fix**: Restore from openvla-oft after running the script.

5. **Docker patches applied partially** (Discovery #30): TF replacements must also be applied at runtime via ConfigMap-mounted training script.

6. **HuggingFace cache must be cleared between runs** (Discovery #31): `~/.cache/huggingface/modules/transformers_modules/` and `__pycache__` on FSx persist across runs.

These are applied via ConfigMap-mounted scripts to avoid expensive Docker rebuilds.

### 1. Storage Path Configuration

Replace local paths with shared storage mounts:

```bash
# Original (local machine)
SFT_MODEL_PATH="/home/user/models/openvla-oft-libero10"
CKPT_PATH="/home/user/checkpoints"

# EKS (FSx for Lustre mount)
SFT_MODEL_PATH="/fsx/models/sft-base/openvla-oft-libero10-traj1"
CKPT_PATH="/fsx/checkpoints"

# EKS (Mountpoint for S3)
SFT_MODEL_PATH="/s3/models/openvla-oft-libero10-traj1"
CKPT_PATH="/s3/checkpoints"
```

### 2. Ray Configuration for Kubernetes

RLinf uses Ray for distributed worker management. Environment variables for the Ray runtime are configured in the training script or pod spec:

```json
{
  "env_vars": {
    "NCCL_DEBUG": "WARN",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "true",
    "FI_PROVIDER": "efa",
    "FI_EFA_USE_DEVICE_RDMA": "1",
    "FI_EFA_FORK_SAFE": "1",
    "NCCL_PROTO": "Simple",
    "NCCL_ALGO": "Ring,Tree",
    "RAY_memory_monitor_refresh_ms": "0",
    "TORCH_USE_CUDA_DSA": "1"
  },
  "excludes": ["*"]
}
```

The `excludes` key prevents Ray from uploading the working directory. `RAY_memory_monitor_refresh_ms=0` disables Ray's memory monitor.

Ray initializes within the pod. For multi-node, Ray head and workers run across pods. The `main_ppo.py` handles this:

```python
# main_ppo.py line 109-114
if config.trainer.runtime_env != 'none':
    with open(config.trainer.runtime_env, 'r') as f:
        runtime_env = json.load(f)
    ray.init(runtime_env=runtime_env)
else:
    ray.init()
```

### 3. Multi-Node Distributed Training

For multi-node training on EKS, each node runs one pod. The pods need to:
1. Discover each other (via headless Service)
2. Initialize NCCL with EFA
3. Coordinate via Ray

**Single-node (1 pod, 8 GPUs):**

```bash
# trainer.nnodes=1 trainer.n_gpus_per_node=8
# Ray starts local workers, FSDP across 8 GPUs
python -u -m verl.trainer.main_ppo \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  ...
```

**Multi-node (2 pods, 16 GPUs):**

```bash
# Pod 0 (head): starts Ray head, launches trainer
# Pod 1 (worker): connects to Ray head
#
# The Ray resource pool handles GPU allocation across nodes:
#   resource_pool_spec = {pool_id: [8] * 2}  # 8 GPUs x 2 nodes
python -u -m verl.trainer.main_ppo \
  trainer.nnodes=2 \
  trainer.n_gpus_per_node=8 \
  ...
```

### 4. FSDP Configuration for EKS

RLinf uses FSDP with aggressive memory optimization:

```yaml
# Actor (training) -- parameters stay on GPU, offload gradients + optimizer
actor_rollout_ref.actor.fsdp_config.param_offload: False
actor_rollout_ref.actor.fsdp_config.grad_offload: True
actor_rollout_ref.actor.fsdp_config.optimizer_offload: True

# Reference model -- offload everything to CPU (read-only)
actor_rollout_ref.ref.fsdp_config.param_offload: True
```

These settings are tuned for 80GB GPUs. On H100 (same VRAM), they should work without changes. If using H200 (141GB), you can disable CPU offload for faster training.

### 5. MLflow Configuration (Optional)

MLflow can be used for experiment tracking when the optional MLflow addon is enabled (`enable_mlflow = true` in `infrastructure/addons/terraform.tfvars`). On EKS, configure the tracking URI to point to the in-cluster MLflow server:

```yaml
env:
  - name: MLFLOW_TRACKING_URI
    value: "http://mlflow.rlinf.svc.cluster.local"
```

### 6. Complete Training Launch Script for EKS

```bash
#!/bin/bash
# run_libero_eks.sh -- adapted for Kubernetes pod execution
set -euo pipefail
set -x

# Environment (set via k8s env vars or here)
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export ROBOT_PLATFORM=LIBERO
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1

# Paths (pointing to shared storage)
PROJECT_NAME='RLinf'
EXPERIMENT_NAME='libero10-traj1-rl-eks'
SFT_MODEL_PATH="/fsx/models/sft-base/openvla-oft-libero10-traj1"
CKPT_PATH="/fsx/checkpoints"
DATASET_NAME="libero_10"
VLA_NAME="openvla-oft"
NUM_GPUS=8
NUM_NODES=1

# Pre-step: ensure VLA code files are in checkpoint dir
if [ -f /workspace/examples/overwrite_vla_ckpt_utils.sh ]; then
    bash /workspace/examples/overwrite_vla_ckpt_utils.sh "$SFT_MODEL_PATH"
fi

# Launch training
HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.n_samples=8 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=64 \
    data.val_batch_size=496 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','mlflow'] \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=25 \
    trainer.test_freq=4 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.val_before_train=True
```

## Key Hyperparameter Tuning Points

| Parameter | Default | Tuning Notes |
|-----------|---------|-------------|
| `data.train_batch_size` | 64 | Scale with GPU count. 64 = 8 tasks x 8 samples/task |
| `data.n_samples` | 8 | GRPO group size. Higher = better advantage estimation, more VRAM |
| `actor.ppo_micro_batch_size` | NUM_GPUS | Gradient accumulation. Must divide mini_batch_size |
| `actor.traj_mini_batch_size` | 16 | Trajectories per gradient step. Lower = less VRAM |
| `rollout.temperature` | 1.6 | Higher = more exploration. Paper found 1.6 optimal |
| `actor.clip_ratio_high` | 0.28 | Asymmetric GRPO clipping. Encourages exploration |
| `trainer.save_freq` | 25 | Checkpoint every N steps. Lower = more storage, safer |
| `trainer.test_freq` | 4 | Validate every N steps. More frequent = slower training |

## Validation Checklist

- [ ] Training script runs successfully in a single-node pod
- [ ] MLflow logging is active (metrics appearing in dashboard)
- [ ] Checkpoints are saved to shared storage at expected intervals
- [ ] Validation runs produce success rate metrics
- [ ] For multi-node: all nodes participate in FSDP training

## Related Skills

- [Skill 07: Container Image Building](container-image-building/SKILL.md) - Image containing these scripts
- [Skill 09: Kubernetes Manifests](kubernetes-manifests/SKILL.md) - Pod specs that launch these scripts
- [Skill 11: Open-Source Plugins](open-source-plugins/SKILL.md) - Ray/veRL configuration
