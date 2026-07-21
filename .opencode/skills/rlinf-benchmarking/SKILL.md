---
name: rlinf-benchmarking
description: "RLinf: Evaluate training quality and infrastructure performance using LIBERO, RoboTwin, and throughput benchmarks"
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 14: Benchmarking

## Purpose

Evaluate Physical AI RL training quality and infrastructure performance using standardized benchmarks. This covers both **task success rate** (model quality) and **training throughput** (infrastructure efficiency).

## Benchmarks

### LIBERO

LIBERO is the primary benchmark for RLinf. It evaluates lifelong robot learning across diverse manipulation tasks.

| Suite | Tasks | Demos/Task | Focus |
|-------|-------|-----------|-------|
| **LIBERO-Long** (libero_10) | 10 | 50 | Long-horizon sequential manipulation |
| **LIBERO-Spatial** | 10 | 50 | Spatial reasoning across layouts |
| **LIBERO-Object** | 10 | 50 | Object-centric manipulation |
| **LIBERO-Goal** | 10 | 50 | Goal-conditioned tasks |
| **LIBERO-90** | 90 | 50 | Large-scale multi-task |

### RoboTwin 2.0

Bimanual manipulation with domain randomization:

| Horizon | Tasks | Steps | Examples |
|---------|-------|-------|----------|
| Short | 4 | 112-130 | lift_pot, beat_block_hammer |
| Medium | 4 | 151-223 | move_can_pot, handover_mic |
| Long | 2 | 283-313 | handover_block, stack_bowls_two |
| Extra-Long | 2 | 466-637 | blocks_rank_rgb, put_bottles_dustbin |

## Running Benchmarks on EKS

### LIBERO Evaluation

```yaml
# eval-libero.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: eval-libero-long
spec:
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        role: gpu-training
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: eval
          image: <ECR_URI>:latest
          command:
            - bash
            - -c
            - |
              # Point to the RL-trained checkpoint
              MODEL_PATH="/fsx/checkpoints/libero10-rl/actor/global_step_100"

              bash examples/overwrite_vla_ckpt_utils.sh $MODEL_PATH

              HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
                data.task_suite_name=libero_10 \
                data.val_batch_size=496 \
                actor_rollout_ref.model.path=$MODEL_PATH \
                actor_rollout_ref.model.vla=openvla-oft \
                actor_rollout_ref.model.action_token_len=7 \
                actor_rollout_ref.model.action_chunks_len=8 \
                actor_rollout_ref.rollout.temperature=1.0 \
                actor_rollout_ref.rollout.task_suite_name=libero_10 \
                actor_rollout_ref.rollout.pretrained_checkpoint=$MODEL_PATH \
                trainer.val_only=True \
                trainer.val_before_train=True \
                trainer.n_gpus_per_node=8 \
                trainer.nnodes=1
          resources:
            requests:
              nvidia.com/gpu: 8
              vpc.amazonaws.com/efa: 4
            limits:
              nvidia.com/gpu: 8
              vpc.amazonaws.com/efa: 4
          volumeMounts:
            - name: fsx
              mountPath: /fsx
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: fsx
          persistentVolumeClaim:
            claimName: fsx-claim
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: "64Gi"
```

### Evaluation Protocol

RLinf evaluation uses:
- **Greedy decoding** (do_sample=False, temperature=1.0 with greedy)
- **50 held-out test scenarios** per task
- **3 evaluation runs** for reproducibility
- **Success Rate (SR)** as the primary metric

## Reference Results

### LIBERO (from the paper)

| Model | Spatial | Object | Goal | Long | Avg |
|-------|---------|--------|------|------|-----|
| OpenVLA-OFT (SFT) | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
| **+ RLinf** | **99.4** | **99.1** | **99.2** | **98.5** | **99.1** |
| One-Traj SFT | 63.6 | 54.9 | 59.6 | 17.3 | 48.9 |
| **One-Traj + RL** | **98.2** | **98.7** | **98.8** | **91.7** | **96.9** |

### RoboTwin 2.0 (from the paper)

| Model | Short | Medium | Long+XL | Overall Avg |
|-------|-------|--------|---------|-------------|
| OpenVLA-OFT (SFT) | 21.3 | 47.1 | 46.5 | 38.3 |
| **+ RLinf** | **64.9** | **72.5** | **69.0** | **68.8** |

## Infrastructure Performance Benchmarks

### GPU Utilization

Track GPU utilization during training to identify bottlenecks:

```bash
# In training pod
watch -n 1 nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
```

Target:
- GPU utilization: >70% during policy update, lower during rollout (expected)
- Memory utilization: >80% (efficient VRAM usage)

### NCCL AllReduce Bandwidth

```bash
# Run nccl-tests between nodes
/usr/bin/all_reduce_perf -b 8 -e 1G -f 2 -g 8
```

| Instance | Expected Inter-Node BusBW (1GB msg) |
|----------|-------------------------------------|
| p4de.24xlarge | 40-50 GB/s |
| p5.48xlarge | 300+ GB/s |

### Training Throughput

Measure end-to-end training speed:

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Steps per hour | MLflow `global_step` / wall time | Depends on batch size and envs |
| Rollout time | MLflow `timing/generate` | Bottleneck is env rendering |
| Update time | MLflow `timing/update` | Scales with batch size |
| Checkpoint time | MLflow `timing/save` | Should be < 60s with FSx |

### Storage I/O

```bash
# Sequential write throughput (checkpoint simulation)
dd if=/dev/zero of=/fsx/benchmark_write bs=1M count=4096 oflag=direct
# Expected: >500 MB/s for 4.8 TiB filesystem at 125 MB/s/TiB

# Sequential read throughput (model loading simulation)
dd if=/fsx/benchmark_write of=/dev/null bs=1M count=4096 iflag=direct
```

## Comparing EKS vs Paper Results

When reproducing paper results on EKS, expect:

| Factor | Paper (A800 80GB) | EKS (A100 80GB) | EKS (H100 80GB) |
|--------|-------------------|------------------|------------------|
| **Success rates** | Baseline | Equivalent | Equivalent |
| **Training speed** | Baseline | Similar | ~2x faster |
| **NCCL bandwidth** | NVLink (intra) | NVSwitch (intra) | NVSwitch (intra) |
| **Inter-node** | InfiniBand | EFA 400 Gbps | EFA 3200 Gbps |

Success rates should be equivalent across hardware. Training speed depends on GPU generation and interconnect.

## Benchmark Automation

### Scheduled Evaluation

Run evaluation automatically after each checkpoint save:

```bash
# In a CronJob or triggered by a webhook
for step in 25 50 75 100; do
  CKPT="/fsx/checkpoints/${EXPERIMENT}/actor/global_step_${step}"
  if [ -d "$CKPT" ]; then
    kubectl create job eval-step-${step} --from=job/eval-libero-long \
      -- --env MODEL_PATH=$CKPT
  fi
done
```

### Results Collection

Evaluation results are logged to MLflow. Extract programmatically:

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient("http://mlflow.mlflow.svc.cluster.local:5000")
experiment = client.get_experiment_by_name("RLinf")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.val_only = 'True'"
)
for run in runs:
    metrics = run.data.metrics
    print(f"{run.info.run_name}: SR={metrics.get('success_rate', 'N/A')}")
```

## Validation Checklist

- [ ] LIBERO evaluation runs and produces per-task success rates
- [ ] Success rates are in expected range (compare to paper)
- [ ] RoboTwin evaluation runs (if applicable)
- [ ] GPU utilization > 70% during policy updates
- [ ] NCCL bandwidth matches expectations for instance type
- [ ] Checkpoint save/load time is acceptable (< 60s)
- [ ] Results are logged to MLflow for comparison

## Related Skills

- [Skill 08: PyTorch Training Scripts](pytorch-training-scripts/SKILL.md) - Training configuration
- [Skill 13: Functional Testing](functional-testing/SKILL.md) - Pre-benchmark validation
- [Skill 15: Monitoring and Observability](monitoring-observability/SKILL.md) - GPU/training metrics
