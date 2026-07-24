---
name: functional-testing
description: Validate the Physical AI RL training stack end-to-end from infrastructure smoke tests to training step verification
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 13: Functional Testing

## Purpose

Validate the RL training stack end-to-end before launching production training runs. This includes smoke tests for individual components, integration tests for the full pipeline, and regression checks after infrastructure or code changes.

> **RLinf Example:** See the `rlinf-functional-testing` skill for RLinf-specific L5 test configurations and expected outputs.

## Testing Levels

```
Level 0: Infrastructure  -->  GPU, EFA, storage accessible
Level 1: Runtime          -->  PyTorch, CUDA, libraries working
Level 2: Single-GPU       -->  Model loads, forward pass succeeds
Level 3: Multi-GPU        -->  FSDP wrapping, NCCL allreduce
Level 3.5: EFA Validation -->  NCCL over EFA (Phase 0, cheap instances)
Level 4: Rollout          -->  Simulation env renders, actions generated
Level 5: End-to-End       -->  Full training step completes (rollout --> update)
Level 6: Validation       -->  Evaluation produces success rate metrics
```

## Phase 0: EFA Validation Sprint (MUST DO FIRST)

> **CRITICAL**: Always validate EFA/NCCL on cheap instances (g6.8xlarge, ~$1/hr) before purchasing expensive capacity blocks (p5.48xlarge, ~$725/hr). A failed $1,450 capacity block taught this lesson.

### Prerequisites

1. EKS cluster with GPU nodes (2x g6.8xlarge for cheapest test)
2. NVIDIA GPU Operator running
3. EFA device plugin running
4. Kubeflow MPI Operator installed (`enable_mpi_operator = true` in `terraform.tfvars`)

### Build NCCL Test Image

A pre-built image is available on public ECR — no need to build your own:

```bash
export NCCL_TEST_IMAGE=public.ecr.aws/hpc-cloud/nccl-tests:cuda12.8.1-efa1.43.2-ofiv1.16.3-ncclv2.27.7-1-testsv2.16.9
```

### Run NCCL Test (MPIJob)

```bash
envsubst '${NCCL_TEST_IMAGE}' < infrastructure/manifests/nccl-tests-mpijob.yaml | kubectl apply -f -

# Watch pods
watch kubectl get pods -o wide

# View logs
kubectl logs -f $(kubectl get pods | grep launcher | cut -d ' ' -f 1)
```

### What to Check

1. **EFA provider selected**: Look for `NET/OFI Selected Provider is efa` (NOT `socket`)
2. **No crashes**: No `cm_req_handle_error_entry` or `Caught signal 11`
3. **Reasonable bandwidth**: g6.8xlarge should show ~2-3 GB/s BusBW at 1GB
4. **All GPUs participating**: Verify `-np` matches expected GPU count

### Cleanup

```bash
kubectl delete mpijob nccl-tests
```

Only after Phase 0 passes, proceed to purchase capacity blocks for production training.

## Level 0: Infrastructure Smoke Tests

### GPU Availability

```bash
kubectl run gpu-test --rm -it \
  --image=nvcr.io/nvidia/pytorch:24.07-py3 \
  --overrides='{
    "spec": {
      "nodeSelector": {"role": "gpu-training"},
      "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
      "containers": [{
        "name": "gpu-test",
        "image": "nvcr.io/nvidia/pytorch:24.07-py3",
        "command": ["nvidia-smi"],
        "resources": {"requests": {"nvidia.com/gpu": "8"}, "limits": {"nvidia.com/gpu": "8"}}
      }]
    }
  }'
```

Expected: 8 GPUs visible with correct type (A100 80GB or H100).

### EFA Availability

```bash
kubectl exec <gpu-pod> -- ls /sys/class/infiniband/
# Expected: efa0, efa1, ... (4 for p4de, 32 for p5)
```

### Storage Availability

```bash
kubectl exec <gpu-pod> -- bash -c "
  echo 'Write test...' && echo 'test' > /fsx/test_write && \
  echo 'Read test...' && cat /fsx/test_write && \
  echo 'Delete test...' && rm /fsx/test_write && \
  echo 'Storage OK'
"
```

## Level 1: Runtime Validation

```bash
kubectl exec <gpu-pod> -- python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'GPUs: {torch.cuda.device_count()}')
assert torch.cuda.is_available(), 'CUDA not available!'
assert torch.cuda.device_count() == 8, f'Expected 8 GPUs, got {torch.cuda.device_count()}'

# Test key imports
import verl  # Replace with your RL framework
from flash_attn import flash_attn_func
import transformers
import ray
print('All imports OK')
"
```

## Level 2: Single-GPU Model Test

```bash
kubectl exec <gpu-pod> -- python -c "
import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer

model_path = '/fsx/models/openvla-7b'  # Replace with your model path
print(f'Loading model from {model_path}...')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='cuda:0'
)

params = sum(p.numel() for p in model.parameters())
print(f'Model loaded: {params / 1e9:.1f}B parameters')
print(f'Tokenizer vocab: {tokenizer.vocab_size}')
assert params > 6e9, f'Model too small: {params}'
print('Single-GPU model test PASSED')
"
```

## Level 3: Multi-GPU FSDP Test

```bash
kubectl exec <gpu-pod> -- python -c "
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# Init process group for single-node multi-GPU
dist.init_process_group('nccl', rank=0, world_size=1)

# Test NCCL allreduce
tensor = torch.ones(1024, device='cuda:0')
dist.all_reduce(tensor)
print(f'AllReduce result: {tensor[0].item()} (expected 1.0)')

dist.destroy_process_group()
print('Multi-GPU NCCL test PASSED')
"
```

### NCCL Performance Test (Multi-Node via MPIJob)

> **Discovery #46**: The canonical NCCL test on EKS uses an MPIJob (Kubeflow MPI Operator), NOT a StatefulSet with torchrun.

See `infrastructure/manifests/nccl-tests-mpijob.yaml`. Adjust for your instance type:

| Setting | g6.8xlarge | p5.48xlarge |
|---------|-----------|-------------|
| `slotsPerWorker` | 1 | 8 |
| `-np` | 2 | 16 |
| `nvidia.com/gpu` | 1 | 8 |
| `vpc.amazonaws.com/efa` | 1 | 32 |

Expected results (2x p5.48xlarge):
- 1 GB message: ~437 GB/s BusBW
- 16 GB message: ~487 GB/s BusBW

## Level 4: Rollout Test

Test that the simulation environment initializes and the model generates actions:

```bash
kubectl exec <gpu-pod> -- python -c "
import os, sys

# CRITICAL: Many robotics simulation libraries call input() at module import
# time if config directories aren't set. This will HANG in non-interactive
# containers. Always set env vars AND redirect stdin before importing.
os.environ['ROBOT_PLATFORM'] = 'LIBERO'  # Replace with your simulation platform
os.environ['LIBERO_DATASET_DIR'] = '/fsx/datasets/libero'  # Replace with your dataset env var and path
sys.stdin = open('/dev/null')  # Prevent input() hang

# Test simulation environment
import libero.libero  # Replace with your simulation library
# ... load benchmark/tasks and verify
print('Rollout environment test PASSED')
"
```

> **Example (RLinf):** The RLinf reference tests LIBERO with `os.environ['ROBOT_PLATFORM'] = 'LIBERO'`, `os.environ['LIBERO_DATASET_DIR'] = '/fsx/datasets/libero'`, then imports `libero.libero` and verifies 10 tasks via `get_benchmark('libero_10')`. See `rlinf-functional-testing` skill for details.

> **General pattern**: Always test simulation library imports in a non-interactive container (Kubernetes pod, CI) before assuming they work. Many robotics libraries assume interactive terminals.

## Level 5: End-to-End Training Step

Run a minimal training step (1 epoch, small batch):

```yaml
# test-training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: test-training-e2e
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 3600  # 1 hour timeout
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
        - name: test
          image: <ECR_URI>:latest
          command:
            - bash
            - -c
            - |
              # Run framework-specific pre-training setup
              # Then launch a single training step with minimal data
              bash /workspace/eks/scripts/run_training_eks.sh --max-steps 1  # Replace with your launch script
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

> **Example (RLinf):** The RLinf L5 test runs `verl.trainer.main_ppo` with Hydra overrides for minimal data (`n_samples=2`, `train_batch_size=4`, `total_epochs=1`) and uses `overwrite_vla_ckpt_utils.sh` for model preparation. The full veRL command includes `data.task_suite_name=libero_10`, `actor_rollout_ref.model.path=/fsx/models/sft-base/openvla-oft-libero10-traj1`, and `algorithm.adv_estimator=grpo`. See `rlinf-functional-testing` skill for complete configurations.

```bash
kubectl apply -f test-training-job.yaml
kubectl logs -f job/test-training-e2e
```

Expected: Training completes 1 epoch without errors. Validation success rate is reported.

## Level 6: Validation-Only Run

Run evaluation to ensure the full inference pipeline works:

```bash
# Same as training but with trainer.val_only=True
# This runs greedy rollout across all test scenarios and reports success rates
```

## Automated Test Suite

The repo includes `validate.sh`, a full validation harness that runs all levels:

```bash
# Local validation (no cluster required)
# Runs: terraform fmt/validate, kubeconform, shellcheck, reference integrity
./validate.sh --mode local

# Cluster validation (requires live EKS cluster)
# Runs levels L0-L6 sequentially, stops on first failure
./validate.sh --mode cluster --level 6

# Run up to a specific level
./validate.sh --mode cluster --level 3    # Stop after NCCL/EFA

# Skip earlier levels (resume from a specific level)
./validate.sh --mode cluster --skip-to 2  # Start from container test

# Test a specific example in L5
./validate.sh --mode cluster --level 5 --example maniskill-openvlaoft  # Replace with your example name

# Continue past failures
./validate.sh --mode cluster --continue-on-error

# Override image URIs via environment variables
ECR_URI=123456789.dkr.ecr.us-east-2.amazonaws.com/rlinf-on-eks/rlinf \
  ./validate.sh --mode cluster --skip-to 2 --level 3
```

### Validation Output

```
========================================
VALIDATION RESULTS
========================================
Level   Name                     Status   Time
-----   ------------------------ ------   ----
L0      Infrastructure           PASS     12s
L1      Container Build          PASS     847s
L2      Container Test           PASS     95s
L3      NCCL/EFA                 PASS     180s
L4      Model Download           PASS     320s
L5      <example-name>           PASS     540s
L5      <example-name>           PASS     510s
L5      <example-name>           PASS     620s
L6      Multi-Node               PASS     480s
========================================
9/9 passed in 3604s
```

### Implementation Details

| File | Purpose |
|------|---------|
| `validate.sh` | Main orchestrator (arg parsing, level dispatch, summary) |
| `tests/lib/common.sh` | Shared utilities (logging, timers, K8s helpers) |
| `tests/lib/local.sh` | Local mode checks (terraform, kubeconform, shellcheck) |
| `tests/lib/cluster.sh` | Cluster levels L0-L6 |
| `tests/manifests/training-step-test.yaml` | L5 single-step training Job template |
| `infrastructure/manifests/container-test.yaml` | L2 container validation (multi-venv, EFA, CUDA checks) |
| `examples/<example>/manifests/model-download.yaml` | L4 model download Jobs (per-example) |

### L5 Training Step Tests

Uses `runner.max_steps=1` (or equivalent) override to limit the framework to exactly 1 training step per example. The key insight: find the single override that limits the training loop to one step.

> **Example (RLinf):** RLinf uses `runner.max_steps=1` Hydra override to limit to exactly 1 PPO step. RLinf's `num_steps_per_epoch` is hardcoded to 1.

The training launch script (`examples/scripts/<launch-script>`) supports `HYDRA_OVERRIDES` env var for injecting additional overrides without modifying the script:

## Common Testing Pitfalls for Physical AI Stacks

These issues affect any RL training deployment, not just specific frameworks:

### 1. Interactive Prompts in Simulation Libraries

Many robotics simulation libraries (LIBERO, RoboSuite, Habitat, etc.) call `input()` at module import time to configure dataset directories. In non-interactive containers, this causes the pod to hang indefinitely with no error message.

**Fix**: Set all required environment variables (e.g., `LIBERO_DATASET_DIR`) via the pod spec `env` block AND redirect stdin in the test script: `sys.stdin = open('/dev/null')`.

### 2. Model/Config Version Skew

Model checkpoints are saved with a `config.json` that reflects the code version at save time. When the modeling code evolves and adds new config attributes (e.g., `use_proprio`, `num_action_chunks`), older checkpoints lack these attributes and `from_pretrained()` fails with `AttributeError`.

**Fix**: Apply patches that use `getattr(config, 'attr', default)` instead of direct attribute access. Store patches in `examples/patches/` for traceability.

### 3. Container Image Pull Timeouts

Physical AI RL images are 15-20 GB. First pull takes 5-10 minutes. If `activeDeadlineSeconds` is set too low, test jobs timeout during image pull -- before any code runs.

**Fix**: Set `activeDeadlineSeconds` to at least 600s (10 min) for any job using a large image. Use `imagePullPolicy: IfNotPresent` (not `Always`) to avoid re-pulling on every run. Tag images with specific versions.

### 4. YAML/Bash Script Interactions

Complex bash scripts embedded in YAML `|` block scalars fail in subtle ways. Semicolons, curly braces, and `&&`/`||` chains can be misinterpreted by the YAML parser.

**Fix**: Keep inline bash simple. For complex test logic, write a Python script and mount it via ConfigMap.

### 5. GPU Memory Not Freed Between Tests

If running multiple test levels in a single pod, GPU memory from one test may not be freed for the next. Model loading followed by FSDP wrapping can OOM.

**Fix**: Run each test level in a separate pod, or explicitly `del model; torch.cuda.empty_cache(); gc.collect()` between levels.

## Validation Checklist

- [ ] Level 0: All 8 GPUs visible, EFA devices present, FSx mounted
- [ ] Level 1: PyTorch + CUDA + all imports succeed
- [ ] Level 2: VLA model loads on a single GPU
- [ ] Level 3: FSDP wraps model, NCCL allreduce works
- [ ] Level 4: Simulation environment initializes
- [ ] Level 5: One training epoch completes without errors
- [ ] Level 6: Validation run produces success rate metrics

## Related Skills

- [Skill 09: Kubernetes Manifests](kubernetes-manifests/SKILL.md) - Manifests used in tests
- [Skill 12: Deployment Method](deployment-method/SKILL.md) - Pre-flight checks
- [Skill 14: Benchmarking](benchmarking/SKILL.md) - Performance evaluation
