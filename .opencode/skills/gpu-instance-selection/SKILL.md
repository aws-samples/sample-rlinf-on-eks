---
name: gpu-instance-selection
description: Select optimal EC2 GPU instance types for Physical AI RL training based on VRAM, interconnect, and cost requirements
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 02: GPU Instance Selection

## Purpose

Select the optimal EC2 GPU instance type for Physical AI RL training workloads on EKS. The choice depends on model size, VRAM requirements, inter-node communication needs, budget, and availability.

## Decision Framework

### Key Requirements for Physical AI RL

| Requirement | Why |
|-------------|-----|
| **80GB+ VRAM per GPU** | 7B VLA models with FSDP full-shard + parallel env rendering |
| **8 GPUs per node** | Standard configuration for distributed training |
| **High-bandwidth interconnect** | NVLink/NVSwitch intra-node, EFA inter-node |
| **High CPU core count** | Parallel simulation environment rendering |
| **Large system RAM** | Model weights + optimizer states + env buffers |
| **EFA support** | NCCL allreduce across nodes |

### Instance Comparison Matrix

| Instance | GPU | VRAM | GPUs | GPU Interconnect | EFA BW | vCPUs | RAM | Local Storage | On-Demand $/hr (us-east-1) |
|----------|-----|------|------|------------------|--------|-------|-----|---------------|---------------------------|
| **p5.48xlarge** | H100 | 80GB | 8 | NVSwitch 900 GB/s | 3200 Gbps | 192 | 2048 GB | 8x 3.84TB NVMe | ~$98.32 |
| **p5e.48xlarge** | H200 | 141GB | 8 | NVSwitch 900 GB/s | 3200 Gbps | 192 | 2048 GB | 8x 3.84TB NVMe | ~$115.00 |
| **p4de.24xlarge** | A100 | 80GB | 8 | NVSwitch 600 GB/s | 400 Gbps | 96 | 1152 GB | 8x 1TB NVMe | ~$40.97 |
| **p4d.24xlarge** | A100 | 40GB | 8 | NVSwitch 600 GB/s | 400 Gbps | 96 | 1152 GB | 8x 1TB NVMe | ~$32.77 |

**Validation-tier instances** (infrastructure testing, NOT for VLA training):

| Instance | GPU | VRAM | GPUs | EFA | vCPUs | RAM | On-Demand $/hr |
|----------|-----|------|------|-----|-------|-----|----------------|
| **g6.8xlarge** | L4 | 24GB | 1 | 1x 100 Gbps | 32 | 128 GB | ~$0.98 |
| **g5.48xlarge** | A10G | 24GB | 8 | 1x 100 Gbps | 192 | 768 GB | ~$16.29 |

> **CRITICAL**: g6.8xlarge (1x L4 24GB) **CANNOT** run VLA RL training for 7.5B+ models. FSDP on a single GPU cannot shard -- the full model must fit in VRAM during initialization. Even with all offloading enabled (param, grad, optimizer offload + gradient checkpointing), the L4 OOMs during `flatten_tensors_into_flat_param`. Use g6 for infrastructure validation (EFA, NCCL, container tests) only.

> **Note**: Pricing is approximate and changes. Always verify current pricing at https://aws.amazon.com/ec2/pricing/on-demand/

### Decision Tree

```
Is VRAM > 40GB per GPU required?
├── YES (7B+ VLA models with FSDP, parallel env rendering)
│   ├── Is maximum training throughput critical?
│   │   ├── YES → p5.48xlarge (H100) or p5e.48xlarge (H200)
│   │   │         - 3200 Gbps EFA for fast allreduce
│   │   │         - NVSwitch 900 GB/s intra-node
│   │   │         - 192 vCPUs for parallel env rendering
│   │   └── NO → p4de.24xlarge (A100 80GB)
│   │             - Best cost/performance for 80GB VRAM
│   │             - 400 Gbps EFA (sufficient for 2-node training)
│   │             - Closest to A800 used in the original paper
│   └── Is budget the primary constraint?
│       └── YES → p4de.24xlarge (A100 80GB)
│                 ~60% cheaper than p5.48xlarge
└── NO (smaller models, experimenting)
    └── p4d.24xlarge (A100 40GB)
        - Lowest cost with EFA
        - May require aggressive memory optimization
```

### VRAM Budget Estimation

For a 7B VLA model with FSDP:

| Component | Per-GPU VRAM (FSDP full-shard) |
|-----------|-------------------------------|
| Model parameters (bf16) | ~1.75 GB (14GB / 8 GPUs) |
| Gradients (fp32, with CPU offload) | ~0 GB (offloaded) |
| Optimizer states (with CPU offload) | ~0 GB (offloaded) |
| Activations (no gradient checkpointing) | ~15-25 GB |
| Simulation env rendering buffers | ~5-10 GB |
| KV cache, attention buffers | ~5-10 GB |
| PyTorch CUDA memory overhead | ~2-5 GB |
| **Total estimated** | **~30-55 GB** |

> **Result**: 40GB GPUs are marginal. 80GB GPUs provide comfortable headroom for batch size scaling and parallel environments.

## 7B+ VLA Model GPU Sizing

### Recommended Configuration

| Scenario | Instance | Nodes | Total GPUs | Estimated Cost/hr |
|----------|----------|-------|------------|-------------------|
| **Development / single-task RL** | p4de.24xlarge | 1 | 8 | ~$41 |
| **Production / multi-task RL** | p4de.24xlarge | 2 | 16 | ~$82 |
| **Maximum throughput** | p5.48xlarge | 2 | 16 | ~$197 |

> **RLinf Example:** See `examples/AGENTS.md` for RLinf-specific GPU sizing requirements.

### Why p4de.24xlarge is the Default Recommendation

1. **Hardware match**: Many Physical AI RL frameworks were developed on NVIDIA A800 80GB, which is architecturally identical to A100 80GB (A800 is the China-export variant)
2. **Cost efficiency**: ~60% cheaper than p5.48xlarge with the same 80GB VRAM
3. **EFA bandwidth**: 400 Gbps is sufficient for 2-node FSDP training of 7B models
4. **Availability**: Generally easier to obtain capacity than p5 instances

### When to Choose p5.48xlarge

- Training on 4+ nodes where EFA bandwidth becomes the bottleneck (3200 Gbps vs 400 Gbps)
- Faster time-to-result justifies the 2.4x cost premium
- Need higher CPU core count (192 vs 96) for parallel environment rendering
- Need more system RAM (2048 GB vs 1152 GB)

## Capacity Planning

### Request Quotas Early

GPU instances have strict service quotas. Request increases before you need them:

```bash
# Check current quota for p4de instances
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-417A185B \
  --region us-east-1

# Request quota increase (example: 96 vCPUs = 1x p4de.24xlarge)
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-417A185B \
  --desired-value 192 \
  --region us-east-1
```

Common quota codes:
- `L-417A185B`: Running On-Demand P instances (p4d, p4de)
- `L-5480EFD2`: Running On-Demand P5 instances

### Capacity Reservations

For predictable workloads, use Capacity Reservations to guarantee instance availability:

```bash
aws ec2 create-capacity-reservation \
  --instance-type p4de.24xlarge \
  --instance-platform Linux/UNIX \
  --availability-zone us-east-1a \
  --instance-count 2 \
  --instance-match-criteria targeted
```

## Spot vs On-Demand

| Factor | On-Demand | Spot |
|--------|-----------|------|
| **RL Training Jobs** | Recommended | Not recommended |
| **Reason** | RL training is stateful, long-running, and sensitive to interruptions | 2-minute warning is insufficient to checkpoint a large model |
| **Exception** | - | Short evaluation/benchmarking runs with frequent checkpoints |

## Related Skills

- [Skill 01: EKS Cluster Provisioning](eks-cluster-provisioning/SKILL.md)
- [Skill 03: AMI and Node Configuration](ami-and-node-configuration/SKILL.md)
