---
name: networking-efa
description: Configure Elastic Fabric Adapter networking for high-performance NCCL allreduce in multi-node GPU training
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 04: Networking - Elastic Fabric Adapter (EFA)

## Purpose

Configure EFA-based networking for high-performance inter-node GPU communication via NCCL. EFA provides RDMA-like semantics that are critical for efficient allreduce operations in multi-node FSDP training.

## Why EFA Matters for Physical AI RL

Multi-node model training with FSDP requires frequent allreduce operations to synchronize gradients across GPUs. Without EFA:
- NCCL falls back to TCP sockets over standard ENI
- Bandwidth drops to ~3 GB/s (0.6% of theoretical on p5.48xlarge)
- Multi-node training throughput drops significantly

With EFA:
- NCCL uses `aws-ofi-nccl` plugin for direct EFA transport
- Sub-microsecond latency for small messages
- Up to 3200 Gbps aggregate bandwidth (p5.48xlarge)
- ~487 GB/s BusBW for large allreduce operations (2x p5.48xlarge)

## CRITICAL: EFA Installer Version Requirements

> **Lesson learned the hard way ($1,450 wasted capacity block)**: EFA installer versions 1.34.0 and 1.38.0 bundle ancient libfabric 1.22 which crashes with `cm_req_handle_error_entry` errors. These are NOT configuration issues -- they are bugs in old libfabric.

| Installer | libfabric | aws-ofi-nccl | Status |
|-----------|-----------|-------------|--------|
| 1.34.0 | 1.22.0amzn4.0 | NOT bundled | **BROKEN** -- crashes |
| 1.38.0 | 1.22.0amzn4.0 | NOT bundled | **BROKEN** -- crashes |
| 1.43.0 | 2.1.0amzn4.0 | 1.16.1 bundled | Likely works |
| **1.47.0** | **2.4.0amzn1.0** | **1.18.0 bundled** | **USE THIS** |

**Always use EFA installer 1.47.0 or newer.** It bundles everything needed:
- libfabric 2.4.0amzn1.0 (RDMA + DMA-BUF support)
- aws-ofi-nccl 1.18.0 (installed to `/opt/amazon/ofi-nccl`)
- OpenMPI 4 and 5
- No need to build aws-ofi-nccl from source.

## EFA Bandwidth by Instance Type

| Instance | EFA Interfaces | Per-Interface BW | Aggregate BW | Use Case |
|----------|---------------|------------------|--------------|----------|
| **g6.8xlarge** | 1 | 25 Gbps | 25 Gbps | **Phase 0 EFA validation ($0.98/hr)** |
| g6e.8xlarge | 1 | 25 Gbps | 25 Gbps | EFA validation ($1.86/hr) |
| p4de.24xlarge | 4 | 100 Gbps | 400 Gbps | A100 training |
| p5.48xlarge | 32 | 100 Gbps | 3200 Gbps | H100 training |
| p5e.48xlarge | 32 | 100 Gbps | 3200 Gbps | H200 training |

## Architecture

```
Node 1 (p5.48xlarge)              Node 2 (p5.48xlarge)
+-------------------+            +-------------------+
| GPU 0 ... GPU 7   |            | GPU 0 ... GPU 7   |
|   NVSwitch 900GB/s|            |   NVSwitch 900GB/s|
+--------+----------+            +--------+----------+
         |                                |
   EFA interfaces                   EFA interfaces
   (32x 100 Gbps)                  (32x 100 Gbps)
         |                                |
         +---------- EFA Fabric ----------+
              NCCL via aws-ofi-nccl
              RDMA / SRD protocol
```

## Phase 0: Validate EFA on Cheap Instances FIRST

> **ALWAYS validate EFA/NCCL on cheap instances before purchasing expensive capacity blocks.** A $1,450 capacity block was wasted because we didn't validate EFA first.

The cheapest EFA+GPU instance is `g6.8xlarge` at ~$0.98/hr (1 L4 GPU, 1 EFA interface). Run the NCCL test on 2x g6.8xlarge (~$2/hr) to validate:

1. EFA provider is selected (`NET/OFI Selected Provider is efa`)
2. NCCL allreduce completes without crashes
3. Bandwidth is reasonable for the instance type

See `infrastructure/manifests/nccl-tests-mpijob.yaml` for the full validation procedure using the MPIJob pattern.

## Step-by-Step Configuration

### 1. EKS Cluster-Level: Enable EFA on Node Group

In your Terraform configuration (see Skill 01), EFA is enabled via the EKS module's built-in support:

```hcl
# In main.tf gpu-training node group
enable_efa_support = true
# The module auto-discovers EFA interface count for the instance type,
# configures network interfaces, and creates self-referencing security
# group rules for EFA SRD traffic.
```

The EFA device plugin is installed as a separate Helm release in Terraform (see Skill 01 `helm.tf`).

### 2. Security Group Configuration

> **Discovery #35, #53**: The EKS module's `enable_efa_support = true` creates EFA interfaces but the auto-generated security group rules are insufficient. The default CIDR-based egress rule (`0.0.0.0/0`) does NOT cover EFA SRD (Scalable Reliable Datagram) packets -- the Nitro card checks SG rules at the hardware level and only matches **security-group-referenced** rules for SRD traffic. You MUST have BOTH ingress AND egress self-referencing rules. Without the egress rule, EFA memory registration (`REG_MR`) fails with `comp_status 5 err -22` and ALL EFA data transfer (including loopback) is broken.

Both rules are added in `main.tf`:

```hcl
# CRITICAL: Both ingress AND egress are required for EFA SRD traffic.
# A CIDR-based egress rule (0.0.0.0/0) is NOT sufficient -- EFA SRD
# is not IP traffic and requires security-group-referenced rules.
resource "aws_security_group_rule" "efa_all_traffic" {
  type                     = "ingress"
  from_port                = 0
  to_port                  = 0
  protocol                 = "-1"
  source_security_group_id = module.eks.node_security_group_id
  security_group_id        = module.eks.node_security_group_id
  description              = "EFA: Allow all inbound traffic between GPU nodes for NCCL/RDMA"
}

resource "aws_security_group_rule" "efa_all_traffic_egress" {
  type                     = "egress"
  from_port                = 0
  to_port                  = 0
  protocol                 = "-1"
  source_security_group_id = module.eks.node_security_group_id
  security_group_id        = module.eks.node_security_group_id
  description              = "EFA: Allow all outbound traffic between GPU nodes for NCCL/RDMA"
}
```

### 3. AMI: Use AL2023 (NOT AL2)

> **Discovery #39-40**: EKS AL2 GPU AMI kernel 5.10 does NOT support DMA-BUF (requires kernel 5.12+). Use AL2023 (`AL2023_x86_64_NVIDIA`) which runs kernel 6.1.163 with full DMA-BUF support.

```hcl
# In main.tf gpu-training node group
ami_type = "AL2023_x86_64_NVIDIA"
```

### 4. Kernel Module: efa_nv_peermem

> **Discovery #36**: The `efa_nv_peermem` module is INSTALLED but NOT LOADED on EKS GPU AMI nodes. Without it, GPU Direct RDMA fails.

Deploy a DaemonSet to load the module on each GPU node:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: load-peermem
spec:
  selector:
    matchLabels:
      app: load-peermem
  template:
    metadata:
      labels:
        app: load-peermem
    spec:
      nodeSelector:
        role: gpu-training
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      initContainers:
        - name: load-module
          image: public.ecr.aws/amazonlinux/amazonlinux:2023
          securityContext:
            privileged: true
          command:
            - nsenter
            - -t
            - "1"
            - -m
            - -u
            - -i
            - -n
            - --
            - modprobe
            - efa_nv_peermem
      containers:
        - name: pause
          image: public.ecr.aws/eks-distro/kubernetes/pause:3.9
```

### 5. NCCL Environment Variables

With EFA installer 1.47.0+, only minimal env vars are needed:

```bash
# Required
export FI_PROVIDER=efa
export FI_EFA_USE_HUGE_PAGE=0         # Required in containers without huge pages

# Optional performance tuning
export FI_EFA_FORK_SAFE=1             # Fork safety for multiprocessing
export NCCL_BUFFSIZE=8388608          # 8MB buffer
export NCCL_P2P_NET_CHUNKSIZE=524288  # 512KB chunks
export NCCL_DEBUG=WARN                # Log level (INFO for debugging)
export TORCH_NCCL_AVOID_RECORD_STREAMS=1  # Reduce CUDA memory fragmentation

# NCCL tuner plugin (auto-optimizes for AWS network topology)
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so
```

> **Discovery #44**: For libfabric>=1.18.0 and aws-ofi-nccl>=1.7.0, modern versions auto-detect most settings. The old-style env vars (`FI_EFA_USE_DEVICE_RDMA`, `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_NET_GDR_LEVEL`) are no longer needed.

### 6. Container Image: EFA Stack Installation

> **Discovery #34**: NGC PyTorch containers do NOT include aws-ofi-nccl or libfabric. They ship with HPC-X RDMA plugin which speaks IB verbs, NOT libfabric/EFA.

Install the EFA stack in the Dockerfile:

```dockerfile
# GDRCopy (GPU Direct RDMA copy library)
RUN git clone -b v2.5.1 https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

# EFA Installer 1.47.0 (bundles libfabric 2.4.0 + aws-ofi-nccl 1.18.0)
RUN curl -sL https://efa-installer.amazonaws.com/aws-efa-installer-1.47.0.tar.gz -o /tmp/efa.tar.gz \
    && tar xzf /tmp/efa.tar.gz -C /tmp \
    && cd /tmp/aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf /tmp/aws-efa-installer /tmp/efa.tar.gz

ENV LD_LIBRARY_PATH="/opt/gdrcopy/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"
```

### 7. Requesting EFA in Pod Specs

Training pods must request EFA interfaces:

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: training
      resources:
        requests:
          nvidia.com/gpu: 8
          vpc.amazonaws.com/efa: 32   # All EFA interfaces for p5.48xlarge
        limits:
          nvidia.com/gpu: 8
          vpc.amazonaws.com/efa: 32
```

**How many EFA interfaces to request:**

| Instance | Total EFA | Recommended per Pod |
|----------|-----------|-------------------|
| g6.8xlarge | 1 | 1 |
| p4de.24xlarge | 4 | 4 |
| p5.48xlarge | 32 | 32 (1 pod per node = all) |

## NCCL Test Validation (MPIJob Pattern)

> **Discovery #46**: The canonical NCCL test on EKS uses an MPIJob (Kubeflow MPI Operator), NOT a StatefulSet with torchrun. The MPIJob has a Launcher pod (runs mpirun) and Worker pods (run sshd).

See `infrastructure/manifests/nccl-tests-mpijob.yaml` for the NCCL test setup (uses pre-built public ECR image, no custom build needed).

Expected NCCL performance (2x p5.48xlarge, 16 GPUs):

| Message Size | BusBW (GB/s) |
|-------------|-------------|
| 1 MB        | ~16 GB/s    |
| 64 MB       | ~209 GB/s   |
| 1 GB        | ~437 GB/s   |
| 16 GB       | ~487 GB/s   |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NET/IB : No device found` + fallback to `NET/Socket` | Container missing aws-ofi-nccl | Install EFA installer 1.47.0 in Dockerfile |
| `cm_req_handle_error_entry: Error: 15` crash | Old libfabric (1.22.x) | Upgrade to EFA installer 1.47.0+ |
| `Caught signal 11` during NCCL warmup | Missing efa_nv_peermem module | Load module via DaemonSet |
| `NET/OFI Support for DMA-BUF registrations: false` | AL2 kernel 5.10 too old | Switch to AL2023 AMI (`AL2023_x86_64_NVIDIA`) |
| ~3 GB/s BusBW on p5.48xlarge | Falling back to TCP sockets | Verify `NET/OFI Selected Provider is efa` in NCCL_DEBUG=INFO |
| `No EFA devices found` | EFA not enabled on node group | Verify `enable_efa_support = true` in Terraform |
| `fi_info` shows no EFA | EFA device plugin not running | Check `aws-efa-k8s-device-plugin` daemonset |
| Training hangs on init | Security group blocking EFA | Verify BOTH ingress AND egress self-referencing SG rules exist (CIDR egress is NOT enough for SRD) |
| `REG_MR` fails with `comp_status 5 err -22` | Missing egress self-referencing SG rule | Add `efa_all_traffic_egress` rule (see Step 2) |

## Topology-Aware Scheduling for Multi-Node Training

### Why Network Topology Matters

AWS organizes its datacenter network in a hierarchy of layers. Each EC2 instance connects to a specific network node at the bottom layer, which chains upward through 2-3 layers of switches. Two instances sharing the same bottom-layer network node have the lowest latency and highest bandwidth between them.

### EC2 Instance Topology API

```bash
aws ec2 describe-instance-topology \
  --filters "Name=instance-type,Values=p5.48xlarge" \
  --region us-east-1
```

Labels applied:
- `topology.k8s.aws/network-node-layer-1` -- top-level network node
- `topology.k8s.aws/network-node-layer-2` -- mid-level network node
- `topology.k8s.aws/network-node-layer-3` -- bottom-level network node

See the topology labeler DaemonSet in `manifests/topology-labeler.yaml`.

## Related Skills

- [Skill 01: EKS Cluster Provisioning](eks-cluster-provisioning/SKILL.md)
- [Skill 03: AMI and Node Configuration](ami-and-node-configuration/SKILL.md)
- [Skill 09: Kubernetes Manifests](kubernetes-manifests/SKILL.md)
- [Skill 13: Functional Testing](functional-testing/SKILL.md)
