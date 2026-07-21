---
name: ami-and-node-configuration
description: Configure AMI selection and GPU node bootstrap including NVIDIA drivers, EFA, and Lustre client
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 03: AMI and Node Configuration

## Purpose

Configure the AMI (Amazon Machine Image) and node bootstrap for GPU training nodes. The nodes must have NVIDIA drivers, CUDA toolkit, EFA drivers, Lustre client, and the correct kernel modules loaded before pods can schedule.

## AMI Selection

### Recommended: Amazon Linux 2023 EKS GPU AMI

> **Discovery #39-40**: AL2 kernel 5.10 does NOT support DMA-BUF (requires kernel 5.12+). AL2023 runs kernel 6.1.163 with full DMA-BUF support required for EFA RDMA. **Always use AL2023 for GPU training nodes.**

Use the **Amazon Linux 2023 NVIDIA GPU AMI** (`ami_type: AL2023_x86_64_NVIDIA` in Terraform):

```hcl
# In main.tf gpu-training node group
ami_type = "AL2023_x86_64_NVIDIA"
```

The AL2023 GPU AMI comes pre-installed with NVIDIA drivers and CUDA toolkit. Key advantages over AL2:
- Kernel 6.1.163 with DMA-BUF support for GPU Direct RDMA
- `efa_nv_peermem` kernel module available (though not auto-loaded -- see below)
- Better security baseline and package management via `dnf`

### DO NOT USE: Amazon Linux 2 EKS GPU AMI

The older `AL2_x86_64_GPU` AMI runs kernel 5.10 which:
- Does NOT support DMA-BUF registrations (required for efficient EFA RDMA)
- NCCL logs show `NET/OFI Support for DMA-BUF registrations: false`
- Can still work with TCP socket fallback (~3 GB/s) but defeats the purpose of EFA

### Alternative: Ubuntu 22.04

Ubuntu 22.04 EKS-optimized AMI can work (set `ami_type = "CUSTOM"` with a `data "aws_ami"` lookup) but requires additional effort for driver and Lustre client installation. The bootstrap script (`infrastructure/cluster/scripts/gpu-node-bootstrap.sh`) is OS-agnostic and supports both AL2 and Ubuntu.

## Required Software Stack on Nodes

| Component | Version | Purpose |
|-----------|---------|---------|
| NVIDIA Driver | 535+ (or 550+) | GPU compute |
| CUDA Toolkit | 12.4 | PyTorch CUDA backend |
| EFA Driver | 2.12.1+ | RDMA networking for NCCL |
| Lustre Client | 2.15+ | FSx for Lustre mount |
| NCCL | 2.21+ | GPU collective communications |
| `aws-ofi-nccl` | Latest | NCCL transport plugin for EFA |

## Node Bootstrap Script

This is the full node bootstrap script. This script is injected via `cloudinit_pre_nodeadm` as a `text/x-shellscript` MIME part in the AL2023 managed node group user data (see Skill 01 `main.tf`). **IMPORTANT**: AL2023 does NOT use `pre_bootstrap_user_data` -- that field is silently ignored. Only `cloudinit_pre_nodeadm` works. A second MIME part configures the `FastImagePull` feature gate for SOCI parallel pull. The script is OS-agnostic and supports both Amazon Linux 2023 and Ubuntu.

```bash
#!/bin/bash
# =============================================================================
# GPU Node Bootstrap Script (cloudinit_pre_nodeadm)
#
# Injected via cloudinit_pre_nodeadm as a text/x-shellscript part in the
# MIME multi-part user data for AL2023 EKS managed node groups.
#
# IMPORTANT: AL2023 does NOT use pre_bootstrap_user_data -- that field is
# silently ignored. Only cloudinit_pre_nodeadm works for AL2023 AMI types.
# See infrastructure/cluster/main.tf for how this script is referenced.
#
# This script installs and configures:
#   1. EFA driver (if version is below minimum)
#   2. Lustre client for FSx
#   3. Kernel modules (LNet, EFA, Lustre)
#   4. Network tuning for Lustre over EFA
#
# Note: GDRCopy userspace library is installed in the container Dockerfile.
# The GDRCopy kernel module (gdrdrv) cannot be built on EKS GPU AMIs due to
# missing NVIDIA kernel driver source headers (Discovery #37). NCCL works
# without it via host-memory staging.
#
# Note: Topology labeling is handled by a DaemonSet after the node joins
# the cluster (see manifests/topology-labeler.yaml).
#
# Note: SOCI parallel pull (FastImagePull) is configured via a separate
# NodeConfig part in cloudinit_pre_nodeadm, not in this script.
# =============================================================================
set -euo pipefail
exec 1> >(logger -s -t gpu-node-bootstrap) 2>&1

echo "=== Starting GPU node pre-bootstrap ==="

# Detect OS early -- needed by Lustre installation section
if [ -f /etc/os-release ]; then
    . /etc/os-release
fi

# -----------------------------------------------------------------------------
# 1. EFA Driver Installation
# -----------------------------------------------------------------------------
efa_version=$(modinfo efa 2>/dev/null | awk '/^version:/ {print $2}' | sed 's/[^0-9.]//g' || echo "0.0.0")
min_efa_version="2.12.1"

if [[ "$(printf '%s\n' "$min_efa_version" "$efa_version" | sort -V | head -n1)" != "$min_efa_version" ]]; then
    echo "Installing EFA driver (current: $efa_version, minimum: $min_efa_version)..."
    curl -sO https://efa-installer.amazonaws.com/aws-efa-installer-1.47.0.tar.gz
    tar -xf aws-efa-installer-1.47.0.tar.gz && cd aws-efa-installer
    yum install -y pciutils environment-modules libnl3-devel dkms 2>/dev/null || \
      apt-get update && apt-get install -y pciutils environment-modules libnl-3-dev libnl-route-3-200 libnl-route-3-dev dkms
    ./efa_installer.sh -y
    cd ..
    rm -rf aws-efa-installer aws-efa-installer-1.47.0.tar.gz
    modinfo efa
else
    echo "EFA driver version $efa_version meets minimum $min_efa_version"
fi

# -----------------------------------------------------------------------------
# 2. Lustre Client Installation
# -----------------------------------------------------------------------------
echo "Installing Lustre client..."

if [[ "${ID:-}" == "amzn" ]]; then
    # Amazon Linux 2 / AL2023
    amazon-linux-extras install -y lustre 2>/dev/null || \
      yum install -y lustre-client 2>/dev/null || \
      dnf install -y lustre-client 2>/dev/null || true
elif [[ "${ID:-}" == "ubuntu" ]]; then
    wget -qO - https://fsx-lustre-client-repo-public-keys.s3.amazonaws.com/fsx-ubuntu-public-key.asc \
      | gpg --dearmor | tee /usr/share/keyrings/fsx-ubuntu-public-key.gpg > /dev/null
    echo "deb [signed-by=/usr/share/keyrings/fsx-ubuntu-public-key.gpg] https://fsx-lustre-client-repo.s3.amazonaws.com/ubuntu jammy main" \
      | tee /etc/apt/sources.list.d/fsxlustreclientrepo.list
    apt-get update
    apt-get install -y lustre-client-modules-$(uname -r) lustre-client
fi

# Verify
modinfo lustre 2>/dev/null && echo "Lustre client installed" || echo "WARNING: Lustre client not available"

# -----------------------------------------------------------------------------
# 3. Kernel Module Loading & LNet Configuration
# -----------------------------------------------------------------------------
echo "Configuring LNet for Lustre over EFA..."

# Get primary network interface
eth_intf="$(/sbin/ip -br -4 a sh | grep "$(hostname -i)/" | awk '{print $1}' || echo "eth0")"

modprobe lnet 2>/dev/null || true
modprobe ksocklnd 2>/dev/null || true

# Try loading kefalnd (EFA LNet driver) -- may not be available on all AMIs
modprobe kefalnd ipif_name="$eth_intf" 2>/dev/null || true

if command -v lnetctl &>/dev/null; then
    lnetctl lnet configure 2>/dev/null || true

    # TCP interface
    lnetctl net del --net tcp 2>/dev/null || true
    lnetctl net add --net tcp --if "$eth_intf" 2>/dev/null || true

    # EFA interfaces (if available)
    num_efa_devices="$(ls -1 /sys/class/infiniband 2>/dev/null | wc -l)"
    if [[ $num_efa_devices -gt 0 ]]; then
        instance_type="$(TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds:60'); curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type)"

        if [[ "$instance_type" == p5.* || "$instance_type" == p5e.* ]]; then
            for intf in $(ls -1 /sys/class/infiniband | awk 'NR % 4 == 1'); do
                lnetctl net add --net efa --if "$intf" --peer-credits 32 2>/dev/null || true
            done
        else
            lnetctl net add --net efa --if "$(ls -1 /sys/class/infiniband | head -n1)" --peer-credits 32 2>/dev/null || true
            if [[ $num_efa_devices -gt 1 ]]; then
                lnetctl net add --net efa --if "$(ls -1 /sys/class/infiniband | tail -n1)" --peer-credits 32 2>/dev/null || true
            fi
        fi

        lnetctl set discovery 1 2>/dev/null || true
        lnetctl udsp add --src efa --priority 0 2>/dev/null || true
        echo "EFA interfaces configured: $(lnetctl net show 2>/dev/null | grep -c '@efa' || echo 0)"
    fi

    modprobe lustre 2>/dev/null || true
fi

# -----------------------------------------------------------------------------
# 4. Network Tuning
# -----------------------------------------------------------------------------
echo "Applying network tuning..."
mkdir -p /etc/modprobe.d
grep -q "ptlrpc ptlrpcd_per_cpt_max" /etc/modprobe.d/lustre.conf 2>/dev/null || \
    echo "options ptlrpc ptlrpcd_per_cpt_max=64" >> /etc/modprobe.d/lustre.conf
grep -q "ksocklnd credits" /etc/modprobe.d/lustre.conf 2>/dev/null || \
    echo "options ksocklnd credits=2560" >> /etc/modprobe.d/lustre.conf

echo "=== GPU node pre-bootstrap complete ==="
```

## SOCI Parallel Pull (FastImagePull)

> **Discovery #27, #29, #30**: SOCI snapshotter is pre-installed in the AL2023 EKS NVIDIA GPU AMI. The `FastImagePull` feature gate enables parallel pull/unpack mode, which parallelizes both layer download (HTTP range requests) and layer unpacking. No SOCI index is needed -- it works with any standard OCI image.

SOCI parallel pull is configured via a separate `application/node.eks.aws` part in `cloudinit_pre_nodeadm`:

```yaml
---
apiVersion: node.eks.aws/v1alpha1
kind: NodeConfig
spec:
  featureGates:
    FastImagePull: true
```

Requirements:
- Instance size 2xlarge or larger
- High EBS throughput (600+ MiB/s) for best results
- No build pipeline changes needed

On g6.8xlarge (17GB training image), validated that soci-snapshotter.service is active, containerd uses `snapshotter = "soci"`, and logs show `"preparing snapshot with parallel pull/unpack"`.

## NVIDIA Driver Verification

Once nodes are running, verify GPU access:

```bash
# SSH or exec into a pod on the GPU node
nvidia-smi

# Expected output should show:
# - 8 GPUs (H100 or A100 depending on instance type)
# - CUDA version 12.4+
# - Driver version 535+ or 550+
```

## Verification Checklist

- [ ] `nvidia-smi` shows all 8 GPUs on each training node
- [ ] `modinfo efa` returns version >= 2.12.1
- [ ] `modinfo lustre` returns successfully
- [ ] `lnetctl net show` shows EFA interfaces configured
- [ ] `/sys/class/infiniband/` lists expected number of EFA devices
- [ ] `fi_info -p efa` shows EFA provider (requires `libfabric` utils)

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `nvidia-smi` not found | NVIDIA driver not installed on AMI | Use DLAMI or install driver in preBootstrapCommands |
| `modinfo efa` fails | EFA driver version too old | Upgrade via `efa_installer.sh` |
| `lustre-client-modules` package not found | Kernel version mismatch | Ensure `lustre-client-modules-$(uname -r)` matches the running kernel |
| No devices in `/sys/class/infiniband/` | Instance type does not support EFA or EFA not enabled | Verify `enable_efa_support = true` in EKS managed node group |
| `lnetctl` commands fail | Lustre kernel modules not loaded | Run `modprobe lnet && modprobe lustre` |

## Related Skills

- [Skill 01: EKS Cluster Provisioning](eks-cluster-provisioning/SKILL.md)
- [Skill 02: GPU Instance Selection](gpu-instance-selection/SKILL.md)
- [Skill 04: Networking - EFA](networking-efa/SKILL.md)
