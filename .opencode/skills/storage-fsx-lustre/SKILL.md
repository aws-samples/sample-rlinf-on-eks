---
name: storage-fsx-lustre
description: Deploy Amazon FSx for Lustre as shared high-performance storage for datasets, checkpoints, and model weights
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 05: Storage - FSx for Lustre

## Purpose

Deploy Amazon FSx for Lustre as the shared high-performance filesystem for Physical AI RL training. FSx for Lustre provides POSIX-compatible storage needed for:
- **Datasets**: Demonstration trajectories for RL simulation benchmarks
- **Checkpoints**: FSDP full state dict saves (multi-GB per checkpoint)
- **SFT model weights**: Pre-trained model loaded by all nodes simultaneously
- **Logs and rollout videos**: Training artifacts

## Why FSx for Lustre

| Feature | FSx for Lustre | EFS | S3 (direct) |
|---------|---------------|-----|-------------|
| POSIX compatible | Yes | Yes | No |
| Throughput | Up to 1000+ MB/s per TiB | Limited | High (single stream) |
| Latency | Sub-ms | Low ms | 10s of ms |
| Concurrent multi-node access | Excellent | Good | Via API only |
| Checkpoint save (FSDP gather) | Fast (direct write) | Slower | Requires staging |
| S3 integration | Native (data repository) | No | Native |
| EFA support | Yes (Lustre over EFA) | No | No |
| Cost (1.2 TiB PERSISTENT_2) | ~$0.145/GB/month | ~$0.30/GB/month | ~$0.023/GB/month |

**FSx for Lustre is the recommended choice** for active training due to its throughput and POSIX compatibility. S3 should be used as the durable backing store (see Skill 05b for S3 Connector for PyTorch as an alternative).

## Architecture

```
EKS GPU Nodes                FSx for Lustre            Amazon S3
+------------+              +----------------+         +----------+
| Pod        |  Lustre      | /fsx/          |  Auto   | s3://    |
| mount:     | ──────────── | ├── datasets/  | ─────── | datasets/|
| /fsx       |  client      | ├── models/    |  import | models/  |
|            |              | ├── ckpts/     |         |          |
+------------+              | └── logs/      |         +----------+
                            +----------------+
                            PERSISTENT_2
                            125 MB/s/TiB
```

## Step-by-Step

### 1. Create IAM Role for FSx CSI Driver

The IAM role for the FSx CSI driver is created in Terraform (see Skill 01 `iam.tf`). The EKS managed add-on for `aws-fsx-csi-driver` is configured with this role in `main.tf`.

If configuring manually outside Terraform:

```bash
eksctl create iamserviceaccount \
  --name fsx-csi-controller-sa \
  --namespace kube-system \
  --cluster <cluster-name> \
  --role-name AmazonEKS_FSx_CSI_DriverRole \
  --role-only \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonFSxFullAccess \
  --approve
```

### 2. Install FSx CSI Driver

The FSx CSI driver is installed as an EKS managed add-on in Terraform:

```hcl
# In main.tf cluster_addons block
aws-fsx-csi-driver = {
  most_recent              = true
  service_account_role_arn = aws_iam_role.fsx_csi.arn
}
```

If installing manually outside Terraform:

```bash
aws eks create-addon \
  --addon-name aws-fsx-csi-driver \
  --cluster-name <cluster-name> \
  --resolve-conflicts OVERWRITE
```

Verify:

```bash
kubectl get pods -n kube-system -l app=fsx-csi-controller
```

### 3. Create FSx Security Group

The FSx security group is defined in Terraform (see Skill 01). If creating manually:

```bash
# Get cluster VPC and security group
CLUSTER_SG=$(aws eks describe-cluster --name <cluster-name> \
  --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)

VPC_ID=$(aws eks describe-cluster --name <cluster-name> \
  --query "cluster.resourcesVpcConfig.vpcId" --output text)

# Create FSx security group
FSX_SG=$(aws ec2 create-security-group \
  --group-name <cluster-name>-fsx-sg \
  --description "Security group for FSx for Lustre" \
  --vpc-id $VPC_ID \
  --query "GroupId" --output text)

# Allow Lustre traffic from cluster nodes
aws ec2 authorize-security-group-ingress \
  --group-id $FSX_SG \
  --protocol tcp \
  --port 988 \
  --source-group $CLUSTER_SG

# Allow Lustre internode traffic
aws ec2 authorize-security-group-ingress \
  --group-id $FSX_SG \
  --protocol tcp \
  --port 1018-1023 \
  --source-group $CLUSTER_SG

# CRITICAL: Self-referencing rules for FSx inter-OST communication
# Without these, FSx creation may succeed but mounts will hang or fail.
aws ec2 authorize-security-group-ingress \
  --group-id $FSX_SG \
  --protocol tcp \
  --port 988 \
  --source-group $FSX_SG

aws ec2 authorize-security-group-ingress \
  --group-id $FSX_SG \
  --protocol tcp \
  --port 1018-1023 \
  --source-group $FSX_SG
```

### 4. Create FSx Filesystem

#### Option A: Dynamic Provisioning (via StorageClass)

```yaml
# fsx-storageclass.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fsx-lustre
provisioner: fsx.csi.aws.com
parameters:
  subnetId: subnet-0abc123def456789    # MUST be in the same AZ as GPU and system nodes (local.target_subnet_ids)
  securityGroupIds: sg-0abc123def456789 # FSx security group created above
  deploymentType: PERSISTENT_2
  perUnitStorageThroughput: "125"        # 125/250/500/1000 MB/s per TiB
  dataCompressionType: "NONE"            # Disable for maximum throughput
  fileSystemTypeVersion: "2.15"
  automaticBackupRetentionDays: "0"      # Disable backups (use S3 for durability)
```

```bash
kubectl apply -f fsx-storageclass.yaml
```

#### Option B: Static Provisioning (pre-existing filesystem)

Create filesystem via CLI:

```bash
# Get subnet ID for the GPU node AZ
SUBNET_ID=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=availability-zone,Values=us-east-1a" \
  --query "Subnets[?MapPublicIpOnLaunch==\`false\`].SubnetId | [0]" --output text)

# Create 4.8 TiB PERSISTENT_2 filesystem
aws fsx create-file-system \
  --file-system-type LUSTRE \
  --storage-capacity 4800 \
  --storage-type SSD \
  --subnet-ids $SUBNET_ID \
  --security-group-ids $FSX_SG \
  --lustre-configuration '{
    "DeploymentType": "PERSISTENT_2",
    "PerUnitStorageThroughput": 125,
    "DataCompressionType": "NONE"
  }' \
  --tags "Key=Project,Value=<cluster-name>"
```

Then create PV/PVC:

```yaml
# fsx-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: fsx-pv
spec:
  capacity:
    storage: 4800Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteMany
  mountOptions:
    - flock
  persistentVolumeReclaimPolicy: Retain
  csi:
    driver: fsx.csi.aws.com
    volumeHandle: fs-0123456789abcdef0
    volumeAttributes:
      dnsname: fs-0123456789abcdef0.fsx.us-east-1.amazonaws.com
      mountname: abcdef01
---
# fsx-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: fsx-claim
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ""
  resources:
    requests:
      storage: 4800Gi
  volumeName: fsx-pv
```

### 5. Directory Structure on FSx

Create the standard directory layout:

```bash
# From a pod with FSx mounted
mkdir -p /fsx/datasets/<dataset-a>
mkdir -p /fsx/datasets/<dataset-b>
mkdir -p /fsx/models/sft-base
mkdir -p /fsx/checkpoints
mkdir -p /fsx/logs/mlflow
```

### 6. S3 Data Repository Association (Recommended)

> **Design principle**: Treat FSx as ephemeral, S3 as durable. A single DRA links the entire FSx filesystem (`/`) to an S3 bucket, so all writes are automatically backed to S3. The filesystem can be destroyed and recreated without data loss.

A single Data Repository Association is managed in Terraform (`storage.tf`):

| FSx Path | S3 Path | Auto-Import | Auto-Export |
|----------|---------|-------------|-------------|
| `/` (entire filesystem) | `s3://${cluster}-fsx-data/` | NEW, CHANGED, DELETED | NEW, CHANGED, DELETED |

Using a single root-level DRA (rather than per-directory DRAs) is simpler and ensures all data is backed regardless of directory structure. Training code can create any directories it needs (`/datasets`, `/checkpoints`, `/models`, `/logs`, etc.) and everything is automatically synced.

**Workflow:**
1. Pre-stage data to S3: `aws s3 cp model.tar.gz s3://<bucket-name>/models/`
2. When FSx is created, DRA auto-imports metadata (files appear as lazy-loaded stubs)
3. First read of a file triggers actual data transfer from S3 to FSx
4. Writes to FSx are auto-exported back to S3
5. If FSx is destroyed and recreated, S3 data is re-imported automatically

If configuring manually outside Terraform:

```bash
aws fsx create-data-repository-association \
  --file-system-id fs-0123456789abcdef0 \
  --file-system-path / \
  --data-repository-path s3://<bucket-name>/ \
  --s3 '{
    "AutoImportPolicy":{"Events":["NEW","CHANGED","DELETED"]},
    "AutoExportPolicy":{"Events":["NEW","CHANGED","DELETED"]}
  }' \
  --batch-import-meta-data-on-create
```

## Storage Sizing Guide

| Component | Estimated Size | Notes |
|-----------|---------------|-------|
| SFT model (7B VLA, bf16) | ~14 GB | bf16 weights |
| Training datasets | ~5-25 GB | Varies by benchmark |
| Additional datasets | ~20 GB | Varies by task count |
| Single checkpoint | ~14-28 GB | Full FSDP state dict |
| 100 checkpoints | ~1.4-2.8 TB | Save every 20-25 steps |
| Training logs + videos | ~50-200 GB | Rollout videos are large |
| **Total recommended** | **4.8 TiB** | Minimum PERSISTENT_2 size |

> **Reference-specific details:** See `examples/AGENTS.md` for exact dataset sizes and model weight paths.

## Performance Tuning

### Throughput Selection

| Throughput Tier | MB/s per TiB | Use Case | Cost Multiplier |
|-----------------|-------------|----------|-----------------|
| 125 | 125 | Dataset reads, checkpoint writes | 1x |
| 250 | 250 | Frequent large checkpoints | ~1.5x |
| 500 | 500 | Very large models, many parallel readers | ~2.5x |
| 1000 | 1000 | Maximum I/O performance | ~5x |

**Recommendation**: Start with 125 MB/s/TiB. A 4.8 TiB filesystem gets 600 MB/s aggregate throughput, which is sufficient for most VLA training.

### Client-Side Tuning (applied in Skill 03 bootstrap)

```bash
# Applied via lctl on each node
lctl set_param ldlm.namespaces.*.lru_max_age=600000
lctl set_param ldlm.namespaces.*.lru_size=$((100 * $(nproc)))
lctl set_param llite.*.max_cached_mb=64
lctl set_param osc.*OST*.max_rpcs_in_flight=32
lctl set_param mdc.*.max_rpcs_in_flight=64
lctl set_param mdc.*.max_mod_rpcs_in_flight=50
```

## Using FSx in Training Pods

```yaml
# In your training Job/Pod spec
spec:
  containers:
    - name: training
      volumeMounts:
        - name: fsx-storage
          mountPath: /fsx
  volumes:
    - name: fsx-storage
      persistentVolumeClaim:
        claimName: fsx-claim
```

Then in training scripts, all paths reference `/fsx`:

```bash
SFT_MODEL_PATH="/fsx/models/sft-base/openvla-7b"  # Replace with your model path
CKPT_PATH="/fsx/checkpoints"
```

## Validation Checklist

- [ ] FSx filesystem is in `AVAILABLE` state
- [ ] PV and PVC are bound (`kubectl get pvc fsx-claim`)
- [ ] Test pod can read/write to `/fsx`
- [ ] Multi-node write test: two pods on different nodes can both write and see each other's files
- [ ] Throughput test: `dd if=/dev/zero of=/fsx/test bs=1M count=1024` shows expected bandwidth

## Related Skills

- [Skill 03: AMI and Node Configuration](ami-and-node-configuration/SKILL.md) - Lustre client installation
- [Skill 05b: Storage - S3 Connector for PyTorch](storage-s3-connector-pytorch/SKILL.md) - Alternative storage approach
- [Skill 05c: Storage - Mountpoint for S3](storage-mountpoint-s3/SKILL.md) - FUSE-based S3 mount for read-heavy workloads
- [Skill 05d: Storage - Amazon S3 Files](storage-s3-files/SKILL.md) - NFS file system access to S3 (dev/debug, dataset prep)
- [Skill 06: Dataset Preparation](dataset-preparation/SKILL.md) - Staging data on FSx
