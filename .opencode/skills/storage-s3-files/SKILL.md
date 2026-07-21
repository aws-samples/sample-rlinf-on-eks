---
name: storage-s3-files
description: Mount S3 buckets as native NFS file systems on EKS via Amazon S3 Files for shared, low-latency file access
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 05d: Storage - Amazon S3 Files

## Purpose

Use Amazon S3 Files to mount S3 buckets as native NFS file systems on EKS. S3 Files (launched April 2026) is a shared file system built on Amazon EFS that provides low-latency file access to data stored in S3. It is the first cloud object store to offer fully-featured NFS v4.1+ file system access directly to bucket data.

S3 Files is suitable for:
- **Development and debugging** workloads that need file-system access to S3 data
- **Dataset browsing and preparation** before training
- **Inference serving** (read-heavy, large sequential reads streamed from S3)
- **Cost-sensitive persistent environments** where FSx for Lustre's minimum 1200 GiB is overkill
- **Multi-compute shared workspaces** (up to 25,000 concurrent connections)

## When to Use S3 Files vs Other Storage Options

| Factor | S3 Files | FSx for Lustre | Mountpoint for S3 |
|--------|----------|----------------|-------------------|
| **Protocol** | NFS v4.1+ | Lustre (POSIX) | FUSE |
| **Consistency** | NFS close-to-open | Full POSIX | Eventual |
| **Read latency (small files)** | ~1ms (EFS-backed cache) | Sub-ms | S3 API latency (10s ms) |
| **Read throughput (large files)** | TB/s aggregate (from S3) | Hundreds MB/s per TiB | 10-100 Gbps |
| **Write support** | Full (NFS) | Full (POSIX) | New files only |
| **Write sync to S3** | ~60s automatic | DRA auto-export | Immediate (direct S3 PUT) |
| **Provisioned capacity** | None (pay active working set) | Fixed (min 1200 GiB) | None (S3 only) |
| **File locking** | Advisory (NFS) | POSIX mandatory | Not supported |
| **Rename/move** | Non-atomic on S3 backend | Atomic | Not supported across dirs |
| **Max per-client read** | 3 GiB/s | Instance-limited | Instance-limited |
| **`nconnect` mount** | Not supported | N/A (Lustre native) | N/A |
| **Cost model** | Storage ($0.30/GB/month active) + access | Fixed capacity ($0.145/GB/month) | S3 only ($0.023/GB/month) |
| **Best for** | Dev/debug, dataset prep, inference | Active RL training | Read-heavy model loading |

### Why FSx for Lustre Remains Recommended for Training

For active Physical AI RL training (RL framework with GRPO), FSx for Lustre is the better choice because:

1. **POSIX consistency**: RL framework workers on multiple nodes write checkpoints and shared state. FSx provides full POSIX consistency; S3 Files provides NFS close-to-open consistency (writes not visible to other clients until file is closed and reopened). This can cause subtle issues with frameworks that expect immediate cross-node visibility.

2. **Checkpoint I/O patterns**: Training checkpoints involve rapid overwrites and renames. FSx handles these atomically. S3 Files translates renames into copy+delete operations on S3 (non-atomic, with synchronization delay).

3. **Predictable latency**: FSx provides sub-ms latency for all operations. S3 Files has first-access latency (metadata import on first directory listing) that can add seconds to job startup.

4. **Cost context**: FSx at $174/month for 1200 GiB is negligible compared to GPU costs ($2/hr per g6, $36/hr per p5). Optimizing storage cost is not the priority during active training.

## Architecture

```
EKS Training Pod              S3 File System                Amazon S3
+------------------+         +--------------------+        +------------------+
| /s3files/        |  NFS    | High-perf storage  |        | s3://bucket/     |
| +-- datasets/    | <-----> | (EFS-backed, ~1ms) | <----> | +-- datasets/    |
| +-- models/      |  v4.1+  |                    |  sync  | +-- models/      |
| +-- outputs/     |         | Small files cached  |        | +-- outputs/     |
+------------------+         | Large reads -> S3   |        +------------------+
                             +--------------------+
                             Mount target in VPC
                             (one per AZ)
```

### Data Flow

- **Small file reads (<128KB)**: Served from high-performance storage at ~1ms latency
- **Large file reads (>=128KB)**: Streamed directly from S3 at TB/s throughput (no S3 Files storage charge)
- **Writes**: Go to high-performance storage (durable immediately), synced to S3 within ~60s
- **Unused data**: Automatically expired from high-perf storage after configurable window (1-365 days, default 30)

## Performance Specifications

| Metric | Value |
|--------|-------|
| Aggregate read throughput per file system | Up to TB/s |
| Aggregate write throughput per file system | 1-5 GiB/s |
| Maximum read IOPS per file system | 250,000 |
| Maximum write IOPS per file system | 50,000 |
| Maximum per-client read throughput | 3 GiB/s |
| Import from S3 IOPS | 2,400 objects/s per file system |
| Import from S3 throughput | 700 MB/s |
| Export to S3 IOPS | 800 files/s per file system |
| Export to S3 throughput | 2,700 MB/s |
| Maximum connections per file system | 25,000 |

## Pricing

S3 Files has no provisioned capacity. You pay for:

1. **High-performance storage**: $0.30/GB-month (only for actively cached data)
2. **Data writes** (to high-perf storage): $0.06/GB
3. **Data reads** (from high-perf storage): $0.03/GB
4. **Large reads (>=128KB)**: Streamed from S3 at S3 GET request rates -- **no S3 Files charge**
5. **S3 storage**: Standard S3 rates still apply for data in the bucket

### Cost Example for Training on EKS

If active working set is ~50GB (datasets loaded, a few checkpoints cached):
- High-perf storage: 50GB x $0.30 = **$15/month**
- Plus S3 storage and access charges

Compare with FSx for Lustre:
- 1200 GiB PERSISTENT_2 @ $0.145/GB = **$174/month** (fixed, regardless of usage)

## Step-by-Step

### Prerequisites

- An S3 general-purpose bucket (e.g., `<bucket-name>`)
- S3 Versioning enabled on the bucket (required by S3 Files)
- EKS cluster in the same VPC
- EFS CSI driver installed (S3 Files uses the same CSI driver as EFS)

### 1. Create the S3 File System

#### Via AWS CLI

```bash
export AWS_REGION=us-east-2

# Create the file system linked to your S3 bucket
aws s3api create-file-system \
  --bucket <bucket-name> \
  --file-system-configuration '{
    "PerformanceMode": "generalPurpose",
    "ThroughputMode": "elastic"
  }' \
  --region $AWS_REGION

# Note the FileSystemId from the output (e.g., fs-0aa860d05df9afdfe)
```

#### Via Terraform (future integration)

At the time of writing (April 2026), S3 Files is new and Terraform provider support may be limited. Check the `aws` provider changelog for `aws_s3_file_system` resource availability. Until then, use CLI or console to create the file system and manage the mount targets.

### 2. Create Mount Targets

Mount targets provide network access to the file system within your VPC. You need one per AZ where your pods run.

```bash
# Get the VPC subnet for your target AZ
SUBNET_ID="subnet-0c0eba4f645d647cc"  # us-east-2a private subnet

# Get the node security group (pods need NFS port 2049 access)
NODE_SG=$(aws eks describe-cluster --name <cluster-name> \
  --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text \
  --region $AWS_REGION)

# Create a security group for the mount target
MT_SG=$(aws ec2 create-security-group \
  --group-name <cluster-name>-s3files-mt \
  --description "S3 Files mount target SG" \
  --vpc-id vpc-0d0cced196d7bda32 \
  --query "GroupId" --output text \
  --region $AWS_REGION)

# Allow NFS from EKS nodes
aws ec2 authorize-security-group-ingress \
  --group-id $MT_SG \
  --protocol tcp \
  --port 2049 \
  --source-group $NODE_SG \
  --region $AWS_REGION

# Create the mount target
aws s3api create-mount-target \
  --file-system-id fs-0aa860d05df9afdfe \
  --subnet-id $SUBNET_ID \
  --security-groups $MT_SG \
  --region $AWS_REGION
```

### 3. Install the EFS CSI Driver (if not already installed)

S3 Files uses the EFS CSI driver (`efs.csi.aws.com`). If you already have Amazon EFS configured, the same driver works for S3 Files.

#### Via EKS Managed Add-on

```bash
# Create IAM role with AmazonS3FilesCSIDriverPolicy
# (See AWS docs for the managed policy ARN)

aws eks create-addon \
  --cluster-name <cluster-name> \
  --addon-name aws-efs-csi-driver \
  --service-account-role-arn arn:aws:iam::<account-id>:role/<cluster-name>-efs-csi \
  --region $AWS_REGION
```

#### Via Terraform

```hcl
# In main.tf cluster_addons block
aws-efs-csi-driver = {
  most_recent = true
  pod_identity_association {
    role_arn        = aws_iam_role.efs_csi.arn
    service_account = "efs-csi-controller-sa"
  }
}
```

### 4. Create PersistentVolume and PersistentVolumeClaim

```yaml
# s3files-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: s3files-pv
spec:
  capacity:
    storage: 1Ti  # Informational -- S3 Files scales automatically
  volumeMode: Filesystem
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  csi:
    driver: efs.csi.aws.com
    # volumeHandle format: <file-system-id>::<access-point-id>
    # Use the access point created by S3 Files
    volumeHandle: fs-0aa860d05df9afdfe::fsap-0123456789abcdef0
---
# s3files-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: s3files-claim
  namespace: default
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ""
  resources:
    requests:
      storage: 1Ti
  volumeName: s3files-pv
```

```bash
kubectl apply -f s3files-pv.yaml
kubectl get pvc s3files-claim  # Should show Bound
```

### 5. Mount in Pods

```yaml
spec:
  containers:
    - name: workload
      image: <your-image>
      volumeMounts:
        - name: s3files
          mountPath: /s3data
  volumes:
    - name: s3files
      persistentVolumeClaim:
        claimName: s3files-claim
```

### 6. Verify

```bash
# Test pod
kubectl run s3files-test --image=amazonlinux:2023 --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "test",
        "image": "amazonlinux:2023",
        "command": ["sh", "-c", "ls -la /s3data && echo hello > /s3data/test.txt && cat /s3data/test.txt && sleep 120"],
        "volumeMounts": [{"name": "s3files", "mountPath": "/s3data"}]
      }],
      "volumes": [{"name": "s3files", "persistentVolumeClaim": {"claimName": "s3files-claim"}}]
    }
  }'

# Check output
kubectl logs s3files-test

# Verify file appeared in S3 (after ~60s sync delay)
aws s3 ls s3://<bucket-name>/test.txt --region us-east-2

# Cleanup
kubectl delete pod s3files-test
```

## Synchronization Behavior

### File System to S3

- Writes are durable immediately on high-performance storage
- Changes are batched for ~60 seconds before syncing to S3 (reduces PUT request costs)
- Export rate: up to 800 files/s, 2,700 MB/s

### S3 to File System

- Changes in S3 appear in file system within seconds (via S3 Event Notifications)
- Only files with data on high-perf storage are updated immediately
- Expired files are updated on next access
- Import rate: up to 2,400 objects/s, 700 MB/s

### Conflict Resolution

- **S3 is the source of truth**: If the same file is modified on both the file system and S3 before sync completes, the S3 version wins
- Conflicting file system version is moved to a `.nfs-conflicts/` directory

### First Access Latency

- First `ls` of a directory imports metadata for all files in that directory
- Files below the import size threshold (default 128KB) have their data imported asynchronously
- Subsequent directory listings and file access are fast

## Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| NFS close-to-open consistency | Writes not visible to other clients until file closed and reopened | Acceptable for non-training workloads; use FSx for POSIX consistency |
| No `nconnect` mount option | Limited per-client NFS throughput | Large reads bypass NFS (streamed from S3); small file reads cached |
| No hard links | Minor -- rarely needed | Use symlinks or separate files |
| Rename = copy+delete on S3 | Directory renames with millions of files are slow on S3 | Avoid renaming large directory trees; use unique paths instead |
| S3 key limit 1024 bytes | Deep directory paths may not export | Keep paths reasonable |
| No Kerberos auth | NFS security via POSIX UID/GID only | Use IAM policies at file system level |
| No archive storage class access | Glacier objects not accessible via file system | Restore to Standard first |
| Max file size 48 TiB | Sufficient for ML workloads | Not a practical limitation |
| S3 Versioning required | Must enable on bucket | Already enabled on `<bucket-name>` |

## Training Framework Integration

> **Reference-specific details:** See `examples/AGENTS.md` for framework-specific S3 Files integration patterns and recommended use cases.

### Recommended Use Cases (Not Training)

S3 Files is well-suited for non-training workflows in the RL pipeline:

| Workflow | S3 Files Mount | Notes |
|----------|---------------|-------|
| Dataset browsing | `/s3data/datasets/` | Browse training datasets, inspect files |
| Model weight download | `/s3data/models/` | Stage SFT weights from HuggingFace to S3 once, access as files |
| Post-training analysis | `/s3data/checkpoints/` | Inspect checkpoint files, run offline evaluation |
| Report generation | `/s3data/logs/` | Read training logs and rollout videos |

### Not Recommended For

- **Active RL training**: Use FSx for Lustre (POSIX consistency, predictable latency)
- **FSDP checkpoint saves**: Rapid overwrites and gathers across ranks need POSIX semantics
- **MLflow logging**: Updates files in-place; sync delay may lose recent logs on crash

## Validation Checklist

- [ ] S3 file system is in `AVAILABLE` state
- [ ] Mount target exists in the target AZ
- [ ] Security group allows NFS port 2049 from EKS nodes
- [ ] EFS CSI driver pods are running
- [ ] PV and PVC are bound
- [ ] Test pod can `ls` mounted path and see S3 objects as files
- [ ] Test pod can write a file and verify it appears in S3 (~60s delay)
- [ ] Multi-node test: pods on different nodes see the same data

## Related Skills

- [Skill 05: Storage - FSx for Lustre](storage-fsx-lustre/SKILL.md) - Recommended for active training (POSIX, sub-ms latency)
- [Skill 05b: Storage - S3 Connector for PyTorch](storage-s3-connector-pytorch/SKILL.md) - Programmatic S3 access from training code
- [Skill 05c: Storage - Mountpoint for S3](storage-mountpoint-s3/SKILL.md) - FUSE-based S3 mount (read-heavy, no NFS overhead)
- [Skill 06: Dataset Preparation](dataset-preparation/SKILL.md) - Staging data on S3
