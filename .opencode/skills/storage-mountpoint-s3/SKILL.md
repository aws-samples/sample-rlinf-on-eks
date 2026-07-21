---
name: storage-mountpoint-s3
description: Mount S3 buckets as POSIX-like filesystems in training pods via the Mountpoint for Amazon S3 CSI driver
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 05c: Storage - Mountpoint for Amazon S3

## Purpose

Use the Mountpoint for Amazon S3 CSI driver to mount S3 buckets as POSIX-like filesystems in training pods. This provides a middle-ground between FSx for Lustre (full POSIX, higher cost) and the S3 Connector for PyTorch (requires code changes): Mountpoint gives you a filesystem path backed by S3, with no application code changes needed.

## When to Use Mountpoint for S3

| Factor | Mountpoint for S3 | FSx for Lustre | S3 Connector for PyTorch |
|--------|-------------------|----------------|--------------------------|
| **Setup complexity** | Low (CSI driver + PV/PVC) | Medium (filesystem + CSI) | Low (pip install + code) |
| **Code changes** | None (mount path) | None (mount path) | Required (API changes) |
| **Cost** | S3 only (~$0.023/GB/month) | ~$0.145+/GB/month | S3 only (~$0.023/GB/month) |
| **Read throughput** | High (parallel prefetch) | Very high | High |
| **Write support** | New files only (no overwrites) | Full POSIX | Full (via API) |
| **Random writes** | Not supported | Full support | Via S3 API |
| **Metadata operations** | S3 API latency | Sub-ms | N/A |
| **Best for** | Read-heavy: model loading, dataset reads | Active checkpoint I/O, scratch space | Programmatic S3 access, DCP |

### Key Limitation

Mountpoint for S3 does **not** support:
- Overwriting existing files (append or modify in place)
- Hard links, symlinks, or file locking
- `rename()` across directories

This means it works well for:
- Loading pre-trained SFT model weights (read-only)
- Reading dataset files (read-only)
- Writing new checkpoint files to a unique path per step (write-once)

It does **not** work well for:
- In-place checkpoint updates
- MLflow local log directories that update files in place
- Any workflow that modifies existing files

## Architecture

```
EKS Training Pod                Mountpoint CSI Driver            Amazon S3
+------------------+           +-------------------+           +------------------+
| /s3-data/        |           | FUSE filesystem   |           | s3://<bucket-name>/  |
| ├── models/      | ──mount── | - Prefetch cache  | ──S3 API─ | ├── models/      |
| ├── datasets/    |           | - Read-ahead      |           | ├── datasets/    |
| └── checkpoints/ |           | - Parallel reads   |           | └── checkpoints/ |
+------------------+           +-------------------+           +------------------+
```

## Step-by-Step

### 1. Create an IAM Policy

```bash
cat > mountpoint-s3-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "MountpointFullBucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::<bucket-name>"
      ]
    },
    {
      "Sid": "MountpointFullObjectAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:AbortMultipartUpload",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::<bucket-name>/*"
      ]
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name AmazonS3CSIDriverPolicy \
  --policy-document file://mountpoint-s3-policy.json
```

### 2. Create an IAM Role (via IRSA)

The IAM role is created in Terraform (see Skill 01 `iam.tf`). The Mountpoint S3 CSI driver add-on is configured with this role in `main.tf`. For manual setup outside Terraform:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

eksctl create iamserviceaccount \
  --name s3-csi-driver-sa \
  --namespace kube-system \
  --cluster <cluster-name> \
  --attach-policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/AmazonS3CSIDriverPolicy \
  --approve \
  --role-name AmazonEKS_S3_CSI_DriverRole \
  --region us-east-1 \
  --role-only
```

### 3. Install the Mountpoint for Amazon S3 CSI Driver

The Mountpoint S3 CSI driver is installed as an EKS managed add-on in Terraform:

```hcl
# In main.tf cluster_addons block
aws-mountpoint-s3-csi-driver = {
  most_recent              = true
  service_account_role_arn = aws_iam_role.mountpoint_s3.arn
  configuration_values     = jsonencode({
    node = { tolerateAllTaints = true }
  })
}
```

If installing manually outside Terraform:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws eks create-addon \
  --cluster-name <cluster-name> \
  --addon-name aws-mountpoint-s3-csi-driver \
  --service-account-role-arn arn:aws:iam::${ACCOUNT_ID}:role/AmazonEKS_S3_CSI_DriverRole \
  --configuration-values '{"node":{"tolerateAllTaints":true}}'
```

The `tolerateAllTaints` ensures the CSI driver runs on GPU nodes that have the `nvidia.com/gpu` taint.

Verify:

```bash
kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-mountpoint-s3-csi-driver
```

### 4. Create PersistentVolume and PersistentVolumeClaim

```yaml
# s3-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: s3-pv
spec:
  capacity:
    storage: 1Ti  # Informational only -- S3 is effectively unlimited
  accessModes:
    - ReadWriteMany
  mountOptions:
    - allow-delete
    - allow-overwrite        # Enable if needed for checkpoint overwrites
    - region us-east-1
    - prefix datasets/       # Optional: mount only a sub-prefix
  csi:
    driver: s3.csi.aws.com
    volumeHandle: s3-csi-driver-volume
    volumeAttributes:
      bucketName: <bucket-name>
  storageClassName: ""
---
# s3-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: s3-claim
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ""
  resources:
    requests:
      storage: 1Ti
  volumeName: s3-pv
```

```bash
kubectl apply -f s3-pv.yaml
kubectl apply -f s3-pvc.yaml
kubectl get pvc s3-claim  # Should show Bound
```

#### Multiple Mount Points for Different Purposes

You can create separate PV/PVCs for different S3 prefixes:

```yaml
# Models (read-only mount)
apiVersion: v1
kind: PersistentVolume
metadata:
  name: s3-models-pv
spec:
  capacity:
    storage: 500Gi
  accessModes:
    - ReadOnlyMany
  mountOptions:
    - region us-east-1
    - prefix models/
    - read-only
  csi:
    driver: s3.csi.aws.com
    volumeHandle: s3-models-volume
    volumeAttributes:
      bucketName: <bucket-name>
  storageClassName: ""
---
# Checkpoints (write-enabled mount)
apiVersion: v1
kind: PersistentVolume
metadata:
  name: s3-checkpoints-pv
spec:
  capacity:
    storage: 5Ti
  accessModes:
    - ReadWriteMany
  mountOptions:
    - allow-delete
    - region us-east-1
    - prefix checkpoints/
  csi:
    driver: s3.csi.aws.com
    volumeHandle: s3-checkpoints-volume
    volumeAttributes:
      bucketName: <bucket-name>
  storageClassName: ""
```

### 5. Mount in Training Pods

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training
spec:
  containers:
    - name: training
      image: <ecr-image>
      volumeMounts:
        - name: s3-models
          mountPath: /s3/models
          readOnly: true
        - name: s3-checkpoints
          mountPath: /s3/checkpoints
  volumes:
    - name: s3-models
      persistentVolumeClaim:
        claimName: s3-models-claim
    - name: s3-checkpoints
      persistentVolumeClaim:
        claimName: s3-checkpoints-claim
```

Training scripts then reference paths directly:

```bash
SFT_MODEL_PATH="/s3/models/<model-path>"
CKPT_PATH="/s3/checkpoints"
```

## Performance Tuning

### Mountpoint Cache Configuration

Mountpoint for S3 supports a local metadata and data cache to reduce S3 API calls:

```yaml
mountOptions:
  - region us-east-1
  - cache /tmp/s3-cache           # Local cache directory
  - metadata-ttl 60               # Cache metadata for 60 seconds
  - max-cache-size 50000          # 50GB local cache on NVMe
```

Use the node's local NVMe storage (available on p4d/p5 instances) as the cache directory for best performance.

### Read-Ahead and Prefetch

Mountpoint automatically prefetches sequential reads. For large model files, this provides near-native throughput. No additional configuration needed for sequential reads.

### Throughput Expectations

| Operation | Expected Throughput | Notes |
|-----------|-------------------|-------|
| Sequential read (large files) | 10-100 Gbps | Depends on instance network bandwidth |
| Small file reads | Lower (S3 API latency) | Use cache to mitigate |
| New file writes | 10+ Gbps | Multi-part upload |
| Directory listing | S3 ListObjects latency | Slower than POSIX filesystems |

## Training Framework Integration

> **Reference-specific details:** See `examples/AGENTS.md` for framework-specific Mountpoint integration patterns (model loading paths, checkpoint write conventions).

### Read-Only Model Loading (No Code Changes)

The simplest integration -- mount the S3 bucket containing SFT model weights at `/s3/models`:

```bash
# In training script
SFT_MODEL_PATH="/s3/models/<model-path>"
# from_pretrained(SFT_MODEL_PATH) reads from S3 via Mountpoint
```

### Checkpoint Writes (Requires Unique Paths)

Most RL frameworks write checkpoints to unique directories (`global_step_N`), so the write-once limitation is not a problem:

```bash
CKPT_PATH="/s3/checkpoints"
# Each checkpoint saves to /s3/checkpoints/<experiment>/actor/global_step_100/
# This creates new files each time -- compatible with Mountpoint
```

### Hybrid Approach (Recommended)

Use Mountpoint for S3 for model loading and checkpoint archival, FSx for Lustre for active training scratch:

| Data | Storage | Mount Path |
|------|---------|------------|
| SFT models | Mountpoint for S3 (read-only) | `/s3/models` |
| Datasets | Mountpoint for S3 (read-only) | `/s3/datasets` |
| Active checkpoints | FSx for Lustre | `/fsx/checkpoints` |
| Rollout videos/logs | FSx for Lustre | `/fsx/logs` |
| Checkpoint archive | Mountpoint for S3 (write) | `/s3/archive` |

## Validation Checklist

- [ ] Mountpoint CSI driver pods running on all nodes (`kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-mountpoint-s3-csi-driver`)
- [ ] PV and PVC are bound
- [ ] Test pod can `ls` the mounted S3 path
- [ ] Test pod can read a file from S3 via the mount
- [ ] Test pod can write a new file and verify it appears in S3
- [ ] Multi-node test: pods on different nodes see the same S3 data

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Mount fails with permission error | IRSA not configured or wrong policy | Check IAM role trust policy and S3 permissions |
| `Read-only file system` on write | Mount is read-only or missing `allow-delete` | Add `allow-delete` to `mountOptions` |
| Slow directory listings | S3 ListObjects API latency | Use `metadata-ttl` cache option |
| `Operation not permitted` on overwrite | Mountpoint does not support in-place overwrites by default | Use `allow-overwrite` mount option, or write to new paths |
| CSI driver not running on GPU nodes | Tainted nodes not tolerated | Set `tolerateAllTaints: true` in driver config |

## Related Skills

- [Skill 05: Storage - FSx for Lustre](storage-fsx-lustre/SKILL.md) - High-performance shared filesystem
- [Skill 05b: Storage - S3 Connector for PyTorch](storage-s3-connector-pytorch/SKILL.md) - Programmatic S3 access
- [Skill 05d: Storage - Amazon S3 Files](storage-s3-files/SKILL.md) - NFS file system access to S3 (dev/debug, dataset prep)
- [Skill 06: Dataset Preparation](dataset-preparation/SKILL.md) - Staging data on S3/FSx
