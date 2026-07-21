---
name: storage-s3-connector-pytorch
description: Use Amazon S3 Connector for PyTorch to read datasets and save checkpoints directly to S3
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 05b: Storage - S3 Connector for PyTorch

## Purpose

Use the Amazon S3 Connector for PyTorch (`s3torchconnector`) as an alternative or complement to FSx for Lustre. This approach reads datasets and saves checkpoints directly to/from Amazon S3 without requiring a managed filesystem.

## When to Use S3 Connector vs FSx for Lustre

| Factor | S3 Connector for PyTorch | FSx for Lustre |
|--------|--------------------------|----------------|
| **Setup complexity** | Low (pip install + IAM) | Medium (filesystem provisioning + CSI driver) |
| **Cost** | S3 storage only (~$0.023/GB/month) | $0.145+/GB/month for PERSISTENT_2 |
| **Random read latency** | 10s of ms (first byte) | Sub-ms |
| **Sequential throughput** | High (multi-part parallel) | Very high (native POSIX) |
| **Checkpoint write** | Good (DCP support) | Excellent (POSIX write) |
| **POSIX compatibility** | No (PyTorch API only) | Full POSIX |
| **Multi-node concurrent** | Via S3 API (eventually consistent for overwrites) | Full Lustre locking |
| **Best for** | Datasets, final checkpoints, cost-sensitive workloads | Active training with frequent checkpoint I/O |

### Recommended Hybrid Approach

Use **both** for different purposes:
- **S3 Connector**: Load datasets, export final model artifacts, archive checkpoints
- **FSx for Lustre**: Active checkpoint saves during training, shared scratch space

## S3 Connector for PyTorch Overview

The `s3torchconnector` package (v1.5.0+) provides:

| Class | Purpose |
|-------|---------|
| `S3MapDataset` | Map-style dataset backed by S3 objects |
| `S3IterableDataset` | Iterable-style dataset for streaming from S3 |
| `S3Checkpoint` | Save/load PyTorch checkpoints directly to/from S3 |
| `S3StorageWriter` / `S3StorageReader` | Distributed Checkpoint (DCP) support for FSDP |

## Step-by-Step

### 1. Install the Package

In your container image (Dockerfile):

```dockerfile
RUN pip install s3torchconnector[dcp]>=1.5.0
```

Or at runtime:

```bash
pip install "s3torchconnector[dcp]>=1.5.0"
```

The `[dcp]` extra installs support for PyTorch Distributed Checkpoint.

### 2. IAM Configuration

Training pods need S3 access. Use IRSA (IAM Roles for Service Accounts). This is typically configured in Terraform alongside the cluster (see Skill 01 `iam.tf`). Manual setup:

```bash
# Create IAM policy for S3 access
cat > s3-training-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::<bucket-name>",
        "arn:aws:s3:::<bucket-name>/*"
      ]
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name VLAonEKSS3Access \
  --policy-document file://s3-training-policy.json

# Create service account with IRSA (if not using Terraform)
eksctl create iamserviceaccount \
  --name training-sa \
  --namespace default \
  --cluster <cluster-name> \
  --attach-policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/VLAonEKSS3Access \
  --approve
```

### 3. Loading Datasets from S3

#### Map-Style Dataset (for indexed access)

```python
from s3torchconnector import S3MapDataset

# Load all objects under a prefix
dataset = S3MapDataset.from_prefix(
    s3_uri="s3://<bucket-name>/datasets/<dataset-a>/",
    region="us-east-1",
)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

#### Iterable Dataset (for streaming)

```python
from s3torchconnector import S3IterableDataset

dataset = S3IterableDataset.from_prefix(
    s3_uri="s3://<bucket-name>/datasets/<dataset-b>/",
    region="us-east-1",
)

loader = DataLoader(dataset, batch_size=32)
for batch in loader:
    # Process batch -- data is streamed from S3
    pass
```

### 4. Saving Checkpoints to S3

#### Standard Checkpoints

```python
from s3torchconnector import S3Checkpoint

# Create checkpoint handler
checkpoint = S3Checkpoint(region="us-east-1")

# Save model state dict
with checkpoint.writer("s3://<bucket-name>/checkpoints/step_100/model.pt") as f:
    torch.save(model.state_dict(), f)

# Load model state dict
with checkpoint.reader("s3://<bucket-name>/checkpoints/step_100/model.pt") as f:
    state_dict = torch.load(f)
```

#### FSDP Distributed Checkpoint (DCP)

For FSDP workloads, use the DCP integration:

```python
from torch.distributed.checkpoint import save, load
from s3torchconnector.dcp import S3StorageWriter, S3StorageReader

# Save FSDP distributed checkpoint directly to S3
save(
    state_dict={"model": model.state_dict()},
    storage_writer=S3StorageWriter(
        path="s3://<bucket-name>/checkpoints/step_100/",
        region="us-east-1",
    ),
)

# Load FSDP distributed checkpoint from S3
load(
    state_dict={"model": model.state_dict()},
    storage_reader=S3StorageReader(
        path="s3://<bucket-name>/checkpoints/step_100/",
        region="us-east-1",
    ),
)
```

### 5. S3 Bucket Setup

```bash
# Create bucket with versioning for checkpoint safety
aws s3api create-bucket \
  --bucket <bucket-name> \
  --region us-east-1

# Enable versioning (protects against accidental overwrites)
aws s3api put-bucket-versioning \
  --bucket <bucket-name> \
  --versioning-configuration Status=Enabled

# Set lifecycle policy to expire old checkpoint versions
cat > lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "ID": "ExpireOldCheckpoints",
      "Filter": {"Prefix": "checkpoints/"},
      "Status": "Enabled",
      "NoncurrentVersionExpiration": {"NoncurrentDays": 7}
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket <bucket-name> \
  --lifecycle-configuration file://lifecycle.json
```

### 6. Using in Training Pods

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training
spec:
  serviceAccountName: training-sa  # IRSA for S3 access
  containers:
    - name: training
      image: <ecr-image>
      env:
        - name: AWS_DEFAULT_REGION
          value: us-east-1
        # S3 Connector picks up credentials from IRSA automatically
```

## Training Framework Integration Points

> **Reference-specific details:** See `examples/AGENTS.md` for framework-specific S3 Connector integration (model loading, checkpoint saving, dataset paths).

Most RL training frameworks use local filesystem paths for:

1. **SFT model loading**: `from_pretrained(SFT_MODEL_PATH)` -- can use `from_pretrained("s3://...")` with HuggingFace S3 support
2. **Checkpoint saving**: FSDP state dict saves -- can be modified to use `S3Checkpoint` or DCP `S3StorageWriter`
3. **Dataset loading**: Task/trial metadata (not large files) -- stays in-memory
4. **Rollout videos**: Saved locally during validation -- can be uploaded to S3 post-hoc

### Minimal Integration (No Code Changes)

Stage data from S3 to local/FSx at job start, upload results at job end:

```bash
# Init container or pre-training step
aws s3 sync s3://<bucket-name>/models/sft-base/ /fsx/models/sft-base/
# ... training runs with local paths ...
# Post-training step
aws s3 sync /fsx/checkpoints/ s3://<bucket-name>/checkpoints/
```

### Deep Integration (Code Changes)

Modify checkpoint saving to use DCP with S3StorageWriter. This removes FSx as a requirement for checkpointing.

## Performance Considerations

- **Dataset loading**: S3 Connector uses multi-part parallel reads. Throughput scales with object count and worker threads.
- **Checkpoint saves**: DCP writes each shard in parallel from each FSDP rank. No rank-0 gather required (unlike a standard full-state-dict approach).
- **Network bandwidth**: EC2 instances have 25-100 Gbps to S3. Not a bottleneck for checkpoint I/O.
- **Latency**: First-byte latency is 10-50ms. For streaming datasets this is amortized. For small random reads, FSx is faster.

## Validation Checklist

- [ ] `s3torchconnector` installed in container image
- [ ] IRSA service account has S3 read/write permissions
- [ ] Test write: save a small tensor to S3 and read it back
- [ ] Test DCP: save/load an FSDP checkpoint to/from S3
- [ ] Verify no credential errors in training pods

## Related Skills

- [Skill 05: Storage - FSx for Lustre](storage-fsx-lustre/SKILL.md) - Primary shared filesystem
- [Skill 06: Dataset Preparation](dataset-preparation/SKILL.md) - Staging data
- [Skill 07: Container Image Building](container-image-building/SKILL.md) - Including s3torchconnector in image
