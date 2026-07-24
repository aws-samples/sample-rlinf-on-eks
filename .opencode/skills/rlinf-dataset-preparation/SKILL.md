---
name: rlinf-dataset-preparation
description: "RLinf: Download, preprocess, and stage datasets and model weights for RLinf training on EKS"
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 06: Dataset Preparation

## Purpose

Download, preprocess, and stage datasets for Physical AI RL training on EKS. The datasets include expert demonstration trajectories used for SFT (Supervised Fine-Tuning) and simulation environment assets needed for RL rollout.

## Datasets for RLinf

### Overview

| Dataset | Size | Source | Purpose |
|---------|------|--------|---------|
| **SFT Model Weights** | ~14 GB | HuggingFace Hub | Pre-trained VLA model for RL |
| **LIBERO Benchmark** | ~5 GB | GitHub + auto-download | 5 task suites, 50 demos/task |
| **RoboTwin 2.0 Assets** | ~10-20 GB | GitHub script | Robot and object meshes |
| **Pre-collected Seeds** | ~1 MB | RLinf repo | Feasible seeds for RoboTwin tasks |

### SFT Model Weights

RLinf requires a pre-trained SFT model as the starting point for RL. Available from HuggingFace:

| Model | HuggingFace Path | Description |
|-------|------------------|-------------|
| OpenVLA-OFT LIBERO-10 SFT | `RLinf/Openvla-oft-SFT-libero10-trajall` | All 500 trajectories per task |
| Pi0 Spatial Object Goal SFT | `RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal` | Pi0 model for spatial tasks |

Full collection: https://huggingface.co/RLinf

## Step-by-Step

### 1. Download SFT Model Weights

#### Option A: Direct to FSx (from a staging pod)

```yaml
# dataset-prep-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: download-sft-model
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: downloader
           image: python:3.10-slim
          command:
            - bash
            - -c
            - |
              pip install huggingface_hub
              huggingface-cli download \
                RLinf/Openvla-oft-SFT-libero10-trajall \
                --local-dir /fsx/models/sft-base/openvla-oft-sft-libero10 \
                --local-dir-use-symlinks False
          volumeMounts:
            - name: fsx
              mountPath: /fsx
      volumes:
        - name: fsx
          persistentVolumeClaim:
            claimName: fsx-claim
```

```bash
kubectl apply -f dataset-prep-job.yaml
kubectl logs -f job/download-sft-model
```

#### Option B: Download to S3, then sync to FSx

```bash
# On a local machine or CI runner
pip install huggingface_hub
huggingface-cli download \
  RLinf/Openvla-oft-SFT-libero10-trajall \
  --local-dir ./openvla-oft-sft-libero10

# Upload to S3
aws s3 sync ./openvla-oft-sft-libero10 \
  s3://YOUR-DATA-BUCKET/models/sft-base/openvla-oft-sft-libero10/

# If using FSx with S3 data repository association, files auto-import
# Otherwise, sync from S3 inside a pod:
# aws s3 sync s3://YOUR-DATA-BUCKET/models/ /fsx/models/
```

### 2. Prepare the VLA Checkpoint Directory

RLinf requires that VLA model code files are present in the checkpoint directory. This is done via the `overwrite_vla_ckpt_utils.sh` script:

```bash
# This copies modeling_prismatic.py, configuration_prismatic.py, etc.
# into the SFT model directory so from_pretrained() can load them
bash examples/overwrite_vla_ckpt_utils.sh /fsx/models/sft-base/openvla-oft-sft-libero10
```

This step must run **before** training starts. It can be an init container or a pre-training script in the training Job.

### 3. LIBERO Environment Setup

LIBERO downloads its environment assets automatically on first use. To pre-stage them:

```bash
# Inside a container with LIBERO installed
python -c "
import os, sys
os.environ['LIBERO_DATASET_DIR'] = '/fsx/datasets/libero'
sys.stdin = open('/dev/null')  # Prevent interactive prompt
import libero.libero as ll
# This triggers asset download
benchmark = ll.get_benchmark('libero_10')
print(f'Loaded {len(benchmark.get_task_names())} tasks')
"
```

> **Warning**: LIBERO's `__init__.py` calls `input()` at module import time if `LIBERO_DATASET_DIR` is not set or the directory doesn't exist. In non-interactive environments (Kubernetes pods, CI/CD), this will **hang indefinitely** with no error message. Always set `LIBERO_DATASET_DIR` as an environment variable AND redirect stdin before importing LIBERO. This is a general pattern with many robotics simulation libraries -- always test imports in non-interactive containers.

The assets are stored in `~/.libero/` by default. In a container, set `LIBERO_DATASET_DIR` to point to FSx:

```bash
export LIBERO_DATASET_DIR=/fsx/datasets/libero
```

### 4. RoboTwin 2.0 Environment Setup

RoboTwin requires explicit asset download and configuration:

```bash
# Clone and install RoboTwin
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /fsx/repos/RoboTwin
cd /fsx/repos/RoboTwin
bash script/_download_assets.sh  # Downloads robot meshes, object models

# Apply RLinf modifications
cd /workspace
bash copy_overwrite_robotwin2.sh /workspace /fsx/repos/RoboTwin
```

System dependencies for RoboTwin (must be in the container image):

```bash
apt-get install -y libvulkan1 mesa-vulkan-drivers vulkan-tools
```

### 5. Pre-collect Feasible Seeds (RoboTwin Only)

RoboTwin tasks can have infeasible initial configurations. Pre-collecting feasible seeds avoids wasted rollouts:

```bash
# Seeds are already included in the repo at:
# RLinf/rlinf/envs/robotwin/seeds/robotwin2_train_seeds.json
# RLinf/rlinf/envs/robotwin/seeds/robotwin2_eval_seeds.json

# To collect seeds for new tasks:
bash pre_collect_robotwin2_seed.sh
# Output: robotwin2_train_seeds.json
# Copy into: rlinf/envs/robotwin/seeds/
```

## Directory Layout on Shared Storage

```
/fsx/
├── models/
│   └── sft-base/
│       └── openvla-oft-sft-libero10/         # OpenVLA-OFT LIBERO-10 SFT model
│           ├── config.json
│           ├── model-*.safetensors
│           ├── tokenizer.json
│           ├── modeling_prismatic.py        # Copied by overwrite script
│           └── ...
├── datasets/
│   ├── libero/                              # LIBERO env assets
│   └── robotwin/                            # RoboTwin env assets
├── checkpoints/                             # Training outputs
│   └── rlinf/
│       └── <experiment-name>/
│           └── actor/
│               ├── global_step_25/
│               ├── global_step_50/
│               └── ...
└── logs/
    └── mlflow/
```

## Data Integrity Verification

```bash
# Verify SFT model is complete
python -c "
from transformers import AutoModelForVision2Seq, AutoTokenizer
model = AutoModelForVision2Seq.from_pretrained('/fsx/models/sft-base/openvla-oft-sft-libero10')
print(f'Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params')
tokenizer = AutoTokenizer.from_pretrained('/fsx/models/sft-base/openvla-oft-sft-libero10')
print(f'Tokenizer vocab size: {tokenizer.vocab_size}')
"
```

## Validation Checklist

- [ ] SFT model weights downloaded and verified (correct param count ~7B)
- [ ] `overwrite_vla_ckpt_utils.sh` has been run on the model directory
- [ ] LIBERO environment assets downloaded (if using LIBERO benchmark)
- [ ] RoboTwin assets downloaded (if using RoboTwin benchmark)
- [ ] Feasible seeds JSON files present in `rlinf/envs/robotwin/seeds/`
- [ ] All data is accessible from GPU training pods via shared storage mount

## Related Skills

- [Skill 05: Storage - FSx for Lustre](storage-fsx-lustre/SKILL.md) - Where data is stored
- [Skill 05c: Storage - Mountpoint for S3](storage-mountpoint-s3/SKILL.md) - Alternative storage
- [Skill 07: Container Image Building](container-image-building/SKILL.md) - Including dataset tools in image
