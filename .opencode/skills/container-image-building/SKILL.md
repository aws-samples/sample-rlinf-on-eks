---
name: container-image-building
description: Build container images for Physical AI RL training including dependency conflict management, patching, and CodeBuild CI
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 07: Container Image Building

## Purpose

Build a container image that contains the full software stack for RL training: PyTorch, CUDA, RL framework, simulation environments, EFA libraries, and all dependencies. The image is pushed to Amazon ECR for use in EKS training pods.

> **Reference implementations**: The working Dockerfile lives at `examples/Dockerfile`. The content below explains the generic design decisions and patterns.
>
> **RLinf Example:** See the `rlinf-pytorch-training-scripts` skill and `examples/AGENTS.md` for RLinf-specific container build details including BUILD_TARGET, NO_MIRROR=1, and multi-venv pattern.

## Image Requirements

| Component | Version | Size Impact |
|-----------|---------|-------------|
| Base: Upstream framework image (nvidia/cuda + package manager) | PyTorch + CUDA per venv | ~14 GB |
| RL framework (`<framework>`) | As required | ~200 MB |
| Model package (`<model-package>`) | Latest | ~500 MB |
| Flash Attention 2 | Latest | ~200 MB |
| Simulation environment (`<sim-env>`) | Latest | ~500 MB |
| Rendering dependencies (e.g., Vulkan + Mesa) | System packages | ~200 MB |
| HuggingFace transformers, timm, peft | Latest compatible | ~500 MB |
| **Total estimated** | | **~17 GB** |

> **Two-Stage Build Strategy**: The correct approach is a two-stage build: (1) Build the upstream framework image from `<upstream-repo>/docker/Dockerfile` with the appropriate `BUILD_TARGET`, which handles multiple build targets with complex dependency management. (2) Layer EFA/NCCL/GDRCopy on top with the thin `examples/Dockerfile`. This preserves ALL upstream dependency management while adding EFA networking support.
>
> **Example (RLinf):** The RLinf reference uses `RLinf/docker/Dockerfile` with 15 build targets and `uv` (not pip/conda) for package management with per-model venvs (openvla, openvla-oft, openpi, gr00t, dexbotic). See `rlinf-pytorch-training-scripts` skill for details.
>
> **The vanilla `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` base IS compatible with the EFA installer** -- only the NGC PyTorch image (with HPC-X) causes conflicts. The public NCCL test image (`public.ecr.aws/hpc-cloud/nccl-tests`) proves EFA works on plain CUDA devel images.

## Dockerfile

Stage 2 builds on top of the upstream framework image. The actual Dockerfile is at `examples/Dockerfile`.

> The actual two-stage Dockerfile is at `examples/Dockerfile`. Stage 1 is the upstream framework Dockerfile built by CodeBuild. Stage 2 (shown below) adds the EFA stack:

```dockerfile
# Stage 2: EFA overlay on upstream framework image
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install EFA prerequisites (missing from upstream CUDA base)
RUN apt-get update && apt-get install -y --no-install-recommends \
    environment-modules tcl udev libevent-pthreads-2.1-7 libhwloc15 \
    && rm -rf /var/lib/apt/lists/*

# EFA installer 1.47.0 (bundles libfabric 2.4.0 + aws-ofi-nccl 1.18.0)
# DO NOT use --minimal flag -- it skips aws-ofi-nccl installation
RUN curl -sO https://efa-installer.amazonaws.com/aws-efa-installer-1.47.0.tar.gz \
    && tar -xf aws-efa-installer-1.47.0.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
    && cd .. && rm -rf aws-efa-installer*

# EFA environment
ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"
ENV FI_EFA_USE_HUGE_PAGE=0
ENV NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so

# Copy EKS training scripts
COPY scripts/ /workspace/eks/scripts/
RUN chmod +x /workspace/eks/scripts/*.sh
```

## EFA Stack Installation (Stage 2)

> The EFA stack is installed in Stage 2 of the two-stage build. The upstream framework image uses a plain `nvidia/cuda` base which is compatible with the EFA installer. Key packages installed: EFA installer 1.47.0 (bundles libfabric 2.4.0, aws-ofi-nccl 1.18.0, OpenMPI), plus prerequisites (`environment-modules`, `tcl`, `udev`, `libevent-pthreads-2.1-7`, `libhwloc15`).
>
> **DO NOT use `--minimal` flag** on `efa_installer.sh` -- it skips aws-ofi-nccl installation. The aws-ofi-nccl library installs to `/opt/amazon/ofi-nccl/lib/` (not under `x86_64-linux-gnu/` or `lib64/`).

For EFA-only validation (NCCL tests), use the pre-built public image: `public.ecr.aws/hpc-cloud/nccl-tests:cuda12.8.1-efa1.43.2-ofiv1.16.3-ncclv2.27.7-1-testsv2.16.9`

### CodeBuild: Two-Stage Build

CodeBuild builds both stages. The buildspec clones the upstream framework repo, builds Stage 1 with `BUILD_TARGET`, then builds Stage 2 with EFA overlay. Reference-specific build args (e.g., mirror settings, build targets) are configured in the reference's `buildspec.yml`.

> **Example (RLinf):** The RLinf buildspec uses `--build-arg NO_MIRROR=1` to prevent Chinese university mirror defaults and `--build-arg BUILD_TARGET=embodied-maniskill_libero` to select the multi-venv build target. See `examples/buildspec.yml`.

See `examples/buildspec.yml` for the full buildspec.

## Dependency Conflict Management

> **This is the #1 container build issue for RL training workloads.** RL training stacks combine multiple frameworks (RL framework, model packages, simulation libraries, etc.) that each pin different PyTorch versions. Naive `pip install -e .` will silently downgrade or upgrade PyTorch, breaking CUDA compatibility with the base image.

### The Problem

```
DLC base image: torch==2.4.0+cu124 (CUDA 12.4)
  ↓ pip install <framework>    → installs torch==2.6.0 (OVERWRITES)
  ↓ pip install <model-package> → installs torch==2.2.0 (OVERWRITES AGAIN)
Result: Flash Attention segfaults, CUDA mismatched, training fails
```

### Prescriptive Approach

1. **Install each framework with `--no-deps`** to prevent transitive dependency resolution from overwriting PyTorch:

```dockerfile
# BAD: Each framework overwrites PyTorch
RUN pip install "<framework>[extras]"
RUN pip install -e /workspace/<model-package>        # downgrades torch

# GOOD: Install without deps, then pin shared dependencies explicitly
RUN pip install --no-deps "<framework>[extras]" && \
    pip install --no-deps -e /workspace/<model-package> && \
    pip install "ray[default]" pyarrow datasets dill lark timm draccus
```

2. **Pin `transformers` to a version compatible with the base PyTorch version string.** DLC uses official PyTorch releases (e.g., `2.4.0+cu124`) but newer `transformers` (>=5.0) may reject them:

```dockerfile
RUN pip install "transformers>=4.40,<4.46"  # accepts DLC PyTorch version
```

3. **Pin `timm<1.0`** -- versions >=1.0 have breaking API changes that affect VLA model code:

```dockerfile
RUN pip install "timm>=0.9.10,<1.0"
```

4. **Add a version assertion** at the end of the dependency stage to catch silent downgrades:

```dockerfile
RUN python3 -c "import torch; v=torch.__version__; assert '2.4' in v, f'PyTorch downgraded to {v}!'"
```

### Framework Coexistence Warning

**Do not install both PyTorch and TensorFlow** in a training container. TensorFlow's LLVM JIT compiler (`libtensorflow_framework.so`) has symbol collisions with NVIDIA's CUDA libraries, causing `dlopen` segfaults at import time.

**Symptoms:** Process crashes with `SIGSEGV` in `llvm::DebugCounter::addCounter()` during `import tensorflow` or any module that transitively imports TF.

**Mitigations:**
- Audit all `pip install` for TensorFlow transitive dependencies: `pip show <pkg> | grep Requires`
- If a library only uses TF for image processing (JPEG encode/decode, resize), patch those ops to use PIL/torchvision equivalents
- Store patches in `examples/patches/` and apply during Docker build with `git apply`
- If TF is absolutely required, use a conda-based image instead of NGC (conda isolates CUDA libraries)

### Applying Upstream Patches

When upstream code needs modification (e.g., removing TF dependencies, backward-compat fixes), store patches in the reference directory and apply during build:

```dockerfile
# Copy upstream source
COPY <upstream-repo>/ /workspace/<upstream-repo>/

# Apply patches if needed
COPY examples/patches/ /tmp/patches/
RUN cd /workspace/<upstream-repo> && \
    for p in /tmp/patches/*.patch; do \
        echo "Applying: $p" && git apply "$p" || patch -p1 < "$p"; \
    done && rm -rf /tmp/patches
```

Patches should be minimal, well-documented, and stored in version control. When the upstream repo updates, rebase the patches.

## Build Process

### 1. Create ECR Repository

```bash
REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME=<cluster-name>/<reference-name>

aws ecr create-repository \
  --repository-name $REPO_NAME \
  --region $REGION \
  --image-scanning-configuration scanOnPush=true
```

### 2. Build the Image

The build context is the repo root. The upstream framework must be cloned into the build context first:

```bash
# From the repo root
git clone <upstream-repo-url> <upstream-repo>

docker build \
  -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest \
  -f examples/Dockerfile \
  .
```

For ARM-based build machines (e.g., M-series Mac), build for amd64:

```bash
docker buildx build \
  --platform linux/amd64 \
  -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest \
  -f Dockerfile \
  --push \
  .
```

### 3. Push to ECR

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin \
  $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Push
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

# Tag with a version for reproducibility
docker tag \
  $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest \
  $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:v0.1.0

docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:v0.1.0
```

### 4. ECR Lifecycle Policy (Cost Optimization)

```bash
aws ecr put-lifecycle-policy \
  --repository-name $REPO_NAME \
  --lifecycle-policy-text '{
    "rules": [
      {
        "rulePriority": 1,
        "description": "Keep last 10 images",
        "selection": {
          "tagStatus": "any",
          "countType": "imageCountMoreThan",
          "countNumber": 10
        },
        "action": {"type": "expire"}
      }
    ]
  }'
```

## Image Verification

### Local Test (with GPU)

```bash
docker run --gpus all --rm -it \
  $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest \
  bash -c "
    python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA {torch.version.cuda}\")' && \
    python -c 'import <framework>; print(\"Framework imported\")' && \
    python -c 'from flash_attn import flash_attn_func; print(\"Flash Attention 2 OK\")' && \
    python -c 'import transformers; print(f\"transformers {transformers.__version__}\")' && \
    nvidia-smi
  "
```

### Verify EFA Libraries in Image

```bash
docker run --rm -it \
  $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest \
  bash -c "
    fi_info -p efa 2>/dev/null && echo 'EFA provider available' || echo 'EFA not in image (OK if using host networking)'
    ldconfig -p | grep nccl && echo 'NCCL found' || echo 'NCCL not found'
  "
```

## Build Optimization

### Layer Caching Strategy

The Dockerfile is ordered by change frequency (least to most):
1. Base image + system packages (rarely changes)
2. Python dependencies (changes when adding new packages)
3. Framework source code (changes frequently)

This maximizes Docker layer cache hits.

### Multi-Stage Build (Optional)

For smaller production images, use multi-stage builds to exclude build tools:

```dockerfile
# Build stage
FROM nvcr.io/nvidia/pytorch:24.07-py3 AS builder
# ... install all build dependencies ...

# Runtime stage
FROM nvcr.io/nvidia/pytorch:24.07-py3 AS runtime
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /workspace /workspace
# ... only runtime files ...
```

### Using AWS CodeBuild (Recommended)

CodeBuild builds the image inside AWS, avoiding slow local uploads. The buildspec clones the upstream framework during pre_build:

> The full buildspec is at `examples/buildspec.yml`. Key features:
> - Two-stage build: upstream framework image (Stage 1) + EFA overlay (Stage 2)
> - Reference-specific build args (e.g., mirror settings, build target selection)
> - `NO_CACHE` (not `LOCAL` cache) required for `BUILD_GENERAL1_2XLARGE` compute

Trigger a build:

```bash
# Upload source and trigger (from repo root)
# Use zip -r (not git archive -- git archive misses uncommitted files)
zip -r <cluster-name>.zip . -x ".git/*" -x "infrastructure/*/.terraform/*"
aws s3 cp <cluster-name>.zip s3://$(terraform -chdir=infrastructure/build output -raw codebuild_source_bucket)/<cluster-name>.zip
aws codebuild start-build --project-name $(terraform -chdir=infrastructure/build output -json codebuild_project_names | jq -r '."<reference-name>"')
```

### CodeBuild Operational Notes

- **S3 Source versioning**: When using S3 as the source type, `CODEBUILD_RESOLVED_SOURCE_VERSION` is empty (only populated for Git-based sources). Always use a fallback for image tags: `IMAGE_TAG="${CODEBUILD_RESOLVED_SOURCE_VERSION:-$(date +%Y%m%d-%H%M%S)}"`.
- **Cache limitations**: `LOCAL` cache type (Docker layer cache) is **not supported** for `BUILD_GENERAL1_2XLARGE` compute. Use `NO_CACHE` or `S3` cache type for large compute.
- **YAML quoting**: Buildspec commands containing colons (`:`) in variable expansions (e.g., `$ECR_URI:latest`, `${VAR:-default}`) must be wrapped in double quotes. Without quoting, the YAML parser interprets the colon as a key-value separator and the build fails with "unexpected token."
- **Source packaging**: Use `zip -r . -x ".git/*" ...` not `git archive`. The latter only includes committed files, missing local patches and uncommitted changes.

## Validation Checklist

- [ ] Image builds successfully (CodeBuild or local)
- [ ] `nvidia-smi` works inside container
- [ ] PyTorch version matches DLC base image (not downgraded by pip installs)
- [ ] `import <framework>`, `import transformers`, `import flash_attn` all succeed
- [ ] `import <sim-env>` succeeds (for simulation targets)
- [ ] No TensorFlow installed (unless explicitly needed and tested)
- [ ] Simulation environment can be imported without hanging (test in non-interactive container)
- [ ] Image is pushed to ECR and pullable from EKS nodes
- [ ] Patches applied correctly (verify patched files differ from upstream)

## Related Skills

- [Skill 08: PyTorch Training Scripts](pytorch-training-scripts/SKILL.md) - Scripts that run inside this image
- [Skill 09: Kubernetes Manifests](kubernetes-manifests/SKILL.md) - Pod specs referencing this image
- [Skill 12: Deployment Method](deployment-method/SKILL.md) - CI/CD pipeline for image builds
