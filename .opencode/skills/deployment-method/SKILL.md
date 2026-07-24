---
name: deployment-method
description: Deploy and manage Physical AI RL training jobs on EKS including CodeBuild CI, GitOps, and experiment management
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 12: Deployment Method

## Purpose

Define the deployment workflow for RL training on EKS: how to go from code changes to a running training job, including CI/CD pipelines, GitOps patterns, and operational procedures.

> **RLinf Example:** See the `rlinf-pytorch-training-scripts` skill and `examples/AGENTS.md` for RLinf-specific deployment details including buildspec configuration, Hydra overrides, and multi-venv activation.

## Deployment Workflow

```
Code Change
    │
    ▼
┌──────────────┐
│ Git Push     │
│ (main/dev)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│ CI Pipeline  │────▶│ Build Image  │
│ (CodeBuild)  │     │ Push to ECR  │
└──────┬───────┘     └──────────────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│ Update K8s   │────▶│ kubectl apply│
│ Manifests    │     │ or ArgoCD    │
└──────┬───────┘     └──────────────┘
       │
       ▼
┌──────────────┐
│ Training Job │
│ Runs on EKS  │
└──────────────┘
```

## Option A: Manual Deployment (Getting Started)

For initial development and iteration:

### 1. Build and Push Image

**Recommended: CodeBuild** (see Skill 07 and `examples/buildspec.yml`):

```bash
# Upload source and trigger build
zip -r rlinf-on-eks.zip examples/Dockerfile examples/buildspec.yml examples/scripts/
aws s3 cp rlinf-on-eks.zip s3://$(terraform -chdir=infrastructure/build output -raw codebuild_source_bucket)/rlinf-on-eks.zip
aws codebuild start-build --project-name $(terraform -chdir=infrastructure/build output -json codebuild_project_names | jq -r '."rlinf"')
```

**Alternative: Local build**:

```bash
export ECR_URI=$(terraform -chdir=infrastructure/build output -json ecr_repository_urls | jq -r '."rlinf"')  # Replace with your reference name
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
git clone https://github.com/RLinf/RLinf.git RLinf
docker build -t $ECR_URI:latest -f examples/Dockerfile .
docker push $ECR_URI:latest
```

### 2. Update Manifest and Deploy

```bash
# Set ECR URI and deploy with envsubst
export ECR_URI=$(terraform -chdir=infrastructure/build output -json ecr_repository_urls | jq -r '."rlinf"')  # Replace with your reference name
envsubst < examples/maniskill-openvlaoft-ppo/manifests/maniskill-openvlaoft-ppo.yaml | kubectl apply -f -

# Monitor
kubectl logs -f job/rlinf-training
```

### 3. Manage Training Lifecycle

```bash
# Check status
kubectl get jobs
kubectl get pods -l app=rlinf  # Replace with your reference name

# View logs
kubectl logs -f <pod-name>

# Cancel training
kubectl delete job rlinf-training

# Clean up completed jobs
kubectl delete jobs --field-selector status.successful=1
```

## Option B: CI/CD Pipeline (AWS CodeBuild)

### CodeBuild Project

The CodeBuild project is provisioned in Terraform (`infrastructure/build/main.tf`). Source is uploaded to S3 as a zip, and the buildspec clones the research repo during pre_build.

```yaml
# buildspec.yml (see examples/buildspec.yml)
version: 0.2

env:
  variables:
    ECR_REPO: "rlinf-on-eks/rlinf"  # Replace with your cluster-name/reference-name
    UPSTREAM_REPO: "https://github.com/RLinf/RLinf.git"
    UPSTREAM_REF: "main"

phases:
  pre_build:
    commands:
      - "AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)"
      - "ECR_URI=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/${ECR_REPO}"
      - "IMAGE_TAG=${CODEBUILD_RESOLVED_SOURCE_VERSION:-$(date +%Y%m%d-%H%M%S)}"
      - "aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $ECR_URI"
      - "git clone --depth 1 --branch $UPSTREAM_REF $UPSTREAM_REPO RLinf"
  build:
    commands:
      - "docker build -t $ECR_URI:$IMAGE_TAG -t $ECR_URI:latest -f examples/Dockerfile ."
  post_build:
    commands:
      - "docker push $ECR_URI:$IMAGE_TAG"
      - "docker push $ECR_URI:latest"
```

> **Example (RLinf):** The RLinf buildspec uses `ECR_REPO: "rlinf-on-eks/rlinf"`, `RLINF_REPO: "https://github.com/RLinf/RLinf.git"`, and adds `--build-arg NO_MIRROR=1` and `--build-arg BUILD_TARGET=embodied-maniskill_libero`. See `examples/buildspec.yml`.

> **Important**: All commands containing colons (`:`) must be quoted to prevent YAML parse errors. See Skill 07 for details.

### Trigger a Build

```bash
# Upload source (only files needed by CodeBuild: Dockerfile, buildspec, scripts)
zip -r rlinf-on-eks.zip examples/Dockerfile examples/buildspec.yml examples/scripts/
aws s3 cp rlinf-on-eks.zip s3://$(terraform -chdir=infrastructure/build output -raw codebuild_source_bucket)/rlinf-on-eks.zip
aws codebuild start-build --project-name $(terraform -chdir=infrastructure/build output -json codebuild_project_names | jq -r '."rlinf"')
```

### CodeBuild Operational Notes

- **S3 source**: `CODEBUILD_RESOLVED_SOURCE_VERSION` is empty for S3 sources. Always use a fallback for image tags.
- **Cache**: `LOCAL` cache type is not supported for `BUILD_GENERAL1_2XLARGE` compute. Use `NO_CACHE`.
- **Build time**: Expect ~10-15 min for a full build (dependencies + Flash Attention compilation + ECR push).
- **Interrupted builds**: If a build fails mid-push, the image may be partially uploaded. Check ECR for untagged images.

## Option C: GitOps (ArgoCD)

### Install ArgoCD

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

### ArgoCD Application

```yaml
# argocd-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rlinf  # Replace with your reference name
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/rlinf-on-eks.git
    targetRevision: main
    path: manifests/  # Directory containing K8s manifests
  destination:
    server: https://kubernetes.default.svc
    namespace: training
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

### Directory Structure for GitOps

```
rlinf-on-eks/
├── manifests/
│   ├── namespace.yaml
│   ├── secrets.yaml          # Sealed Secrets or ExternalSecrets
│   ├── fsx-pv.yaml
│   ├── fsx-pvc.yaml
│   ├── training-job.yaml     # Current training job
│   └── kustomization.yaml
├── .opencode/skills/
├── AGENTS.md
└── Dockerfile
```

## Experiment Management Pattern

For RL research, you often run multiple experiments with different hyperparameters. Use a naming convention:

```bash
# Experiment naming: <benchmark>-<model>-<key-params>-<date>
EXPERIMENT_NAME="libero10-<key-params>-<date>"  # Replace with your benchmark
```

> **Example (RLinf):** The RLinf reference uses naming like `libero10-traj1-lr5e6-tmp16-ns8-20260331` where `libero10` is the benchmark suite, `traj1` is the trajectory count, and the remaining are hyperparameters. See `examples/AGENTS.md` for details.

### Launch Multiple Experiments

```yaml
# experiment-1.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: exp-libero10-param1  # Replace with your benchmark
  labels:
    experiment: libero10-parameter-sweep
    parameter: "value1"
spec:
  template:
    spec:
      containers:
        - name: training
          env:
            - name: EXPERIMENT_NAME
              value: "libero10-param1"
          # ... rest of spec
---
# experiment-2.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: exp-libero10-param2
  labels:
    experiment: libero10-parameter-sweep
    parameter: "value2"
spec:
  template:
    spec:
      containers:
        - name: training
          env:
            - name: EXPERIMENT_NAME
              value: "libero10-param2"
          # ... rest of spec
```

### Track Experiments

```bash
# List running experiments
kubectl get jobs -l experiment=libero10-parameter-sweep  # Replace with your benchmark

# Compare in MLflow
# All experiments log to the same MLflow experiment with different run names
```

## Rollback Procedure

### Image Rollback

```bash
# List recent images
aws ecr describe-images \
  --repository-name rlinf-on-eks/rlinf \
  --query 'sort_by(imageDetails,& imagePushedAt)[-5:].imageTags' \
  --output table

# Deploy previous version
kubectl set image job/rlinf-training training=$ECR_URI:v0.1.0
```

### Checkpoint Recovery

If training fails, resume from the last checkpoint:

```bash
# Find latest checkpoint
ls /fsx/checkpoints/<experiment>/actor/

# Update training script to resume
# Set the model path to the checkpoint path and enable resume
# (exact parameters depend on the RL framework)
```

> **Example (RLinf):** For veRL-based training, set `actor_rollout_ref.model.path` to the checkpoint path and `actor_rollout_ref.model.resume=True` in the Hydra overrides. See `examples/AGENTS.md`.

## Operational Runbook

### Pre-Flight Checks

```bash
# 1. Verify GPU nodes are available
kubectl get nodes -l role=gpu-training

# 2. Verify storage is mounted
kubectl run test-fsx --rm -it --image=busybox --overrides='
{"spec":{"tolerations":[{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}],
"nodeSelector":{"role":"gpu-training"},
"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-claim"}}],
"containers":[{"name":"test","image":"busybox","command":["ls","/fsx"],"volumeMounts":[{"name":"fsx","mountPath":"/fsx"}]}]}}'

# 3. Verify model weights exist
kubectl run test-model --rm -it --image=busybox -- ls /fsx/checkpoints/libero10-rl/actor/global_step_100  # Replace with your model path

# 4. Verify ECR image is pullable
kubectl run test-image --rm -it --image=$ECR_URI:latest -- echo "Image pull successful"
```

### During Training

```bash
# Monitor training progress
kubectl logs -f <pod-name> | grep -E "(success_rate|step|loss)"

# Check GPU utilization
kubectl exec <pod-name> -- nvidia-smi

# Check disk usage
kubectl exec <pod-name> -- df -h /fsx
```

### Post-Training

```bash
# Verify checkpoints saved
kubectl exec <pod-name> -- ls /fsx/checkpoints/<experiment>/actor/

# Archive to S3
aws s3 sync /fsx/checkpoints/ s3://YOUR-DATA-BUCKET/checkpoints/  # Replace with your bucket name

# Clean up Job
kubectl delete job rlinf-training
```

## Validation Checklist

- [ ] CI/CD pipeline builds and pushes images on code changes
- [ ] Training jobs deploy successfully from manifests
- [ ] MLflow dashboard shows experiment metrics
- [ ] Checkpoints are saved to shared storage
- [ ] Rollback procedure tested and documented
- [ ] Pre-flight checks pass before each training run

## Related Skills

- [Skill 07: Container Image Building](container-image-building/SKILL.md) - Image build process
- [Skill 09: Kubernetes Manifests](kubernetes-manifests/SKILL.md) - Manifest definitions
- [Skill 13: Functional Testing](functional-testing/SKILL.md) - Post-deployment validation
