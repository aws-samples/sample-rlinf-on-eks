<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# RLinf-on-EKS

**Prescriptive Deployment of the RLinf Framework on Amazon EKS for Embodied and Agentic AI**

Deploy [RLinf](https://github.com/RLinf/RLinf) reinforcement learning (RL) workloads on Amazon Elastic Kubernetes Service (EKS).

RLinf-on-EKS is a skills-based framework that provides working, prescriptive deployments of the RLinf RL infrastructure on GPU (Graphics Processing Unit)-accelerated Kubernetes. An AI agent uses prescriptive [skills](.opencode/skills/) to provision infrastructure, build containers, and validate the training pipeline end-to-end.

---

## About RLinf

[RLinf](https://github.com/RLinf/RLinf) (pinned to main@d844e648, post-v0.4, 2026-10-22; 3k+ GitHub stars) is a flexible and scalable open-source RL infrastructure for Embodied and Agentic AI. Key capabilities:

- **RL Algorithms**: GRPO (Group Relative Policy Optimization), PPO, DAPO, SAC (Soft Actor-Critic), CrossQ, RLPD, SAC-Flow, DSRL, DAgger
- **Simulators**: ManiSkill, LIBERO, RoboTwin, CALVIN, RoboCasa, IsaacLab, MetaWorld, BEHAVIOR
- **Models**: OpenVLA, OpenVLA-OFT, pi0, pi0.5, GR00T, StarVLA, LingBot-VLA, Dexbotic
- **Backends**: FSDP (Fully Sharded Data Parallel) + HuggingFace/SGLang/vLLM, Megatron + SGLang/vLLM
- **Architecture**: Macro-to-micro flow transformation achieves up to 2.434x throughput

Papers: [RLinf System](https://arxiv.org/abs/2509.15965), [RLinf-VLA](https://arxiv.org/abs/2510.06710) | Docs: [rlinf.readthedocs.io](https://rlinf.readthedocs.io/en/latest/)

---

## Architecture

### Single-Node PPO Training (ManiSkill / LIBERO (Lifelong Embodied Robot Operation))

![RLinf PPO Infrastructure](examples/diagrams/infra-rlinf-ppo.drawio.svg)

Karpenter provisions a single GPU node on-demand when the training Job is submitted: a g7e.12xlarge (2x RTX PRO 6000) by default for ManiSkill and LIBERO + pi0, or a p5en.48xlarge (8x H200) for the large-GPU settings and Wan. The pod pulls its container image from ECR (built via CodeBuild's two-stage process) and mounts FSx for Lustre to access pre-trained model weights and write checkpoints. All RL components — actor, rollout, environment, and PPO trainer — run colocated inside a single pod with 8 H200 GPUs allocated via the GPU Operator's device plugin. The actor model occupies GPUs 0-7, rollout inference runs on GPUs 4-7, and the GPU-accelerated ManiSkill3 simulator shares GPUs 0-3 with environment workers. This split maximizes throughput by overlapping simulation rendering with rollout inference on separate GPU subsets. For LIBERO (CPU-based MuJoCo rendering), all components share GPUs 0-7 in a combined placement instead.

### Multi-Node SFT (Supervised Fine-Tuning) (DreamZero)

![DreamZero SFT Infrastructure](examples/dreamzero/diagrams/infra-dreamzero-sft.drawio.svg)

Training runs on two p5en.48xlarge nodes (16x H200 GPUs total) provisioned by Karpenter. A KubeRay `RayJob` launches an embedded RayCluster (head 1 + worker 1); the KubeRay operator elects the Ray head and runs the Ray-agnostic launcher as its entrypoint once the cluster is ready — there is no `torchrun` and no manual rendezvous. Each pod contributes 8 GPU ranks (16 total). RLinf's `Cluster` scheduler shards the 16.48B model with FSDP2 `full_shard` across all 16 GPUs (parameters are sharded, not replicated); gradient synchronization flows through NCCL over EFA RDMA using all 16 EFA interfaces per node. `shutdownAfterJobFinishes` tears the cluster down on completion. Both pods mount a shared 2.4 TiB FSx for Lustre filesystem holding model weights, the DROID dataset, and checkpoints. FSx is backed by an S3 bucket via a Data Repository Association that auto-imports new objects and auto-exports checkpoint writes.

> Infrastructure layer details: [infrastructure/README.md](infrastructure/README.md)  
> DreamZero model architecture: [examples/dreamzero/README.md](examples/dreamzero/README.md#architecture)

---

## Quick Start: Deploy RLinf ManiSkill PPO

Deploy the [RLinf](https://github.com/RLinf/RLinf) ManiSkill PPO (Proximal Policy Optimization) example ([PPO training of OpenVLA-OFT on ManiSkill3](examples/maniskill-openvlaoft-ppo/README.md)) in 6 steps.

### Prerequisites

- AWS account with [GPU capacity](#gpu-capacity)
- [Terraform state bucket](#terraform-state-backend) created
- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.5.0
- [AWS CLI (Command Line Interface) v2](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) configured
- [kubectl](https://kubernetes.io/docs/tasks/tools/) >= 1.36

> See [Prerequisites and Setup](#prerequisites-and-setup) for full software versions and setup instructions.

### Steps

```bash
# 1. Clone and configure
git clone https://github.com/aws-samples/sample-rlinf-on-eks.git && cd sample-rlinf-on-eks

# Configure the Region and Availability Zone.
cat > infrastructure/cluster/terraform.tfvars <<EOF
region    = "us-east-1"
target_az = "us-east-1b"  # an AZ that offers g7e (or your reservation's AZ)
EOF

# ManiSkill (and LIBERO + pi0) run on one g7e.12xlarge (2x RTX PRO 6000) in the gpu-sim pool.
# DreamZero and Wan need p-class nodes in the gpu-training pool: see "GPU Capacity" below.
cat > infrastructure/workloads/terraform.tfvars <<EOF
enable_sim_pool = true
EOF

# 2. Provision infrastructure (~20 min)
./infrastructure/deploy.sh --action apply --layer all
$(terraform -chdir=infrastructure/cluster output -raw configure_kubectl)

# 3. Build and push container image (~45-50 min via CodeBuild)
zip -r rlinf-on-eks.zip examples/Dockerfile examples/Dockerfile.profile examples/profiles.env examples/buildspec.yml examples/scripts/
aws s3 cp rlinf-on-eks.zip s3://$(terraform -chdir=infrastructure/build output -raw codebuild_source_bucket)/rlinf-on-eks.zip
BUILD_ID=$(aws codebuild start-build --project-name $(terraform -chdir=infrastructure/build output -raw codebuild_project_name) --query 'build.id' --output text)
echo "Build started: $BUILD_ID"
aws codebuild batch-get-builds --ids "$BUILD_ID" --query 'builds[0].buildStatus' --output text
# Poll until SUCCEEDED: watch -n30 "aws codebuild batch-get-builds --ids $BUILD_ID --query 'builds[0].buildStatus' --output text"

# 4. Stage model weights (per-example)
export NAMESPACE=rlinf
envsubst '${NAMESPACE}' < examples/maniskill-openvlaoft-ppo/manifests/model-download.yaml | kubectl apply -f -
kubectl logs -f job/model-download-maniskill-openvlaoft -n $NAMESPACE

# 5. Validate container (on the gpu-sim pool, where ManiSkill runs; container-test.yaml
#    is the same test for the gpu-training pool with EFA)
export IMAGE=$(terraform -chdir=infrastructure/build output -raw ecr_repository_url):latest   # or :<profile>-latest
envsubst '${IMAGE} ${REQUIRED_VENVS}' < infrastructure/manifests/container-test-sim.yaml | kubectl apply -f -
kubectl wait --for=condition=complete job/container-test -n rlinf --timeout=20m
kubectl logs job/container-test -n rlinf | tail -3    # expect: CONTAINER TEST PASSED
kubectl delete job container-test -n rlinf

# 6. Launch training
envsubst '${IMAGE} ${NAMESPACE}' < examples/maniskill-openvlaoft-ppo/manifests/maniskill-openvlaoft-ppo.yaml | kubectl apply -f -
kubectl logs -f job/rlinf-maniskill-openvlaoft -n rlinf
```
### Teardown

```bash
# Delete training job
kubectl delete job rlinf-maniskill-openvlaoft -n rlinf

# Full teardown (destroys everything in reverse layer order)
./infrastructure/deploy.sh --action destroy --layer all
```

## Training Examples

| Example | Framework | Target | GPU Requirement | Status |
|---------|-----------|--------|-----------------|--------|
| [ManiSkill+OpenVLA-OFT PPO](examples/maniskill-openvlaoft-ppo/) | RLinf main@d844e648 | ManiSkill + OpenVLA-OFT PPO | g7e.12xlarge (2x RTX PRO 6000 96GB); or p5en.48xlarge | Validated (EKS 1.37, 2026-11) |
| [Wan World Model GRPO](examples/wan-openvlaoft-grpo/) | RLinf main@d844e648 | Wan world model (LIBERO) + OpenVLA-OFT GRPO | p5en.48xlarge (8x H200 141GB) | Validated (EKS 1.37, 2026-11) |
| [LIBERO+Pi0 PPO](examples/libero-pi0-ppo/) | RLinf main@d844e648 | LIBERO Spatial + pi0 PPO | g7e.12xlarge (2x RTX PRO 6000 96GB); or p5en.48xlarge | Validated (EKS 1.37, 2026-11) |
| [DreamZero SFT](examples/dreamzero/) | RLinf main@d844e648 | LIBERO + DreamZero 14B WAM SFT | 2x p5en.48xlarge (16x H200 141GB) | Validated (EKS 1.37, 2026-11) |
| [IsaacLab GR00T PPO](examples/isaaclab-gr00t-ppo/) | RLinf main@d844e648 | Isaac Lab Franka stack-cube + GR00T N1.5 PPO (Isaac Sim 5.1, RT-core GPUs) | g7e.24xlarge (4x RTX PRO 6000 96GB); g6e.24xlarge (4x L40S); or a g7e/g6e.48xlarge | Validated (EKS 1.37, 2026-11) |

All examples were re-validated on **EKS 1.37** (Karpenter 1.15.1, NVIDIA GPU Operator v26.7.2) from a fresh `./infrastructure/deploy.sh --action apply --layer all` in 2026-11: L0 smoke, L3 full-node NCCL on 2× p5en.48xlarge (489.8 GB/s bus bandwidth at 16 GiB over 16 EFA per node), one L5 PPO step each for ManiSkill + OpenVLA-OFT and LIBERO + π₀, the L6 multi-node step, two Wan GRPO steps (~904 s/step), and DreamZero 14B SFT on 2× p5en (20 steps, 16-shard DCP checkpoint) → `.pt` conversion → LIBERO eval smoke. The IsaacLab image passed the container test (L2) and the Isaac Sim smoke test on g7e, and its training manifest ran 2 PPO steps on a g6e.48xlarge (280 s/step).

Each example README was then run exactly as written (`./validate.sh --mode cluster --skip-to 7 --level 7`, smoke edit only) from a fresh clone, on images built with the README build commands (`:latest` and each profile's `<profile>-latest`): ManiSkill (444 s/step), π₀ (428 s/step) and IsaacLab smoke on g7e.12xlarge and training on g6e.48xlarge (280 s/step); Wan (907 s/step) and DreamZero SFT → convert → eval on p5en; both showcase videos regenerated via `generate-showcase-video.sh`. Each slim profile passed L2 and the examples it serves (`dreamzero`: SFT → convert → eval; `maniskill-oft`: ManiSkill + video, Wan; `libero-pi0`: π₀ + video).

---

## Build a slim image (profiles)

The default image (`PROFILE=full`) is upstream's `embodied-maniskill_libero` target: 13 model venvs, ~48 min to build, ~18.6 GB. If you only need some of them, build a **profile** instead. A profile starts from upstream's venv-free `embodied-common-image` and installs only the venvs it lists ([`examples/profiles.env`](examples/profiles.env)). The EFA/NCCL overlay is the same as in the full image.

| Profile | Venvs | Serves | Measured (CodeBuild, 2026-10) |
|---|---|---|---|
| `full` (default) | all 13 | every example; the only profile tagged `:latest` | 49 min, 18.6 GB |
| `examples` | `openvla-oft`, `openpi`, `dreamzero` | all four examples in this repo | 37 min, 15.5 GB |
| `maniskill-oft` | `openvla-oft` | ManiSkill PPO, Wan GRPO | — |
| `libero-pi0` | `openpi` (`--env libero`, no ManiSkill) | LIBERO π₀ PPO | 32 min, 13.1 GB |
| `dreamzero` | `dreamzero` (`--env libero`, no ManiSkill) | DreamZero SFT / convert / eval | 31 min, 12.6 GB |
| `isaaclab-gr00t` | `gr00t` (`--env isaaclab`) + Isaac Sim 5.1 in the image (`ISAAC_SIM_VERSION`) | IsaacLab GR00T PPO | 78 min with `FLASH_ATTN_ARCHS=8.0;9.0;10.0;12.0`, 22.5 GB (13.7 GB without Isaac Sim) |

Most of every image is the shared part: the CUDA base, the uv-managed Python, and the EFA/NCCL/nccl-tests overlay. So a 1-venv profile is still ~12.5–13 GB, and the main saving is build time.

```bash
# Build a profile (tagged <profile>-<timestamp> and <profile>-latest; never :latest)
aws codebuild start-build --project-name $(terraform -chdir=infrastructure/build output -raw codebuild_project_name) \
  --environment-variables-override name=PROFILE,value=examples,type=PLAINTEXT

# Custom profile: name it and pass venv:model:env tuples (upstream requirements/install.sh)
aws codebuild start-build --project-name $(terraform -chdir=infrastructure/build output -raw codebuild_project_name) \
  --environment-variables-override name=PROFILE,value=my-gr00t,type=PLAINTEXT \
    'name=INSTALLS,value=gr00t_n1d7:gr00t_n1d7:maniskill_libero,type=PLAINTEXT'

# Point manifests at the profile image
export IMAGE=$(terraform -chdir=infrastructure/build output -raw ecr_repository_url):examples-latest
```

Manifests take the whole image reference in `${IMAGE}` (`<ecr repo>:<tag>`). Profile images have no default venv, so every manifest sets `VENV_NAME`. To validate a profile:

```bash
PROFILE=examples ./validate.sh --mode cluster --skip-to 1 --level 1            # build it
IMAGE_TAG=examples-latest REQUIRED_VENVS=openvla-oft,openpi,dreamzero \
  ./validate.sh --mode cluster --skip-to 2 --level 2                          # L2 on it
```

---

## Repository Structure

```
rlinf-on-eks/
├── AGENTS.md                        # Agent system instructions: thesis, skills index
├── README.md                        # This file: human-facing documentation
├── validate.sh                      # Validation harness: local checks + cluster L0-L7
├── infrastructure/                  # Layered Terraform (5 layers, orchestrated by deploy.sh)
│   ├── README.md                    #   Layer descriptions, deploy.sh usage
│   ├── AGENTS.md                    #   Terraform gotchas, provider quirks
│   ├── deploy.sh                    #   Orchestrates all layers in order
│   ├── cluster/                     #   VPC, EKS, system nodes, IAM (Identity and Access Management)
│   ├── storage/                     #   FSx for Lustre, S3, PV/PVC
│   ├── addons/                      #   Karpenter, GPU Operator, Prometheus, MLflow, KubeRay
│   ├── workloads/                   #   Karpenter NodePool/EC2NodeClass, ResourceQuota, LimitRange
│   ├── build/                       #   ECR (Elastic Container Registry) repository + CodeBuild project
│   └── manifests/                   #   Validation manifests (smoke test, container test, NCCL, topology)

├── examples/                        # RLinf training examples, Dockerfile, buildspec, manifests
│   ├── README.md                    #   Framework details, compute, quick start, DreamZero results
│   ├── AGENTS.md                    #   RLinf-specific gotchas, dependency pins, GPU sizing
│   ├── Dockerfile                   #   Container image (DLC base + RLinf + ManiSkill)
│   ├── buildspec.yml                #   CodeBuild build specification
│   └── scripts/                     #   Training launch, inference, composition scripts
├── tests/lib/                       # Validation libraries sourced by validate.sh
└── .opencode/skills/                # Deployment skills (14 shared + 5 rlinf-*)
```
---

## Prerequisites and Setup

### Software

| Tool | Version | Install | Required for |
|------|---------|---------|--------------|
| Terraform | >= 1.5.0 | https://developer.hashicorp.com/terraform/install | All steps |
| AWS CLI | v2 | https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html | All steps |
| kubectl | >= 1.36 | https://kubernetes.io/docs/tasks/tools/ | Steps 4-6 |
| Docker | Latest | https://docs.docker.com/get-docker/ | Local builds (alternative to CodeBuild) |
| Helm | >= 3.x | https://helm.sh/docs/intro/install/ | Troubleshooting only (Terraform manages Helm releases) |

### GPU Capacity

P-class GPU instances (p4d, p5, p5en) have limited availability. In most regions you will need to purchase a [Capacity Block for ML (Machine Learning)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-using.html) to reserve access to GPU nodes. You must also have sufficient [service quotas](https://docs.aws.amazon.com/servicequotas/latest/userguide/intro.html) for the instance type you intend to use.

The ManiSkill and LIBERO + pi0 examples default to one **g7e.12xlarge** (2x RTX PRO 6000 96 GB) in the `gpu-sim` pool (`enable_sim_pool = true`; on-demand, or an ODCR via `sim_capacity_type` / `sim_capacity_reservation_selector`). Wan GRPO and DreamZero need p5en.48xlarge, which usually means a Capacity Block. For those, add the following to `infrastructure/workloads/terraform.tfvars`:

```hcl
gpu_instance_types = ["p5en.48xlarge"]          # your reserved instance type
capacity_type      = ["reserved", "on-demand"]  # reserved = Capacity Block; on-demand = fallback
gpu_limit          = 16                         # 8 GPUs x max nodes
capacity_reservation_selector = [
  { id = "cr-0123456789abcdef0" }               # your Capacity Block reservation ID
]
```

These settings configure Karpenter to provision nodes from your Capacity Block. The `target_az` ensures the VPC subnet, FSx filesystem, system nodes, and GPU nodes all deploy in the same Availability Zone as your reserved capacity.

| Variable | File | Purpose |
|----------|------|---------|
| `target_az` | `infrastructure/cluster/terraform.tfvars` | AZ of your reservation — all resources co-locate here |
| `gpu_instance_types` | `infrastructure/workloads/terraform.tfvars` | Instance type(s) to provision |
| `capacity_type` | `infrastructure/workloads/terraform.tfvars` | `["reserved", "on-demand"]` for Capacity Blocks with fallback, or `["on-demand"]` |
| `capacity_reservation_selector` | `infrastructure/workloads/terraform.tfvars` | Capacity Block reservation ID or tags (required when using `reserved`) |
| `gpu_limit` | `infrastructure/workloads/terraform.tfvars` | Max total GPUs (cost guard) |

Karpenter uses `reserved` capacity type for both Capacity Blocks and ODCRs. It prioritizes reserved capacity and falls back to on-demand when reservations are exhausted. The `ReservedCapacity` feature gate is enabled by default in Karpenter >= v1.6.

**Without tfvars** (defaults): Karpenter selects the cheapest available instance from the g5, g6, and g6e families on-demand. Suitable for infrastructure validation but not production training.

### Terraform State Backend

All five infrastructure layers store state in a shared S3 bucket with separate keys per layer (e.g., `cluster/terraform.tfstate`, `storage/terraform.tfstate`). Create this bucket before running `deploy.sh`. **Choose a globally-unique bucket name** (S3 bucket names are global; do not reuse the placeholder verbatim):

```bash
# Replace with your own globally-unique bucket name
export TF_STATE_BUCKET="<your-account>-rlinf-on-eks-tfstate"

aws s3api create-bucket \
  --bucket "$TF_STATE_BUCKET" \
  --region us-east-1

aws s3api put-bucket-versioning \
  --bucket "$TF_STATE_BUCKET" \
  --versioning-configuration Status=Enabled
```

After creating the bucket, copy the backend config template and set your bucket + region:

```bash
cp infrastructure/backend.hcl.example infrastructure/backend.hcl
# Edit infrastructure/backend.hcl:
#   bucket = "<your TF_STATE_BUCKET>"
#   region = "<your region, e.g. us-east-1>"
```

`deploy.sh` passes `infrastructure/backend.hcl` to every layer via `terraform init -backend-config=...`, so you do **not** edit any `versions.tf` file. `backend.hcl` is gitignored, keeping your account-specific values out of version control. The per-layer state key and lock setting are committed in each layer's `versions.tf`. State locking uses Terraform's native S3 lockfile mechanism (`use_lockfile = true`) — no DynamoDB table is required. Versioning is recommended so you can recover state if a layer apply is interrupted.

---

## Validation

Use the included validation harness to verify your deployment before launching training:

```bash
# Offline checks (terraform fmt, kubeconform, shellcheck, path integrity)
./validate.sh --mode local

# Cluster validation -- infrastructure smoke test (GPU, EFA (Elastic Fabric Adapter), FSx)
./validate.sh --mode cluster --level 0

# Full validation up to NCCL (NVIDIA Collective Communications Library)/EFA bandwidth (recommended before first training)
./validate.sh --mode cluster --level 3
```

See [Validation Levels](examples/README.md#validation) for the full L0-L7 breakdown. L7 (`--skip-to 7 --level 7`) runs each example README exactly as written.

---

## Skills Reference

Skills are prescriptive guides for deploying Embodied/Agentic RL workloads on EKS. Shared skills are generic and reusable. RLinf-specific skills use the `rlinf-` prefix. All skills live in `.opencode/skills/`.

### Shared Skills

| # | Skill | Scope |
|---|-------|-------|
| 01 | [EKS Cluster Provisioning](.opencode/skills/eks-cluster-provisioning/SKILL.md) | Terraform: VPC (Virtual Private Cloud), EKS, GPU node groups, EFA, OIDC (OpenID Connect) |
| 02 | [GPU Instance Selection](.opencode/skills/gpu-instance-selection/SKILL.md) | EC2 instance type decision framework |
| 03 | [AMI (Amazon Machine Image) and Node Configuration](.opencode/skills/ami-and-node-configuration/SKILL.md) | NVIDIA drivers, Lustre client, EFA bootstrap |
| 04 | [Networking - EFA](.opencode/skills/networking-efa/SKILL.md) | EFA setup, NCCL tuning, security groups |
| 05 | [Storage - FSx for Lustre](.opencode/skills/storage-fsx-lustre/SKILL.md) | Shared filesystem for datasets/checkpoints |
| 05b | [Storage - S3 Connector for PyTorch](.opencode/skills/storage-s3-connector-pytorch/SKILL.md) | Direct S3 access via s3torchconnector |
| 05c | [Storage - Mountpoint for Amazon S3](.opencode/skills/storage-mountpoint-s3/SKILL.md) | POSIX-like S3 mount via CSI (Container Storage Interface) driver |
| 05d | [Storage - S3 Files](.opencode/skills/storage-s3-files/SKILL.md) | NFS (Network File System) access to S3 via Amazon S3 Files |
| 06 | [Container Image Building](.opencode/skills/container-image-building/SKILL.md) | Dockerfile, dependency management, CodeBuild CI (Continuous Integration) |
| 07 | [Kubernetes Manifests](.opencode/skills/kubernetes-manifests/SKILL.md) | Jobs, StatefulSets, resource requests |
| 08 | [Kubernetes Native Features](.opencode/skills/kubernetes-native-features/SKILL.md) | GPU Operator, topology, scheduling |
| 09 | [Deployment Method](.opencode/skills/deployment-method/SKILL.md) | CI/CD (Continuous Integration/Continuous Delivery), GitOps, experiment management |
| 10 | [Functional Testing](.opencode/skills/functional-testing/SKILL.md) | 7-level validation suite |
| 11 | [Monitoring and Observability](.opencode/skills/monitoring-observability/SKILL.md) | GPU metrics, Prometheus, Grafana, MLflow |

### RLinf-Specific Skills

| # | Skill | Scope |
|---|-------|-------|
| R1 | [RLinf Benchmarking](.opencode/skills/rlinf-benchmarking/SKILL.md) | ManiSkill, LIBERO, RoboTwin evaluation |
| R2 | [RLinf Dataset Preparation](.opencode/skills/rlinf-dataset-preparation/SKILL.md) | Download, preprocess, stage RLinf datasets |
| R3 | [RLinf Functional Testing](.opencode/skills/rlinf-functional-testing/SKILL.md) | L5 per-example test configs and model downloads |
| R4 | [RLinf Open-Source Plugins](.opencode/skills/rlinf-open-source-plugins/SKILL.md) | Ray, vLLM, SGLang, RLinf, KubeRay integration |
| R5 | [RLinf PyTorch Training Scripts](.opencode/skills/rlinf-pytorch-training-scripts/SKILL.md) | RLinf/Ray script adaptation for EKS |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the general pull-request process (fork,
branch, open a PR). This project additionally asks that you validate your changes:

### Workflow

1. Fork the repository and create a feature branch from `main` on your fork
2. Make your changes
3. Run local validation: `./validate.sh --mode local` (terraform fmt, kubeconform, shellcheck, path integrity)
4. If you have cluster access, run relevant cluster validation levels (e.g., `./validate.sh --mode cluster --level 2`)
5. Open a PR from your fork with a description of what changed and which validation levels passed

### Adding a New Training Example

1. Create a new directory `examples/<name>/` with `manifests/<name>.yaml`
2. Add a model download manifest at `examples/<name>/manifests/model-download.yaml`
3. Add a row to the [Training Examples](#training-examples) table
4. If the example needs a different `BUILD_TARGET` or `EXTRAS` build arg, document it in `examples/README.md`
5. Add a `README.md` to your example directory with architecture diagrams, results, and reproduction steps
6. Test the full deployment end-to-end using the [functional testing skill](.opencode/skills/functional-testing/SKILL.md)

### Modifying Infrastructure

Infrastructure is layered (cluster → storage → addons → workloads → build). When making changes:

- Edit only the relevant layer under `infrastructure/<layer>/`
- Use `./infrastructure/deploy.sh --action plan --layer <layer>` to preview changes
- Never apply layers out of order — downstream layers depend on upstream outputs
- If adding a new Terraform variable, update both `variables.tf` and document it in `infrastructure/README.md`
- Backend bucket/region live in `infrastructure/backend.hcl` (gitignored; template: `backend.hcl.example`), not in `versions.tf` — change them in one place

### Adding or Modifying Skills

Skills live in `.opencode/skills/`. Naming conventions:

- **Shared skills** (generic EKS/GPU/container patterns): use descriptive names (e.g., `networking-efa`, `storage-fsx-lustre`)
- **Framework-specific skills**: use the `rlinf-` prefix (e.g., `rlinf-benchmarking`, `rlinf-dataset-preparation`)

Each skill is a directory containing a `SKILL.md` file with prescriptive, step-by-step instructions. Skills should be actionable — exact commands, not general guidance. Update the [Skills Reference](#skills-reference) table when adding new skills.
