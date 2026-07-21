<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# AGENTS.md - RLinf-on-EKS

Prescriptive deployment of the [RLinf](https://github.com/RLinf/RLinf) RL framework on Amazon EKS for Embodied and Agentic AI. See `README.md` for human-facing docs, architecture diagram, and quick start.

## Knowledge Architecture

| Question | Location |
|----------|----------|
| Always true anywhere in the repo? | This file (`AGENTS.md`) |
| True only for RLinf examples? | `examples/AGENTS.md` |
| Infrastructure/Terraform gotchas? | `infrastructure/AGENTS.md` |
| Repeatable workflow (shared)? | `.opencode/skills/<name>/SKILL.md` |
| Repeatable workflow (RLinf-specific)? | `.opencode/skills/rlinf-<name>/SKILL.md` |
| Architecture diagrams? | `examples/<example>/diagrams/` or `examples/diagrams/` |
| One-off fact? | `AGENTS.md` (root or reference), not a skill |

## Key Commands

```bash
# Local validation (no cluster needed) -- runs terraform fmt, kubeconform, shellcheck, path integrity
./validate.sh --mode local

# Cluster validation -- levels L0 (infra smoke) through L6 (multi-node training)
./validate.sh --mode cluster --level 3          # run L0-L3
./validate.sh --mode cluster --level 5 --skip-to 2 --example maniskill-openvla
./validate.sh --mode cluster --continue-on-error

# Infrastructure deploy (layered -- applies all 5 layers in order)
./infrastructure/deploy.sh --action plan --layer all
./infrastructure/deploy.sh --action apply --layer all
$(terraform -chdir=infrastructure/cluster output -raw configure_kubectl)

# Deploy a single layer (e.g., addons only)
./infrastructure/deploy.sh --action apply --layer addons

# terraform fmt check (same as validate.sh local mode)
terraform fmt -check -recursive infrastructure/

# CodeBuild container build
zip -r rlinf-on-eks.zip examples/Dockerfile examples/buildspec.yml examples/scripts/
aws s3 cp rlinf-on-eks.zip s3://$(terraform -chdir=infrastructure/build output -raw codebuild_source_bucket)/rlinf-on-eks.zip
aws codebuild start-build --project-name $(terraform -chdir=infrastructure/build output -raw codebuild_project_name)

# Deploy K8s manifests (ECR_URI required for envsubst)
export ECR_URI=$(terraform -chdir=infrastructure/build output -raw ecr_repository_url)
envsubst < infrastructure/manifests/container-test.yaml | kubectl apply -f -
```

## Repo Structure (What's Not Obvious)

- **`infrastructure/`** -- layered Terraform (5 layers: cluster → storage → addons → workloads → build). Orchestrated by `deploy.sh`. S3 backend with separate state per layer.
- **`infrastructure/cluster/`** -- VPC, EKS, system nodes, IAM.
- **`infrastructure/storage/`** -- FSx for Lustre, S3, PV/PVC.
- **`infrastructure/addons/`** -- Karpenter, GPU Operator, Prometheus, MLflow, KubeRay, MPI Operator.
- **`infrastructure/workloads/`** -- Karpenter NodePool/EC2NodeClass, ResourceQuota, LimitRange.
- **`infrastructure/build/`** -- ECR repository and CodeBuild project for the training container. Buildspec lives at `examples/buildspec.yml`.
- **`infrastructure/manifests/`** -- Infrastructure validation manifests (smoke tests, topology labeler, NCCL/EFA bandwidth tests, multi-node StatefulSet/Service for L6).
- **`examples/`** -- RLinf training examples, Dockerfile, buildspec, and manifests. Has its own `AGENTS.md` with RLinf-specific gotchas, dependency pins, and example configs.
- **`examples/dreamzero/`** -- DreamZero WAM example: manifests, scripts, diagrams, architecture docs, assets, and showcase README.
- **`.opencode/skills/`** -- prescriptive skills for agent-driven deployment. Shared skills are generic (e.g., `container-image-building`). Reference-specific skills use `rlinf-` prefix (e.g., `rlinf-benchmarking`). Load via `skill` tool when the task matches.
- **`tests/lib/`** -- `common.sh` (logging, K8s helpers), `local.sh` (offline checks), `cluster.sh` (L0-L6 live tests). Sourced by `validate.sh`.

## Validation Levels (cluster mode)

| Level | What | Timeout | Notes |
|-------|------|---------|-------|
| L0 | GPU, EFA, CUDA, FSx smoke test | 10m | `infrastructure/manifests/smoke-test.yaml` |
| L1 | CodeBuild container build + ECR push | 60m | Needs `CODEBUILD_PROJECT`, `CODEBUILD_BUCKET` |
| L2 | Container validation (10 tests) | 20m | Tests multi-venv, EFA libs, CUDA |
| L3 | NCCL/EFA bandwidth threshold | 10m | MPIJob via MPI Operator; threshold varies by instance |
| L4 | Model download to FSx | 30m | Downloads models for all examples |
| L5 | 1 training step per example | 15m each | Per-example; see `examples/AGENTS.md` |
| L6 | Multi-node training step | 20m | Requires 2+ GPU nodes; uses StatefulSet + Ray |

## Critical Gotchas

### EFA Networking
- **EKS node SG needs self-referencing egress rule** (protocol `-1`, self -> self). Without it, EFA RDMA silently fails with `comp_status 5 err -22`. The ingress rule alone is NOT enough. Verified in `infrastructure/cluster/main.tf` `efa_all_traffic_egress`.
- **EFA installer must be >= 1.47.0**. Versions 1.34.0/1.38.0 bundle broken libfabric 1.22. Installer 1.47.0 bundles libfabric 2.4.0 + aws-ofi-nccl 1.18.0.
- **NCCL v2.21.5-1 for CUDA 12.4 base** -- NCCL 2.29+ requires CUDA 12.8+.
- **`efa_nv_peermem` kernel module is installed but NOT loaded** on EKS GPU AMI. Needs DaemonSet with `modprobe`.
- **Single-EFA instances**: use `--mca mtl ^ofi` with OpenMPI to avoid OFI MTL conflicts with NCCL.

### Terraform / EKS
- **`pre_bootstrap_user_data` is SILENTLY IGNORED for AL2023 AMI types**. Must use `cloudinit_pre_nodeadm` with `text/x-shellscript` + `application/node.eks.aws` parts. Verified in `infrastructure/cluster/main.tf`.
- **`enable_cluster_creator_admin_permissions = true`** required on EKS module.
- **`kubernetes_manifest` resource CANNOT be used in fresh cluster deploys** -- provider reads API schema during plan. Use `null_resource` + `kubectl apply`.
- **Capacity Blocks**: `capacity_type = ["reserved", "on-demand"]` + `capacity_reservation_selector = [{id = "cr-..."}]` in workloads layer. Karpenter uses `reserved` capacity type (NOT `capacity-block`). `target_az` in cluster layer co-locates all resources.
- **Helm releases get stuck in `pending-install`** if `terraform apply` is interrupted. Manual cleanup needed.
- **ECR `force_delete = true` and S3 `force_destroy = true`** needed for clean `terraform destroy`.
- **Pod Identity replaces IRSA**: `enable_irsa = false` in main.tf. CSI addons use `pod_identity_association` blocks. Provider bug: updating addon from IRSA to Pod Identity fails; workaround is `aws eks update-addon` CLI first.
- **Layered deploy**: always use `infrastructure/deploy.sh` to apply layers in order. Direct `terraform apply` in a single layer requires manually passing `-var` flags from upstream layers.
- **Karpenter replaces managed GPU node groups**. GPU nodes are provisioned on-demand by Karpenter when training pods are pending. NodePool limits prevent runaway spending.

### Container Build (Universal)
- **CodeBuild needs ECR pull permission for DLC registry** (`763104351884.dkr.ecr.<region>.amazonaws.com`).
- **CodeBuild `LOCAL` cache not supported for `BUILD_GENERAL1_2XLARGE`** -- use `NO_CACHE`.
- **`CODEBUILD_RESOLVED_SOURCE_VERSION` is empty for S3 sources** -- buildspec falls back to timestamp tag.

### GPU Sizing (Universal)
- **FSDP on a single GPU cannot shard** -- full model must fit in VRAM.
- **g6.8xlarge (~$1/hr) validates infrastructure (L0-L4 + EFA) but NOT training (L5-L6)**. Use p-class instances for training.
- **DreamZero LIBERO 14B requires 2x p5en.48xlarge** (8x H200 each). FSDP2 `full_shard` across 16 GPUs; the 16.48B model is sharded.
- **DreamZero 14B DCP checkpoint = ~140-206GB** (16 FSDP shards incl. full optimizer state). A FULL FSx truncates `torch.save` mid-write (`inline_container.cc unexpected pos`) → corrupt unreadable shards. Ensure >=250GB free before an SFT run.

### DreamZero LIBERO 14B (validated end-to-end on EKS, 2026-06)
Full details + every gotcha in `examples/AGENTS.md`. Headlines:
- **FSDP2 + KubeRay RayJob multi-node (NOT torchrun/DeepSpeed/StatefulSet)**. Build target `embodied-libero` + a dedicated `dreamzero` venv. `groot` is external via `DREAMZERO_PATH`.
- **The `dreamzero` venv is built by the upstream `embodied-libero` target** (RLinf PR #1272, pinned); the EKS overlay no longer rebuilds it.
- **Apply manifests with restricted envsubst** (`envsubst '${ECR_URI} ${NAMESPACE}'`) — the KubeRay operator handles Ray head election, but the RayJob entrypoint still embeds inline bash that unrestricted envsubst would clobber.
- **Hydra `+` prefix** for keys absent from the config struct (`metadata_json_path`, `save_full_model_weights`).
- **Generate `libero_sim` metadata** (DROID checkpoint bundles only `oxe_droid`) and override `num_action_per_block=16` (DROID default 24 breaks the LIBERO forward pass).
- **Dataset is `physical-intelligence/libero`**, NOT `lerobot/libero` (schema differs).
- **Checkpoint → eval**: SFT writes a sharded FSDP DCP; convert to `.pt` offline on CPU (`convert-checkpoint.yaml`). The launcher forces `+actor.fsdp_config.save_full_model_weights=false` (the 14B config omits it → code default `True` → 16B full gather hits `nccl does not support allgather_into_tensor_coalesced`).
- **DCP finalization crash (reproduced + root-caused + fixed, 2026-06)**: torch 2.6 `dcp.save` broadcasts its multi-MB finalization object over the default NCCL PG on CUDA, racing with NCCL teardown at the end of a long (~209GB) write → `UnpicklingError: invalid load key '\x00'` on non-coordinator ranks AFTER all shards + `.metadata` are on disk. Fixed by `dcp-save-gloo-coordinator.patch` (pass a gloo PG to `dcp.save`, like torch 2.7+). Validated on 2x p5en: SUCCEEDED, zero crashes.
- **Eval** is LIBERO simulator (`eval/success_once` + in-sim rollout video). `total_num_envs=16` for 14B (128 OOMs).

## Design Principles

1. **Prescriptive over descriptive** -- skills contain exact commands, not general guidance.
2. **Validate cheap, train expensive** -- always validate EFA/NCCL on g6.8xlarge (~$1/hr) before using p5.48xlarge (~$98/hr).
3. **Infrastructure as code** -- everything in Terraform, K8s manifests, Dockerfiles. No ClickOps.
4. **ConfigMap-mounted scripts for runtime hotfixes** -- avoids expensive (~15 min) CodeBuild cycle for each fix.
5. **Layered infrastructure** -- 5 independent Terraform layers with explicit cross-layer variable passing. No implicit dependencies.

## Autonomous Session Rules

When you need clarification or a decision from the user, you **MUST** use the `question` tool. Do not ask questions in prose -- they will go unanswered during autonomous windows. Always provide structured options with clear labels. If one option is clearly better, mark it with "(Recommended)" in the label.

## EFA Version Matrix

| Installer | libfabric | aws-ofi-nccl | Status |
|-----------|-----------|-------------|--------|
| 1.34.0 | 1.22.0amzn4.0 | NOT bundled | BROKEN |
| 1.38.0 | 1.22.0amzn4.0 | NOT bundled | BROKEN |
| 1.43.0 | 2.1.0amzn4.0 | 1.16.1 | Works once SG egress fixed |
| **1.47.0** | **2.4.0amzn1.0** | **1.18.0** | Current (in Dockerfile) |
