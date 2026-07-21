<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# infrastructure/ -- Agent Instructions

## Critical Gotchas

### EFA Networking

- **EKS node SG needs self-referencing egress rule** (protocol `-1`, self -> self). Without it, EFA RDMA silently fails with `comp_status 5 err -22`. The ingress rule alone is NOT enough.
- **EFA installer must be >= 1.47.0**. Versions 1.34.0/1.38.0 bundle broken libfabric 1.22.
- **NCCL v2.21.5-1 for CUDA 12.4 base** -- NCCL 2.29+ requires CUDA 12.8+.
- **`efa_nv_peermem` kernel module** is installed but NOT loaded on EKS GPU AMI. Needs DaemonSet with `modprobe`.
- **Single-EFA instances**: use `--mca mtl ^ofi` with OpenMPI to avoid OFI MTL conflicts with NCCL.

### Terraform Provider Quirks

- **`pre_bootstrap_user_data` is SILENTLY IGNORED for AL2023 AMI types**. Must use `cloudinit_pre_nodeadm` with `text/x-shellscript` + `application/node.eks.aws` parts.
- **`kubernetes_manifest` resource CANNOT be used in fresh cluster deploys** -- provider reads API schema during plan, fails if cluster doesn't exist yet. Use `null_resource` + `kubectl apply`.
- **Helm releases get stuck in `pending-install`** if `terraform apply` is interrupted. Manual cleanup: `helm delete <release> -n <ns>`.
- **Pod Identity replaces IRSA**: `enable_irsa = false`. CSI addons use `pod_identity_association` blocks. Updating addon from IRSA to Pod Identity via Terraform fails; workaround: `aws eks update-addon` CLI first, then import.

### EKS Module

- **`enable_cluster_creator_admin_permissions = true`** required on the EKS module or you lose admin access.
- **ECR `force_delete = true` and S3 `force_destroy = true`** needed for clean `terraform destroy`.

### Karpenter

- **Karpenter replaces managed GPU node groups.** GPU nodes are provisioned on-demand when training pods are pending. NodePool `limits` prevent runaway spending.
- **Capacity Blocks / ODCRs**: Set `capacity_type = ["reserved", "on-demand"]` and `capacity_reservation_selector = [{id = "cr-..."}]` in workloads layer. Karpenter uses `karpenter.sh/capacity-type: reserved` (NOT `capacity-block`). The `ReservedCapacity` feature gate is enabled by default since Karpenter v1.6.
- **Instance flexibility**: Default `gpu_instance_families = ["g6", "g6e", "g5"]` gives Karpenter fallback options when specific types lack capacity. Override with `gpu_instance_types` for Capacity Blocks.
- **`target_az`** in cluster layer ensures VPC subnet, FSx, system nodes, and NodePool zone constraint all align to the reservation's AZ.

### State Management

- **Always use `deploy.sh`** to apply layers in order. Direct `terraform apply` in a single layer requires manually passing `-var` flags from upstream layers.
- **Cross-layer dependencies** are resolved via `terraform output` from upstream layers. If you change an output in `cluster/`, downstream layers need re-plan.
- **State bucket must exist before first deploy.** Create manually or use the bootstrap command in the root README.

### CodeBuild

- **CodeBuild needs ECR pull permission for DLC registry** (`763104351884.dkr.ecr.<region>.amazonaws.com`).
- **CodeBuild `LOCAL` cache not supported for `BUILD_GENERAL1_2XLARGE`** -- use `NO_CACHE`.
- **`CODEBUILD_RESOLVED_SOURCE_VERSION` is empty for S3 sources** -- buildspec falls back to timestamp tag.

## Layer Dependency Map

When modifying a layer, these are the downstream impacts:

| Changed Layer | Must Re-plan |
|---------------|-------------|
| cluster | storage, addons, workloads, build |
| storage | (none -- leaf) |
| addons | workloads |
| workloads | (none -- leaf) |
| build | (none -- leaf) |

## Validation

```bash
# Format check
terraform fmt -check -recursive infrastructure/

# Plan all layers (catches cross-layer issues)
./infrastructure/deploy.sh --action plan --layer all

# Validate manifests (kubeconform)
./validate.sh --mode local
```

## OIDC-authenticated clusters (`skip_kubernetes`)

If you deploy the storage layer against an EKS cluster that uses OIDC
authentication instead of IAM-based `aws eks get-token`, Terraform's Kubernetes
provider cannot authenticate to it. Workaround:

- Pass `skip_kubernetes=true` to the storage layer (skips the PV/PVC resources)
- Apply the PV/PVC via `kubectl` as a post-step after `deploy.sh` completes
