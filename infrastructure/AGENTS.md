<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# infrastructure/ -- Agent Instructions

## Critical Gotchas

### EFA Networking

- **EKS node SG needs self-referencing egress rule** (protocol `-1`, self -> self). Without it, EFA RDMA silently fails with `comp_status 5 err -22`. The ingress rule alone is NOT enough.
- **EFA installer 1.50.0 in the image (with `--disable-ngc`); >= 1.47.0 on hosts**. Versions 1.34.0/1.38.0 bundle broken libfabric 1.22. `cluster/scripts/gpu-node-bootstrap.sh` still pins the host installer at 1.47.0 (driver/kmod); bumping it requires node replacement.
- **NCCL v2.31.2-1 on the CUDA 12.8.1 base** (upstream moved to CUDA 12.8 at main@6f46f36a); sm_100 (B200) and sm_120 (RTX PRO 6000, g7e) are in the gencode list. aws-ofi-nccl 1.21.1 (EFA 1.50.0) is tested with NCCL v2.31.2-1; re-check the plugin's release notes before changing either.
- **`efa_nv_peermem` kernel module** is installed but NOT loaded on EKS GPU AMI. Needs DaemonSet with `modprobe`.
- **Single-EFA instances**: use `--mca mtl ^ofi` with OpenMPI to avoid OFI MTL conflicts with NCCL.

### Terraform Provider Quirks

- **`pre_bootstrap_user_data` is SILENTLY IGNORED for AL2023 AMI types**. Must use `cloudinit_pre_nodeadm` with `text/x-shellscript` + `application/node.eks.aws` parts.
- **`kubernetes_manifest` resource CANNOT be used in fresh cluster deploys** -- provider reads API schema during plan, fails if cluster doesn't exist yet. Use `null_resource` + `kubectl apply`.
- **Helm releases get stuck in `pending-install`** if `terraform apply` is interrupted. Manual cleanup: `helm delete <release> -n <ns>`.
- **Pod Identity replaces IRSA**: `enable_irsa = false`. CSI addons use `pod_identity_association` blocks. Updating addon from IRSA to Pod Identity via Terraform fails; workaround: `aws eks update-addon` CLI first, then import.

### EKS Version (1.37)

- **EKS 1.37 needs Karpenter 1.15.x** (1.36 needs >= 1.13; Karpenter 1.11 supports up to 1.35) and a GPU Operator that supports 1.37 (v26.7.x; v24.9 is end of support). Karpenter 1.12+ also needs `ec2:DescribeInstanceStatus` on the controller role (interruption controller). GPU Operator v26.x turns on `ccManager` by default; the addons layer sets `ccManager.enabled=false`.
- **EKS add-on images are not tied to the Kubernetes version.** `most_recent = true` gives the same `aws-ebs-csi-driver`, `aws-fsx-csi-driver`, `vpc-cni` (+ network-policy-agent), `coredns`, `eks-pod-identity-agent` and `mountpoint-s3` builds on 1.35, 1.36 and 1.37 (2026-11); only `kube-proxy` follows the version. A version upgrade does not fix CVEs in those images; a cluster-layer apply picks up newer builds in place once AWS publishes them.

### EKS Module

- **`enable_cluster_creator_admin_permissions = true`** required on the EKS module or you lose admin access.
- **ECR `force_delete = true` and S3 `force_destroy = true`** needed for clean `terraform destroy`.

### Karpenter

- **Karpenter replaces managed GPU node groups.** GPU nodes are provisioned on-demand when training pods are pending. NodePool `limits` prevent runaway spending.
- **Capacity Blocks / ODCRs**: Set `capacity_type = ["reserved", "on-demand"]` and `capacity_reservation_selector = [{id = "cr-..."}]` in workloads layer. Karpenter uses `karpenter.sh/capacity-type: reserved` (NOT `capacity-block`). The `ReservedCapacity` feature gate is enabled by default since Karpenter v1.6.
- **Instance flexibility**: Default `gpu_instance_families = ["g6", "g6e", "g5"]` gives Karpenter fallback options when specific types lack capacity. Override with `gpu_instance_types` for Capacity Blocks.
- **GPU node labels**: the `gpu-training` NodePool sets BOTH `role=gpu-training` (selected by smoke-test, topology-labeler, NCCL MPIJobs, training-statefulset, tests and `tests/lib/cluster.sh`) and `node-role=gpu-training` (selected by `container-test.yaml`). Deployments created before `role` was added must re-apply the workloads layer (`./infrastructure/deploy.sh --action apply --layer workloads`); only nodes provisioned after that carry the `role` label.
- **RT-core sim pool (`enable_sim_pool`)**: optional `gpu-sim` NodePool/EC2NodeClass (labels `role=gpu-sim`, taint `nvidia.com/gpu` only). Do NOT add a custom taint: the GPU Operator DaemonSets tolerate only `nvidia.com/gpu`, so a second taint keeps the device plugin off the node. With `enable_sim_pool=true` the `rlinf` namespace GPU quota becomes `gpu_limit + sim_gpu_limit` (24 by default).
- **GPU NodeClasses set `instanceStorePolicy: RAID0`**: kubelet and containerd (image layers, `emptyDir`) live on the NVMe instance store, striped. GPU instances carry multi-TB NVMe (g6e.24xlarge 2×1.9 TB, g7e.24xlarge 2×3.8 TB, p5en 8×3.84 TB) and the training images are 15–35 GB; on the 500 GiB gp3 root (250 MB/s) the unpack is slow. EKS AI/ML best practice. Instance store is wiped with the node: keep checkpoints, logs and shared models on FSx.
- **dcgm-exporter 4.x for Blackwell profiling metrics**: GPU Operator v24.9.2 (used before v26.7.2) shipped dcgm-exporter 3.3.9 (DCGM 3.3.9), which cannot load DCGM's profiling module on RTX PRO 6000 (`Not collecting DCP metrics`), so `DCGM_FI_PROF_*` (SM_ACTIVE) are missing. The addons layer sets `dcgm_exporter_version = 4.6.1-4.8.4-distroless`, the metric list from the `dcgm-metrics` ConfigMap (`addons/files/dcgm-metrics.csv`, SM_ACTIVE/SM_OCCUPANCY enabled), and lets the operator own the DCGM ServiceMonitor (a separately applied one was deleted on reconcile). 4.x labels the node `hostname` (3.x: `Hostname`).
- **Prometheus (kube-prometheus-stack) has no persistent volume**: history is lost when its pod restarts (e.g. when system nodes move AZ). Export per-run GPU metrics right after each run.
- **The addons layer sets GPU Operator `gdrcopy.enabled=false`** because the AMI manages the driver (`driver.enabled: false`): the EKS AL2023 GPU AMI forces the proprietary driver on g4dn/g5 and ships `gdrdrv.ko` only for the open flavor, so with gdrcopy on the validator's `lsmod | grep gdrdrv` gate never passes and `nvidia.com/gpu` is never allocatable. gdrcopy installs nothing in host-driver mode; p5en still loads gdrdrv via the AMI, and EFA GPUDirect RDMA does not depend on it.
- **`target_az`** (cluster layer) is the AZ of the system nodes and the default for `fsx_az`. FSx and the GPU NodePool zone constraint follow `fsx_az` (storage layer). For Capacity Blocks keep both in the reservation's AZ: set `target_az` there and leave `fsx_az` unset.
- **VPC has subnets in the region's first 3 AZs, in fixed region order**, with one exception: a `target_az` past the first 3 (e.g. `us-west-2d`) replaces the third subnet, so the target AZ always has one (moving `target_az` to or from such an AZ replaces that subnet). System nodes are pinned to `target_az` (cluster layer); FSx is placed in `fsx_az` (storage layer, default `target_az`), and the GPU NodePools follow FSx (`deploy.sh` passes the storage output `fsx_az` to workloads). Subnets are indexed by position, so apart from that exception their order must not depend on `target_az`, or an AZ change replaces every subnet.
- **Moving FSx** (`TF_VAR_fsx_az=<az>`, storage then workloads) replaces only FSx + its data-repository association + PV + PVC (the PVC has `replace_triggered_by` the PV, since a bound PVC blocks PV deletion) and re-pins the GPU pools. System nodes stay in `target_az`: they do not mount FSx. Do not change `target_az` to follow capacity (it replaces the system node group while pods run).

### Scarce GPU capacity

On-demand g6e/g7e/p-class capacity is often unavailable for hours in a given AZ (`InsufficientInstanceCapacity` in the Karpenter log; 2026-10 saw it in every us-east-2 AZ at times, and a high spot placement score was no guide). In order of preference:

1. **Reserve it**: an ODCR or a Capacity Block for ML (p-class only; not offered for g5/g6e/g7e). Set `capacity_type = ["reserved", "on-demand"]` and `capacity_reservation_selector = [{id = "cr-..."}]` (workloads layer), with `target_az` in the reservation's AZ.
2. **Widen the instance list**: several sizes in `sim_instance_types` / `gpu_instance_types` (and size the run to the node that lands).
3. **Chase another AZ**: set `gpu_any_az=true` (workloads) so the GPU NodePools may launch in any AZ. Once a node lands, move FSx to its AZ with `TF_VAR_fsx_az=<az>` (apply storage, then workloads) and set `gpu_any_az=false` again. Nodes outside `fsx_az` read FSx across AZs until then. Moving FSx re-creates it: it refills from S3 through the data-repository association, but a fresh FSx loads files lazily on first read, so prefetch large directories (e.g. `find /fsx/<dir> -type f | xargs -P 48 cat > /dev/null`) before a run.

### State Management

- **Always use `deploy.sh`** to apply layers in order. Direct `terraform apply` in a single layer requires manually passing `-var` flags from upstream layers.
- **Cross-layer dependencies** are resolved via `terraform output` from upstream layers. If you change an output in `cluster/`, downstream layers need re-plan.
- **State bucket must exist before first deploy.** Create manually or use the bootstrap command in the root README.
- **Backend bucket/region come from `infrastructure/backend.hcl`** (gitignored; copy from `backend.hcl.example`). `deploy.sh` passes it via `terraform init -backend-config=... -reconfigure`. Do NOT hardcode bucket/region in `versions.tf` — those blocks are partial config (`key` + `use_lockfile` only).

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
