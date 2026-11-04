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

### EKS Module

- **`enable_cluster_creator_admin_permissions = true`** required on the EKS module or you lose admin access.
- **ECR `force_delete = true` and S3 `force_destroy = true`** needed for clean `terraform destroy`.

### Karpenter

- **Karpenter replaces managed GPU node groups.** GPU nodes are provisioned on-demand when training pods are pending. NodePool `limits` prevent runaway spending.
- **Capacity Blocks / ODCRs**: Set `capacity_type = ["reserved", "on-demand"]` and `capacity_reservation_selector = [{id = "cr-..."}]` in workloads layer. Karpenter uses `karpenter.sh/capacity-type: reserved` (NOT `capacity-block`). The `ReservedCapacity` feature gate is enabled by default since Karpenter v1.6.
- **Instance flexibility**: Default `gpu_instance_families = ["g6", "g6e", "g5"]` gives Karpenter fallback options when specific types lack capacity. Override with `gpu_instance_types` for Capacity Blocks.
- **GPU node labels**: the `gpu-training` NodePool sets BOTH `role=gpu-training` (selected by smoke-test, topology-labeler, NCCL MPIJobs, training-statefulset, tests and `tests/lib/cluster.sh`) and `node-role=gpu-training` (selected by `container-test.yaml`). Deployments created before `role` was added must re-apply the workloads layer (`./infrastructure/deploy.sh --action apply --layer workloads`); only nodes provisioned after that carry the `role` label.
- **RT-core sim pool (`enable_sim_pool`)**: optional `gpu-sim` NodePool/EC2NodeClass (labels `role=gpu-sim`, taint `nvidia.com/gpu` only). Do NOT add a custom taint: the GPU Operator DaemonSets tolerate only `nvidia.com/gpu`, so a second taint keeps the device plugin off the node. Same YAML for non-Terraform clusters: `infrastructure/manifests/gpu-sim-nodepool.yaml`.
- **GPU NodeClasses set `instanceStorePolicy: RAID0`**: kubelet and containerd (image layers, `emptyDir`) live on the NVMe instance store, striped. GPU instances carry multi-TB NVMe (g6e.24xlarge 2×1.9 TB, g7e.24xlarge 2×3.8 TB, p5en 8×3.84 TB) and the training images are 15–35 GB; on the 500 GiB gp3 root (250 MB/s) the unpack is slow. EKS AI/ML best practice. Instance store is wiped with the node: keep checkpoints, logs and shared models on FSx.
- **dcgm-exporter 4.x for Blackwell profiling metrics**: GPU Operator v24.9.2 ships dcgm-exporter 3.3.9 (DCGM 3.3.9), which cannot load DCGM's profiling module on RTX PRO 6000 (`Not collecting DCP metrics`), so `DCGM_FI_PROF_*` (SM_ACTIVE) are missing. The addons layer sets `dcgm_exporter_version = 4.6.1-4.8.4-distroless`, the metric list from the `dcgm-metrics` ConfigMap (`addons/files/dcgm-metrics.csv`, SM_ACTIVE/SM_OCCUPANCY enabled), and lets the operator own the DCGM ServiceMonitor (a separately applied one was deleted on reconcile). 4.x labels the node `hostname` (3.x: `Hostname`).
- **Upgrading addons from a version with `null_resource.dcgm_service_monitor`**: the apply runs that resource's destroy provisioner (`kubectl delete servicemonitor nvidia-dcgm-exporter -n gpu-operator`), which deletes the operator-owned ServiceMonitor of the same name. The GPU Operator does **not** recreate it on its own (tested 2026-11-03: still missing after 5 min). Restart the operator once after that apply: `kubectl rollout restart -n gpu-operator deploy/gpu-operator`. The ServiceMonitor came back ~45 s later, and Prometheus scraped DCGM again ~5 min after that (config reload).
- **Prometheus (kube-prometheus-stack) has no persistent volume**: history is lost when its pod restarts (e.g. when system nodes move AZ). Export per-run GPU metrics right after each run.
- **GPU Operator `gdrcopy.enabled: true` blocks g4dn/g5 GPUs** when the AMI manages the driver (`driver.enabled: false`). The EKS AL2023 GPU AMI forces the proprietary driver on g4dn/g5 and ships `gdrdrv.ko` only for the open flavor, so the validator's `lsmod | grep gdrdrv` gate never passes and `nvidia.com/gpu` is never allocatable. Set `gdrcopy.enabled: false`. It installs nothing in host-driver mode; p5en still loads gdrdrv via the AMI, and EFA GPUDirect RDMA does not depend on it.
- **`target_az`** (cluster layer) is the AZ of the system nodes and the default for `fsx_az`. FSx and the GPU NodePool zone constraint follow `fsx_az` (storage layer). For Capacity Blocks keep both in the reservation's AZ: set `target_az` there and leave `fsx_az` unset.
- **VPC has subnets in every AZ (up to 3), in fixed region order**. System nodes are pinned to `target_az` (cluster layer); FSx is placed in `fsx_az` (storage layer, default `target_az`), and the GPU NodePools follow FSx (`deploy.sh` passes the storage output `fsx_az` to workloads). Subnets are indexed by position, so their order must not depend on `target_az`, or an AZ change replaces every subnet.
- **Moving FSx to where GPU capacity is** (`TF_VAR_fsx_az=<az>`, storage then workloads; `infrastructure/capacity/fsx-move.sh`) replaces only FSx + its data-repository association + PV + PVC (the PVC has `replace_triggered_by` the PV, since a bound PVC blocks PV deletion) and re-pins the GPU pools. System nodes stay in `target_az`: they do not mount FSx. Do not change `target_az` for a capacity move (it replaces the system node group while pods run). **Upgrading a stack whose storage state predates the `fsx_az` output:** the first storage apply places FSx in `target_az`'s subnet. If FSx is in another AZ, run that apply with `TF_VAR_fsx_az=<FSx's AZ>`, or it re-creates FSx.
- **`gpu_any_az = true`** (workloads) drops the zone pin on the GPU NodePools to find capacity; FSx stays in `fsx_az`. Use it to catch a node, then move FSx to the node's AZ. See `infrastructure/capacity/`.
- **On-demand g6e/g7e/p5.4xlarge capacity is often unavailable for hours** (2026-10: ICE in every us-east-2 AZ at times; spot placement score 9 was no guide). Catch any acceptable type with a placeholder pod, keep the node with a CPU-only keeper pod (the gpu-sim pool consolidates an empty node after 10 min), size runs to the node: `infrastructure/capacity/`.

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
