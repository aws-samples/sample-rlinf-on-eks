---
name: eks-cluster-provisioning
description: Provision an Amazon EKS cluster with GPU node groups, EFA networking, and OIDC for Physical AI RL training workloads
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 01: EKS Cluster Provisioning

## Purpose

Provision an Amazon EKS cluster configured for GPU-accelerated Physical AI RL training workloads. The cluster must support multi-node GPU training with EFA networking, FSx for Lustre shared storage, and the necessary OIDC/IAM foundations for add-ons.

We use **Terraform** as the primary IaC tool. Terraform provides state management, plan/apply workflow, module composition, and the ability to bundle Helm releases for open-source plugins (GPU Operator, EFA plugin, KubeRay, etc.) alongside cluster provisioning in a single apply.

> **Note**: eksctl can be used for quick prototyping. See the [eksctl quick-start appendix](#appendix-eksctl-quick-start) at the end of this skill.

## Prerequisites

- AWS CLI v2 configured with appropriate credentials
- Terraform >= 1.10.0 installed
- `kubectl` installed and matching cluster version
- `helm` >= 3.x installed (Terraform calls it via the `helm_release` provider)
- Sufficient service quotas for target GPU instances (see Skill 02)

## Decision Points

### Cluster Version

Use the latest supported EKS version (currently 1.35). GPU drivers and EFA support improve with newer versions.

### Region and Availability Zone Selection

Not all AZs support all GPU instance types or FSx for Lustre. Check capacity:

```bash
# Check which AZs support your target instance type
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=p5.48xlarge" \
  --region us-east-1 \
  --query "InstanceTypeOfferings[].Location" \
  --output table
```

Prefer `us-east-1`, `us-west-2`, or `us-east-2` for broadest GPU availability.

### Networking Mode

Use **private subnets** for GPU node groups. Training nodes do not need public IPs. The cluster control plane should have both public and private endpoint access during setup.

## Terraform Module Structure

```
infrastructure/
├── cluster/          # VPC, EKS, system nodes, IAM
│   ├── main.tf
│   ├── iam.tf
│   ├── agent.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── versions.tf
│   ├── scripts/gpu-node-bootstrap.sh
│   └── profiles/{g6-validation,p5-training,p5en-training}.tfvars
├── storage/          # FSx for Lustre, S3, PV/PVC
├── addons/           # Karpenter, GPU Operator, Prometheus, MLflow
├── workloads/        # Karpenter NodePool/EC2NodeClass, policies
├── build/            # ECR repositories, CodeBuild projects
├── manifests/        # Validation manifests (smoke test, NCCL, container test, topology)
└── deploy.sh         # Orchestrator (applies layers in order)
```

## Step-by-Step

### 1. Provider and Version Configuration

```hcl
# infrastructure/cluster/versions.tf
terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
  }

  backend "s3" {
    bucket       = "<cluster-name>-terraform-state"
    key          = "eks/terraform.tfstate"
    region       = "us-west-2"
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}
```

### 2. Variables

```hcl
# infrastructure/cluster/variables.tf
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "<cluster-name>"
}

variable "cluster_version" {
  description = "EKS Kubernetes version"
  type        = string
  default     = "1.35"
}

variable "gpu_instance_type" {
  description = "GPU instance type for training nodes (see Skill 02)"
  type        = string
  default     = "p5.48xlarge"
}

variable "gpu_node_count" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_max_nodes" {
  description = "Maximum GPU nodes for autoscaling"
  type        = number
  default     = 2
}

variable "fsx_storage_capacity" {
  description = "FSx for Lustre storage capacity in GiB (minimum 1200, increments of 2400)"
  type        = number
  default     = 4800
}

variable "fsx_throughput_per_unit" {
  description = "FSx per-unit storage throughput in MB/s per TiB (125/250/500/1000)"
  type        = number
  default     = 125
}

variable "enable_kuberay" {
  description = "Install KubeRay operator for Ray cluster management"
  type        = bool
  default     = false
}

variable "grafana_password" {
  description = "Grafana admin password. Set via TF_VAR_grafana_password env var or -var flag. Do not commit to version control."
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.grafana_password) >= 8
    error_message = "grafana_password must be at least 8 characters. Set via: export TF_VAR_grafana_password='your-password'"
  }
}

variable "enable_mpi_operator" {
  description = "Install Kubeflow MPI Operator for MPIJob CRD (required for NCCL tests)"
  type        = bool
  default     = true
}

variable "target_az" {
  description = "Target Availability Zone for ALL workload resources (GPU nodes, system nodes, FSx)"
  type        = string
  default     = ""
}

variable "reference_implementations" {
  description = "Map of reference implementations to build. Each gets an ECR repo and CodeBuild project."
  type = map(object({
    repo = string
    ref  = string
  }))
  default = {
    "<reference-name>" = {
      repo = "<upstream-repo-url>"
      ref  = "v0.2"
    }
  }
}
```

### 3. VPC and EKS Cluster

```hcl
# infrastructure/cluster/main.tf
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# --- VPC ---
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 2)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }

  tags = {
    Project     = "<cluster-name>"
    Environment = "training"
  }
}

# --- EKS Cluster ---
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = var.cluster_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # Allow the Terraform caller to administer the cluster via kubectl
  enable_cluster_creator_admin_permissions = true

  # Enable IRSA is disabled -- using EKS Pod Identity instead
  enable_irsa = false

  # EKS Managed Add-ons
  cluster_addons = {
    coredns    = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni    = { most_recent = true }

    # Pod Identity Agent -- required for EKS Pod Identity
    eks-pod-identity-agent = { most_recent = true }

    aws-ebs-csi-driver = {
      most_recent = true
      pod_identity_association = [{
        role_arn        = aws_iam_role.ebs_csi.arn
        service_account = "ebs-csi-controller-sa"
      }]
    }

    aws-fsx-csi-driver = {
      most_recent = true
      pod_identity_association = [{
        role_arn        = aws_iam_role.fsx_csi.arn
        service_account = "fsx-csi-controller-sa"
      }]
    }

    aws-mountpoint-s3-csi-driver = {
      most_recent = true
      pod_identity_association = [{
        role_arn        = aws_iam_role.mountpoint_s3.arn
        service_account = "s3-csi-driver-sa"
      }]
      configuration_values = jsonencode({
        node = { tolerateAllTaints = true }
      })
    }
  }

  # --- Node Groups ---
  eks_managed_node_groups = {
    # System nodes for cluster services (CoreDNS, monitoring, etc.)
    system = {
      instance_types = ["m5.2xlarge"]
      min_size       = 2
      desired_size   = 2
      max_size       = 4

      labels = { role = "system" }

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs         = { volume_size = 100, volume_type = "gp3" }
        }
      }
    }

    # GPU training nodes -- uses module built-in EFA support
    gpu-training = {
      instance_types = [var.gpu_instance_type]
      ami_type       = "AL2023_x86_64_NVIDIA"

      min_size     = 0
      desired_size = var.gpu_node_count
      max_size     = var.gpu_max_nodes

      # Module built-in EFA support: auto-discovers EFA interface count,
      # configures network interfaces, and creates EFA security group rules.
      enable_efa_support = true

      # Placement group for multi-node EFA (low-latency inter-node)
      create_placement_group   = true
      placement_group_strategy = "cluster"

      labels = {
        role                     = "gpu-training"
        "nvidia.com/gpu.present" = "true"
      }

      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      # Bootstrap: AL2023 uses cloudinit_pre_nodeadm (NOT pre_bootstrap_user_data).
      # pre_bootstrap_user_data is silently ignored for AL2023 AMI types.
      cloudinit_pre_nodeadm = [
        {
          content_type = "text/x-shellscript; charset=\"us-ascii\""
          content      = file("${path.module}/scripts/gpu-node-bootstrap.sh")
        },
        {
          content_type = "application/node.eks.aws"
          content      = <<-EOT
            ---
            apiVersion: node.eks.aws/v1alpha1
            kind: NodeConfig
            spec:
              featureGates:
                FastImagePull: true
          EOT
        },
      ]

      # Root volume
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs         = { volume_size = 500, volume_type = "gp3" }
        }
      }

      tags = {
        "k8s.io/cluster-autoscaler/enabled"             = "true"
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
      }
    }
  }

  tags = {
    Project     = "<cluster-name>"
    Environment = "training"
  }
}
```

Key design decisions in `infrastructure/cluster/main.tf`:

- **`ami_type = "AL2023_x86_64_NVIDIA"`**: Uses the AL2023 NVIDIA GPU AMI with kernel 6.1+ for DMA-BUF support. Required for efficient EFA RDMA.
- **`enable_efa_support = true`**: The EKS module v20 handles EFA setup internally. It auto-discovers EFA interface count for the instance type, configures network interfaces, and creates EFA security group rules. No external launch template or EFA security group resource is needed.
- **`create_placement_group = true`**: The module creates the cluster placement group for low-latency inter-node communication. No standalone `aws_placement_group` resource is needed.
- **`cloudinit_pre_nodeadm`**: AL2023 uses MIME multi-part cloud-init (NOT pre_bootstrap_user_data which is silently ignored). Two parts: shell script for EFA/Lustre bootstrap and NodeConfig with FastImagePull feature gate for SOCI parallel pull.
- **`enable_cluster_creator_admin_permissions = true`**: Grants the Terraform caller admin access to the cluster, replacing the need for separate `aws-auth` ConfigMap management.

### 4. Helm Releases for Open-Source Plugins

```hcl
# infrastructure/addons/main.tf

# --- NVIDIA GPU Operator ---
# Replaces standalone device plugin + DCGM exporter with a unified operator.
# Manages: device plugin, DCGM exporter, GPU Feature Discovery, container toolkit.
# Driver installation is disabled because EKS GPU AMI already includes NVIDIA drivers.
resource "helm_release" "gpu_operator" {
  name             = "gpu-operator"
  repository       = "https://helm.ngc.nvidia.com/nvidia"
  chart            = "gpu-operator"
  namespace        = "gpu-operator"
  create_namespace = true
  version          = "v24.9.2"

  # Disable driver install -- EKS GPU AMI already has NVIDIA drivers
  set {
    name  = "driver.enabled"
    value = "false"
  }

  # Enable container toolkit (required for GPU access in containers)
  set {
    name  = "toolkit.enabled"
    value = "true"
  }

  # Enable DCGM exporter (GPU metrics for Prometheus)
  set {
    name  = "dcgmExporter.enabled"
    value = "true"
  }

  # Enable GPU Feature Discovery (auto-labels nodes with GPU model, memory, etc.)
  set {
    name  = "gfd.enabled"
    value = "true"
  }

  # Tolerate GPU node taints so operator components schedule on GPU nodes
  set {
    name  = "daemonsets.tolerations[0].key"
    value = "nvidia.com/gpu"
  }
  set {
    name  = "daemonsets.tolerations[0].operator"
    value = "Exists"
  }
  set {
    name  = "daemonsets.tolerations[0].effect"
    value = "NoSchedule"
  }

  depends_on = [module.eks]
}

# --- EFA Device Plugin ---
resource "helm_release" "efa_device_plugin" {
  name       = "efa-device-plugin"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-efa-k8s-device-plugin"
  namespace  = "kube-system"

  set {
    name  = "tolerations[0].key"
    value = "nvidia.com/gpu"
  }
  set {
    name  = "tolerations[0].operator"
    value = "Exists"
  }
  set {
    name  = "tolerations[0].effect"
    value = "NoSchedule"
  }

  depends_on = [module.eks]
}

# --- kube-prometheus-stack (Prometheus + Grafana) ---
resource "helm_release" "prometheus" {
  name             = "monitoring"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  namespace        = "monitoring"
  create_namespace = true

  set {
    name  = "grafana.adminPassword"
    value = var.grafana_password
  }
  set {
    name  = "prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues"
    value = "false"
  }

  depends_on = [module.eks]
}

# --- KubeRay Operator (Optional) ---
resource "helm_release" "kuberay_operator" {
  count = var.enable_kuberay ? 1 : 0

  name             = "kuberay-operator"
  repository       = "https://ray-project.github.io/kuberay-helm/"
  chart            = "kuberay-operator"
  namespace        = "ray-system"
  create_namespace = true

  depends_on = [module.eks]
}
```

The GPU Operator is the single component that manages all NVIDIA GPU software on the cluster. With `driver.enabled = false`, it skips driver installation (already in the AMI) but handles:

- **Device plugin** -- makes `nvidia.com/gpu` resources available to pods
- **DCGM exporter** -- exposes GPU metrics (temperature, utilization, memory) to Prometheus
- **GPU Feature Discovery** -- auto-labels nodes with GPU model, memory, driver version
- **Container toolkit** -- configures the container runtime for GPU access

### 5. Outputs

```hcl
# infrastructure/cluster/outputs.tf
output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority_data" {
  description = "EKS cluster CA certificate (base64)"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.region}"
}

output "fsx_filesystem_id" {
  description = "FSx for Lustre filesystem ID"
  value       = aws_fsx_lustre_file_system.training.id
}

output "fsx_dns_name" {
  description = "FSx for Lustre DNS name"
  value       = aws_fsx_lustre_file_system.training.dns_name
}

output "fsx_mount_name" {
  description = "FSx for Lustre mount name"
  value       = aws_fsx_lustre_file_system.training.mount_name
}

output "ecr_repository_url" {
  description = "ECR repository URLs per reference implementation"
  value       = { for k, v in aws_ecr_repository.reference : k => v.repository_url }
}

output "codebuild_project_name" {
  description = "CodeBuild project names per reference implementation"
  value       = { for k, v in aws_codebuild_project.reference : k => v.name }
}

output "codebuild_source_bucket" {
  description = "S3 bucket for CodeBuild source uploads"
  value       = aws_s3_bucket.codebuild_source.bucket
}

output "upload_source_command" {
  description = "Command to upload source and trigger a build"
  value       = "zip -r <cluster-name>.zip . -x '.git/*' -x 'infrastructure/*/.terraform/*' && aws s3 cp <cluster-name>.zip s3://${aws_s3_bucket.codebuild_source.bucket}/<cluster-name>.zip && aws codebuild start-build --project-name ${aws_codebuild_project.reference[\"<reference-name>\"].name}"
}
```

### 6. Deploy

```bash
# Deploy all layers (cluster → storage → addons → workloads → build)
./infrastructure/deploy.sh --action plan --layer all
./infrastructure/deploy.sh --action apply --layer all

# Or deploy a single layer
./infrastructure/deploy.sh --action apply --layer cluster

# Configure kubectl
$(terraform -chdir=infrastructure/cluster output -raw configure_kubectl)
```

## Verification

```bash
# Confirm cluster is active
kubectl get nodes
kubectl get pods -n kube-system

# Verify GPU nodes have GPUs registered
kubectl describe node -l role=gpu-training | grep nvidia.com/gpu

# Verify EFA devices are registered
kubectl describe node -l role=gpu-training | grep vpc.amazonaws.com/efa

# Verify GPU Operator components are running
kubectl get pods -n gpu-operator

# Verify Helm releases
helm list -A
```

## Validation Checklist

- [ ] `terraform apply` completes without errors
- [ ] `kubectl get nodes` shows all node groups healthy
- [ ] GPU nodes report `nvidia.com/gpu: 8` in allocatable resources
- [ ] GPU nodes report `vpc.amazonaws.com/efa: <N>` in allocatable resources
- [ ] Pod Identity agent is running
- [ ] FSx CSI driver pod is running in `kube-system`
- [ ] EBS CSI driver pod is running in `kube-system`
- [ ] Mountpoint for S3 CSI driver is running in `kube-system`
- [ ] GPU Operator is running in `gpu-operator` namespace (includes device plugin, DCGM exporter, GPU Feature Discovery)
- [ ] EFA device plugin daemonset is running on GPU nodes
- [ ] Prometheus + Grafana are running in `monitoring` namespace

> **Reference-specific details:** See `examples/AGENTS.md` for framework-specific cluster configuration (minimum node counts, GPU requirements, and scaling recommendations).

## Cost Optimization

- Set `gpu_node_count = 0` and use Cluster Autoscaler or Karpenter to scale on demand
- Consider Capacity Reservations for guaranteed GPU availability
- Use On-Demand for training (Spot is risky for long RL training jobs due to interruptions)
- Scale down system nodes to minimum during idle periods
- Use `terraform destroy -target=module.eks.eks_managed_node_groups["gpu-training"]` to remove GPU nodes without destroying the cluster

## Troubleshooting

### Helm Release Stuck in `pending-install`

If `terraform apply` is interrupted during Helm release installation (e.g., timeout, network error, Ctrl+C), the release may be stuck in `pending-install` state. Subsequent `terraform apply` will fail with "cannot re-use a name that is still in use."

```bash
# Check for stuck releases
helm list -A --all | grep pending

# Uninstall the stuck release
helm uninstall <release-name> -n <namespace>

# Re-run terraform apply
terraform apply
```

### FSx Filesystem Stuck in CREATING

FSx for Lustre creation takes 10-20 minutes. If `terraform apply` times out, the filesystem may still be creating. Check status:

```bash
aws fsx describe-file-systems --query "FileSystems[?Tags[?Key=='Name' && Value=='<cluster-name>-training']].Lifecycle" --output text
```

If it shows `CREATING`, wait for it to become `AVAILABLE`, then re-run `terraform apply`.

### GPU Nodes Not Joining Cluster

If GPU nodes show `NotReady` or don't appear in `kubectl get nodes`:

1. Check the node bootstrap logs: `aws ec2 get-console-output --instance-id <id>`
2. Verify the security group allows nodes to reach the EKS API server
3. Check that the node IAM role has the `AmazonEKSWorkerNodePolicy` attached

## Appendix: eksctl Quick-Start

For rapid prototyping without Terraform, use eksctl:

```bash
# Minimal GPU cluster
eksctl create cluster \
  --name <cluster-name>-dev \
  --region us-east-1 \
  --version 1.35 \
  --with-oidc \
  --managed \
  --node-type p4de.24xlarge \
  --nodes 1 \
  --nodes-min 0 \
  --nodes-max 2 \
  --node-ami-family AmazonLinux2023 \
  --install-nvidia-plugin
```

> This is suitable for quick experiments only. Use Terraform for anything beyond prototyping.

## Related Skills

- [Skill 02: GPU Instance Selection](gpu-instance-selection/SKILL.md)
- [Skill 03: AMI and Node Configuration](ami-and-node-configuration/SKILL.md)
- [Skill 04: Networking - EFA](networking-efa/SKILL.md)
- [Skill 05: Storage - FSx for Lustre](storage-fsx-lustre/SKILL.md)
