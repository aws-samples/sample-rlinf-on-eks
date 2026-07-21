# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# RLinf-on-EKS: Cluster Layer
#
# This layer provisions the foundational AWS infrastructure:
#   - VPC with public/private subnets
#   - EKS cluster with OIDC
#   - System node group (cluster services)
#   - EFA security group rules (for GPU nodes managed by Karpenter)
#   - EKS managed add-ons (VPC CNI, CoreDNS, EBS CSI, FSx CSI, Mountpoint S3)
#
# GPU training nodes are managed by Karpenter (see addons layer).
# See variables.tf for configuration. Override with profiles/*.tfvars.
# =============================================================================

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

locals {
  # ALL workload resources (GPU nodes, system nodes, FSx) MUST be in the same AZ
  # to avoid cross-AZ latency and data transfer costs.
  # Use the explicitly set target_az or default to the first AZ.
  effective_az = var.target_az != "" ? var.target_az : data.aws_availability_zones.available.names[0]

  # VPC AZs: always include the target AZ plus one additional for HA (NAT, ALB).
  # If target_az is already in the first 2 AZs, this is a no-op.
  default_azs = slice(data.aws_availability_zones.available.names, 0, 2)
  vpc_azs = contains(local.default_azs, local.effective_az) ? local.default_azs : [
    local.effective_az,
    [for az in data.aws_availability_zones.available.names : az if az != local.effective_az][0]
  ]

  # Single subnet for all workload resources in the target AZ
  target_subnet_ids = [
    for idx, subnet_id in module.vpc.private_subnets :
    subnet_id if local.vpc_azs[idx] == local.effective_az
  ]
}

# -----------------------------------------------------------------------------
# VPC
# -----------------------------------------------------------------------------

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = local.vpc_azs
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
    "kubernetes.io/role/internal-elb"           = 1
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }

  tags = {
    Project     = var.cluster_name
    Environment = "training"
  }
}

# -----------------------------------------------------------------------------
# EFA Security Group Rules
# The EKS module's enable_efa_support (used by Karpenter NodeClass) creates EFA
# interfaces but the auto-generated node SG rules only allow TCP 1025-65535
# between nodes. EFA RDMA requires ALL protocols (including custom EFA protocol,
# UDP, etc.) between nodes in the same security group.
# -----------------------------------------------------------------------------

resource "aws_security_group_rule" "efa_all_traffic" {
  type                     = "ingress"
  from_port                = 0
  to_port                  = 0
  protocol                 = "-1"
  source_security_group_id = module.eks.node_security_group_id
  security_group_id        = module.eks.node_security_group_id
  description              = "EFA: Allow all inbound traffic between GPU nodes for NCCL/RDMA"
}

resource "aws_security_group_rule" "efa_all_traffic_egress" {
  type                     = "egress"
  from_port                = 0
  to_port                  = 0
  protocol                 = "-1"
  source_security_group_id = module.eks.node_security_group_id
  security_group_id        = module.eks.node_security_group_id
  description              = "EFA: Allow all outbound traffic between GPU nodes for NCCL/RDMA"
}

# -----------------------------------------------------------------------------
# EKS Cluster
# -----------------------------------------------------------------------------

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
  # Only system nodes here. GPU training nodes are managed by Karpenter (addons layer).
  eks_managed_node_groups = {
    # System nodes for cluster services (CoreDNS, monitoring, etc.)
    # Pinned to the same AZ as GPU nodes and FSx for co-locality.
    system = {
      instance_types = ["m5.2xlarge"]
      min_size       = 2
      desired_size   = 2
      max_size       = 4

      subnet_ids = local.target_subnet_ids

      labels = { role = "system" }

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs         = { volume_size = 100, volume_type = "gp3" }
        }
      }
    }
  }

  tags = {
    Project     = var.cluster_name
    Environment = "training"
  }
}
