# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# -------------------------------------------------------------------
# Karpenter NodePool — GPU training nodes
# -------------------------------------------------------------------
locals {
  # Single source of truth for the GPU node IAM role name. The addons layer's
  # Karpenter iam:PassRole grant must target this exact name; both layers default
  # to the same expression so they stay in sync without a cross-layer reference.
  node_role_name = var.node_role_name != "" ? var.node_role_name : "${var.cluster_name}-gpu-training"
}

resource "kubernetes_manifest" "gpu_nodepool" {
  manifest = {
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata = {
      name = "gpu-training"
    }
    spec = {
      template = {
        metadata = {
          labels = {
            "node-role" = "gpu-training"
          }
        }
        spec = {
          requirements = concat(
            [
              {
                key      = "kubernetes.io/arch"
                operator = "In"
                values   = ["amd64"]
              },
              {
                key      = "karpenter.sh/capacity-type"
                operator = "In"
                values   = var.capacity_type
              },
            ],
            # Instance selection: explicit types take precedence over families.
            # Families give Karpenter flexibility to find available capacity.
            length(var.gpu_instance_types) > 0 ? [
              {
                key      = "node.kubernetes.io/instance-type"
                operator = "In"
                values   = var.gpu_instance_types
              },
              ] : [
              {
                key      = "karpenter.k8s.aws/instance-family"
                operator = "In"
                values   = var.gpu_instance_families
              },
              {
                key      = "karpenter.k8s.aws/instance-gpu-count"
                operator = "Gt"
                values   = ["0"]
              },
            ],
            var.target_az != "" ? [
              {
                key      = "topology.kubernetes.io/zone"
                operator = "In"
                values   = [var.target_az]
              },
            ] : []
          )
          nodeClassRef = {
            group = "karpenter.k8s.aws"
            kind  = "EC2NodeClass"
            name  = "gpu-training"
          }
          taints = [
            {
              key    = "nvidia.com/gpu"
              value  = "true"
              effect = "NoSchedule"
            }
          ]
        }
      }
      limits = {
        "nvidia.com/gpu" = var.gpu_limit
      }
      disruption = {
        consolidationPolicy = "WhenEmpty"
        consolidateAfter    = "60s"
      }
    }
  }
}

# -------------------------------------------------------------------
# Karpenter EC2NodeClass — GPU node configuration
# -------------------------------------------------------------------
resource "kubernetes_manifest" "gpu_ec2nodeclass" {
  manifest = {
    apiVersion = "karpenter.k8s.aws/v1"
    kind       = "EC2NodeClass"
    metadata = {
      name = "gpu-training"
    }
    spec = merge(
      {
        amiSelectorTerms = [
          {
            alias = "al2023@latest"
          }
        ]
        role = local.node_role_name
        subnetSelectorTerms = [
          {
            tags = {
              "kubernetes.io/cluster/${var.cluster_name}" = "shared"
            }
          }
        ]
        securityGroupSelectorTerms = [
          {
            id = var.node_security_group_id
          }
        ]
        blockDeviceMappings = [
          {
            deviceName = "/dev/xvda"
            ebs = {
              volumeSize = "500Gi"
              volumeType = "gp3"
              throughput = 500
              iops       = 6000
              encrypted  = true
            }
          }
        ]
        userData = base64encode(file("${path.module}/../cluster/scripts/gpu-node-bootstrap.sh"))
      },
      length(var.capacity_reservation_selector) > 0 ? {
        capacityReservationSelectorTerms = var.capacity_reservation_selector
      } : {}
    )
  }
}

# -------------------------------------------------------------------
# Karpenter IAM Role for GPU nodes
# -------------------------------------------------------------------
data "aws_iam_policy_document" "karpenter_node_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "karpenter_node" {
  name               = local.node_role_name
  assume_role_policy = data.aws_iam_policy_document.karpenter_node_assume.json
}

resource "aws_iam_role_policy_attachment" "karpenter_node_worker" {
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "karpenter_node_cni" {
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "karpenter_node_ecr" {
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "karpenter_node_ssm" {
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "karpenter_node" {
  name = local.node_role_name
  role = aws_iam_role.karpenter_node.name
}

# -------------------------------------------------------------------
# EKS access entry for Karpenter nodes
# -------------------------------------------------------------------
resource "aws_eks_access_entry" "karpenter_node" {
  cluster_name  = var.cluster_name
  principal_arn = aws_iam_role.karpenter_node.arn
  type          = "EC2_LINUX"
}

# -------------------------------------------------------------------
# ResourceQuota (agent guardrails)
# -------------------------------------------------------------------
resource "kubernetes_resource_quota" "rlinf_gpu" {
  count = var.enable_resource_limits ? 1 : 0

  metadata {
    name      = "gpu-quota"
    namespace = "rlinf"
  }

  spec {
    hard = {
      "nvidia.com/gpu" = tostring(var.gpu_limit)
    }
  }
}

# -------------------------------------------------------------------
# LimitRange (prevent runaway resource requests)
# -------------------------------------------------------------------
resource "kubernetes_limit_range" "rlinf" {
  count = var.enable_resource_limits ? 1 : 0

  metadata {
    name      = "default-limits"
    namespace = "rlinf"
  }

  spec {
    limit {
      type = "Container"
      default_request = {
        cpu    = "1"
        memory = "4Gi"
      }
    }
  }
}
