# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# -------------------------------------------------------------------
# Karpenter NodePool — GPU training nodes
# -------------------------------------------------------------------
locals {
  # GPU NodePools follow FSx (storage output fsx_az); target_az when it is unknown.
  gpu_az = var.fsx_az != "" ? var.fsx_az : var.target_az

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
            # Both labels are set: manifests/tests select role=gpu-training;
            # container-test.yaml selects node-role=gpu-training.
            "node-role" = "gpu-training"
            "role"      = "gpu-training"
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
            local.gpu_az != "" && !var.gpu_any_az ? [
              {
                key      = "topology.kubernetes.io/zone"
                operator = "In"
                values   = [local.gpu_az]
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
        # Put kubelet + containerd storage (image layers, emptyDir) on the NVMe instance
        # store, striped RAID0. GPU instances carry multi-TB NVMe (g6e.24xlarge 2x1900 GB,
        # g7e.24xlarge 2x3800 GB, p5en 8x3840 GB) that is otherwise unused, and the training
        # images are 15-35 GB: unpacking them on the 250 MB/s gp3 root is slow. EKS AI/ML
        # best practice. Types without instance store fall back to the EBS root.
        instanceStorePolicy = "RAID0"
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
      "nvidia.com/gpu" = tostring(var.gpu_limit + (var.enable_sim_pool ? var.sim_gpu_limit : 0))
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

# -------------------------------------------------------------------
# Optional Karpenter NodePool + EC2NodeClass for RT-core simulation GPUs
# (gpu-sim). Same AMI, IAM role, subnets and SG as gpu-training. No capacity
# reservation; nvidia.com/gpu taint only (GPU Operator DaemonSets tolerate only
# that taint). Pods select it with nodeSelector role=gpu-sim.
# -------------------------------------------------------------------
resource "kubernetes_manifest" "sim_nodepool" {
  count = var.enable_sim_pool ? 1 : 0
  manifest = {
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata = {
      name = "gpu-sim"
    }
    spec = {
      template = {
        metadata = {
          labels = {
            "node-role" = "gpu-sim"
            "role"      = "gpu-sim"
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
                values   = var.sim_capacity_type
              },
              {
                key      = "node.kubernetes.io/instance-type"
                operator = "In"
                values   = var.sim_instance_types
              },
            ],
            local.gpu_az != "" && !var.gpu_any_az ? [
              {
                key      = "topology.kubernetes.io/zone"
                operator = "In"
                values   = [local.gpu_az]
              },
            ] : []
          )
          nodeClassRef = {
            group = "karpenter.k8s.aws"
            kind  = "EC2NodeClass"
            name  = "gpu-sim"
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
        "nvidia.com/gpu" = var.sim_gpu_limit
      }
      disruption = {
        consolidationPolicy = "WhenEmpty"
        consolidateAfter    = "10m"
      }
    }
  }
}

resource "kubernetes_manifest" "sim_ec2nodeclass" {
  count = var.enable_sim_pool ? 1 : 0
  manifest = {
    apiVersion = "karpenter.k8s.aws/v1"
    kind       = "EC2NodeClass"
    metadata = {
      name = "gpu-sim"
    }
    spec = merge({
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
      # Put kubelet + containerd storage (image layers, emptyDir) on the NVMe instance
      # store, striped RAID0. GPU instances carry multi-TB NVMe (g6e.24xlarge 2x1900 GB,
      # g7e.24xlarge 2x3800 GB, p5en 8x3840 GB) that is otherwise unused, and the training
      # images are 15-35 GB: unpacking them on the 250 MB/s gp3 root is slow. EKS AI/ML
      # best practice. Types without instance store fall back to the EBS root.
      instanceStorePolicy = "RAID0"
      blockDeviceMappings = [
        {
          deviceName = "/dev/xvda"
          ebs = {
            volumeSize = "500Gi"
            volumeType = "gp3"
            # Same root volume as gpu-training: with RAID0 the image layers live on NVMe,
            # but types without instance store (and the OS) still use this volume.
            throughput = 500
            iops       = 6000
            encrypted  = true
          }
        }
      ]
      userData = base64encode(file("${path.module}/../cluster/scripts/gpu-node-bootstrap.sh"))
      },
      length(var.sim_capacity_reservation_selector) > 0 ? {
        capacityReservationSelectorTerms = var.sim_capacity_reservation_selector
      } : {}
    )
  }
}
