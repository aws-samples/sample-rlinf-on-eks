# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# IAM Roles for EKS Pod Identity
#
# Uses EKS Pod Identity (recommended over IRSA) for least-privilege IAM roles:
#   - FSx CSI driver
#   - Mountpoint for S3 CSI driver
#   - EBS CSI driver
#   - Training pods (S3 access for data/checkpoints)
#   - MLflow tracking server (S3 artifact store)
#
# Pod Identity uses a simple service principal trust policy instead of
# cluster-specific OIDC providers. The namespace/SA binding is managed by
# aws_eks_pod_identity_association resources, not the trust policy.
# =============================================================================

locals {
  training_s3_bucket_name = var.training_s3_bucket_name != "" ? var.training_s3_bucket_name : "${var.cluster_name}-fsx-data"
}

# --- Shared Pod Identity trust policy ---

data "aws_iam_policy_document" "pod_identity_assume" {
  statement {
    actions = [
      "sts:AssumeRole",
      "sts:TagSession",
    ]
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["pods.eks.amazonaws.com"]
    }
  }
}

# --- FSx CSI Driver Role ---

resource "aws_iam_role" "fsx_csi" {
  name               = "${var.cluster_name}-fsx-csi"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_assume.json
}

resource "aws_iam_role_policy_attachment" "fsx_csi" {
  role       = aws_iam_role.fsx_csi.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonFSxFullAccess"
}

# --- EBS CSI Driver Role ---

resource "aws_iam_role" "ebs_csi" {
  name               = "${var.cluster_name}-ebs-csi"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_assume.json
}

resource "aws_iam_role_policy_attachment" "ebs_csi" {
  role       = aws_iam_role.ebs_csi.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

# --- Mountpoint for S3 CSI Driver Role ---

resource "aws_iam_role" "mountpoint_s3" {
  name               = "${var.cluster_name}-mountpoint-s3"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_assume.json
}

resource "aws_iam_policy" "mountpoint_s3" {
  name = "${var.cluster_name}-mountpoint-s3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "MountpointBucketAccess"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = ["arn:aws:s3:::${local.training_s3_bucket_name}"]
      },
      {
        Sid    = "MountpointObjectAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:AbortMultipartUpload",
          "s3:DeleteObject"
        ]
        Resource = ["arn:aws:s3:::${local.training_s3_bucket_name}/*"]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "mountpoint_s3" {
  role       = aws_iam_role.mountpoint_s3.name
  policy_arn = aws_iam_policy.mountpoint_s3.arn
}

# --- Training Pod Role (S3 access for data and checkpoints) ---

resource "aws_iam_role" "training" {
  name               = "${var.cluster_name}-training"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_assume.json
}

resource "aws_iam_policy" "training_s3" {
  name = "${var.cluster_name}-training-s3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "TrainingS3Access"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:aws:s3:::${local.training_s3_bucket_name}",
          "arn:aws:s3:::${local.training_s3_bucket_name}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "training_s3" {
  role       = aws_iam_role.training.name
  policy_arn = aws_iam_policy.training_s3.arn
}

# --- Pod Identity Association for Training SA ---
# CSI driver associations are handled inline on the EKS addons (see main.tf).
# The training SA is not an addon, so it needs an explicit association.

resource "aws_eks_pod_identity_association" "training" {
  cluster_name    = module.eks.cluster_name
  namespace       = "rlinf"
  service_account = "training-sa"
  role_arn        = aws_iam_role.training.arn

  depends_on = [module.eks, kubernetes_namespace.rlinf]
}

# --- Kubernetes Namespace for RLinf Training Workloads ---
# Dedicated namespace for all RLinf training workloads (Jobs, StatefulSets, PVCs).
# Keeps training resources isolated from system workloads in default namespace.

resource "kubernetes_namespace" "rlinf" {
  metadata {
    name = "rlinf"
    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "purpose"                      = "training"
    }
  }

  depends_on = [module.eks]
}

# --- Kubernetes Service Account for Training Pods ---
# With Pod Identity, the eks.amazonaws.com/role-arn annotation is not needed.
# The association is managed by aws_eks_pod_identity_association above.

resource "kubernetes_service_account" "training" {
  metadata {
    name      = "training-sa"
    namespace = "rlinf"
  }

  depends_on = [kubernetes_namespace.rlinf]
}

# --- MLflow Tracking Server Role (S3 access for artifact store) ---

resource "aws_iam_role" "mlflow" {
  name               = "${var.cluster_name}-mlflow"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_assume.json
}

resource "aws_iam_policy" "mlflow_s3" {
  name = "${var.cluster_name}-mlflow-s3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "MLflowArtifactBucketAccess"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = ["arn:aws:s3:::${local.training_s3_bucket_name}"]
      },
      {
        Sid    = "MLflowArtifactObjectAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:AbortMultipartUpload"
        ]
        Resource = ["arn:aws:s3:::${local.training_s3_bucket_name}/mlflow/*"]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "mlflow_s3" {
  role       = aws_iam_role.mlflow.name
  policy_arn = aws_iam_policy.mlflow_s3.arn
}

resource "aws_eks_pod_identity_association" "mlflow" {
  cluster_name    = module.eks.cluster_name
  namespace       = "rlinf"
  service_account = "mlflow-tracking"
  role_arn        = aws_iam_role.mlflow.arn

  depends_on = [module.eks]
}
