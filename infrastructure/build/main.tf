# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# Build Layer — ECR Repository + CodeBuild Project
#
# Standalone layer for building container images. Does not require cluster access.
#
# Usage:
#   1. Upload source:
#      zip -r rlinf-on-eks.zip examples/Dockerfile examples/buildspec.yml examples/scripts/
#      aws s3 cp rlinf-on-eks.zip s3://<codebuild_source_bucket>/rlinf-on-eks.zip
#
#   2. Trigger build:
#      aws codebuild start-build --project-name <codebuild_project_name>
# =============================================================================

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# --- S3 Bucket for Build Source ---

resource "aws_s3_bucket" "codebuild_source" {
  bucket        = "${var.cluster_name}-codebuild-source"
  force_destroy = true

  tags = {
    Project = var.cluster_name
  }
}

resource "aws_s3_bucket_versioning" "codebuild_source" {
  bucket = aws_s3_bucket.codebuild_source.id
  versioning_configuration {
    status = "Enabled"
  }
}

# --- ECR Repository ---

# Customer-managed KMS key for encrypting ECR image layers at rest (CKV_AWS_136).
resource "aws_kms_key" "ecr" {
  description             = "${var.cluster_name} ECR repository encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Project = var.cluster_name
  }
}

resource "aws_kms_alias" "ecr" {
  name          = "alias/${var.cluster_name}-ecr"
  target_key_id = aws_kms_key.ecr.key_id
}

resource "aws_ecr_repository" "training" {
  name                 = var.ecr_repository_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.ecr.arn
  }

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project = var.cluster_name
  }
}

resource "aws_ecr_lifecycle_policy" "training" {
  repository = aws_ecr_repository.training.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      }
    ]
  })
}

# --- CodeBuild IAM Role ---

data "aws_iam_policy_document" "codebuild_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["codebuild.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "codebuild" {
  name               = "${var.cluster_name}-codebuild"
  assume_role_policy = data.aws_iam_policy_document.codebuild_assume.json

  tags = {
    Project = var.cluster_name
  }
}

resource "aws_iam_policy" "codebuild" {
  name = "${var.cluster_name}-codebuild"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRAuth"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Sid    = "ECRPush"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = [aws_ecr_repository.training.arn]
      },
      {
        Sid    = "ECRKMS"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = [aws_kms_key.ecr.arn]
      },
      {
        Sid    = "ECRPullDLC"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "arn:aws:ecr:*:763104351884:repository/*"
      },
      {
        Sid    = "S3Source"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:GetBucketVersioning",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.codebuild_source.arn,
          "${aws_s3_bucket.codebuild_source.arn}/*"
        ]
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/codebuild/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "codebuild" {
  role       = aws_iam_role.codebuild.name
  policy_arn = aws_iam_policy.codebuild.arn
}

# --- CodeBuild Project ---

resource "aws_codebuild_project" "training" {
  name          = var.codebuild_project_name
  description   = "Build training container image for EKS deployment"
  build_timeout = 60 # minutes
  service_role  = aws_iam_role.codebuild.arn

  # checkov:skip=CKV_AWS_316: privileged_mode is REQUIRED. This project runs
  #   `docker build` (two-stage: upstream RLinf image + EFA overlay) inside the
  #   CodeBuild container. Docker-in-Docker on CodeBuild has no alternative to
  #   privileged mode. The project is not internet-exposed and the build runs
  #   from a controlled S3 source artifact under a least-privilege service role.

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type                = var.build_compute_type
    image                       = "aws/codebuild/standard:7.0"
    type                        = "LINUX_CONTAINER"
    privileged_mode             = true # Required for Docker builds
    image_pull_credentials_type = "CODEBUILD"

    environment_variable {
      name  = "ECR_REPO"
      value = aws_ecr_repository.training.name
    }
  }

  source {
    type      = "S3"
    location  = "${aws_s3_bucket.codebuild_source.bucket}/rlinf-on-eks.zip"
    buildspec = "examples/buildspec.yml"
  }

  cache {
    type = "NO_CACHE"
  }

  logs_config {
    cloudwatch_logs {
      group_name  = "/aws/codebuild/${var.cluster_name}"
      stream_name = "training"
    }
  }

  tags = {
    Project = var.cluster_name
  }
}
