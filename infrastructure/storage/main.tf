# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# FSx for Lustre Filesystem + S3 Data Repository Association
#
# Shared high-performance storage for datasets, checkpoints, and model weights.
# FSx is treated as ephemeral and tied to the training workload lifecycle.
# A single Data Repository Association links the entire filesystem to S3,
# so all data is automatically backed up and the filesystem can be destroyed
# and recreated without data loss.
#
# Architecture:
#   S3 bucket (durable)  <-->  FSx for Lustre (ephemeral, high-perf)
#     /                           /
#     ├── datasets/               ├── datasets/
#     ├── checkpoints/            ├── checkpoints/
#     ├── models/                 ├── models/
#     └── ...                     └── ...
# =============================================================================

# --- S3 Bucket for FSx Data Repository ---
# Durable backing store for all training data. FSx auto-exports writes here
# and auto-imports from here when the filesystem is (re)created.

resource "aws_s3_bucket" "fsx_data" {
  bucket        = "${var.cluster_name}-fsx-data"
  force_destroy = true

  tags = {
    Name    = "${var.cluster_name}-fsx-data"
    Project = var.cluster_name
  }
}

resource "aws_s3_bucket_versioning" "fsx_data" {
  bucket = aws_s3_bucket.fsx_data.id

  versioning_configuration {
    status = "Enabled"
  }
}

# --- FSx Security Group ---

resource "aws_security_group" "fsx" {
  name_prefix = "${var.cluster_name}-fsx-"
  vpc_id      = var.vpc_id

  # Lustre traffic from EKS worker nodes
  ingress {
    description     = "Lustre from nodes"
    from_port       = 988
    to_port         = 988
    protocol        = "tcp"
    security_groups = [var.node_security_group_id]
  }

  ingress {
    description     = "Lustre internode from nodes"
    from_port       = 1018
    to_port         = 1023
    protocol        = "tcp"
    security_groups = [var.node_security_group_id]
  }

  # Self-referencing rules for Lustre inter-server communication
  ingress {
    description = "Lustre self"
    from_port   = 988
    to_port     = 988
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "Lustre internode self"
    from_port   = 1018
    to_port     = 1023
    protocol    = "tcp"
    self        = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.cluster_name}-fsx"
    Project = var.cluster_name
  }
}

# --- FSx for Lustre Filesystem ---
# Placed in the same AZ as GPU and system nodes (target_subnet_ids).
# PERSISTENT_2 deployment type is required for Data Repository Associations.
#
# WARNING: Changing subnet_ids forces filesystem replacement.
# This is acceptable because all data is backed by S3 via DRA.

# Customer-managed KMS key for FSx Lustre at-rest encryption (CKV_AWS_190).
# FSx uses KMS grants on this key to encrypt the filesystem.
resource "aws_kms_key" "fsx" {
  description             = "${var.cluster_name} FSx for Lustre encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name    = "${var.cluster_name}-fsx"
    Project = var.cluster_name
  }
}

resource "aws_kms_alias" "fsx" {
  name          = "alias/${var.cluster_name}-fsx"
  target_key_id = aws_kms_key.fsx.key_id
}

resource "aws_fsx_lustre_file_system" "training" {
  storage_capacity            = var.fsx_storage_capacity
  storage_type                = "SSD"
  subnet_ids                  = [jsondecode(var.target_subnet_ids)[0]]
  security_group_ids          = [aws_security_group.fsx.id]
  deployment_type             = "PERSISTENT_2"
  per_unit_storage_throughput = var.fsx_throughput_per_unit
  data_compression_type       = "NONE"
  kms_key_id                  = aws_kms_key.fsx.arn

  tags = {
    Name    = "${var.cluster_name}-training"
    Project = var.cluster_name
  }
}

# --- Data Repository Association ---
# Single DRA linking the entire FSx filesystem to S3. All files written to
# FSx are auto-exported to S3, and all S3 objects are auto-imported to FSx.
# This makes FSx fully ephemeral: destroy and recreate without data loss.
#
# NOTE: auto_export and export data repository tasks can't be used simultaneously.
# We use auto_export for continuous backup during training.

resource "aws_fsx_data_repository_association" "training" {
  file_system_id       = aws_fsx_lustre_file_system.training.id
  data_repository_path = "s3://${aws_s3_bucket.fsx_data.id}"
  file_system_path     = "/"

  # Import metadata on creation so existing S3 data is immediately visible
  batch_import_meta_data_on_create = true

  s3 {
    auto_import_policy {
      events = ["NEW", "CHANGED", "DELETED"]
    }
    auto_export_policy {
      events = ["NEW", "CHANGED", "DELETED"]
    }
  }

  # DRA creation can take 15+ minutes for root-level associations
  timeouts {
    create = "30m"
    delete = "30m"
  }

  tags = {
    Name    = "${var.cluster_name}-dra"
    Project = var.cluster_name
  }
}

# --- Kubernetes PV + PVC for FSx ---
# Skipped when skip_kubernetes=true (e.g., OIDC-authenticated clusters).
# In that case, apply PV/PVC via kubectl using the template in outputs.

resource "kubernetes_persistent_volume" "fsx" {
  count = var.skip_kubernetes ? 0 : 1

  metadata {
    name = "fsx-pv-${var.cluster_name}"
  }

  spec {
    capacity = {
      storage = "${var.fsx_storage_capacity}Gi"
    }

    volume_mode                      = "Filesystem"
    access_modes                     = ["ReadWriteMany"]
    persistent_volume_reclaim_policy = "Retain"

    mount_options = ["flock"]

    persistent_volume_source {
      csi {
        driver        = "fsx.csi.aws.com"
        volume_handle = aws_fsx_lustre_file_system.training.id
        volume_attributes = {
          dnsname   = aws_fsx_lustre_file_system.training.dns_name
          mountname = aws_fsx_lustre_file_system.training.mount_name
        }
      }
    }
  }
}

resource "kubernetes_persistent_volume_claim" "fsx" {
  count = var.skip_kubernetes ? 0 : 1

  metadata {
    name      = "fsx-claim"
    namespace = var.namespace
  }

  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = ""

    resources {
      requests = {
        storage = "${var.fsx_storage_capacity}Gi"
      }
    }

    volume_name = kubernetes_persistent_volume.fsx[0].metadata[0].name
  }

  depends_on = [kubernetes_persistent_volume.fsx]
}
