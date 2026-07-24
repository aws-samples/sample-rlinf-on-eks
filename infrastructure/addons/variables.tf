# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  type        = string
}

variable "cluster_certificate_authority_data" {
  description = "EKS cluster CA certificate (base64)"
  type        = string
  sensitive   = true
}

variable "node_security_group_id" {
  description = "EKS node security group ID (for Karpenter discovery)"
  type        = string
}

variable "node_role_name" {
  description = "IAM role name for Karpenter-provisioned GPU nodes (created in the workloads layer). The Karpenter controller's iam:PassRole grant is scoped to this role. Must match the workloads layer's node_role_name. Defaults to <cluster_name>-gpu-training when empty."
  type        = string
  default     = ""
}

variable "enable_kuberay" {
  description = "Deploy KubeRay operator (required for DreamZero SFT RayJob)"
  type        = bool
  default     = true
}

variable "enable_mpi_operator" {
  description = "Deploy MPI Operator (required for NCCL bandwidth tests)"
  type        = bool
  default     = false
}

variable "enable_mlflow" {
  description = "Deploy MLflow tracking server"
  type        = bool
  default     = false
}

variable "enable_monitoring" {
  description = "Deploy kube-prometheus-stack (Prometheus + Grafana)"
  type        = bool
  default     = false
}

variable "grafana_password" {
  description = "Grafana admin password (required only when enable_monitoring = true). Set via TF_VAR_grafana_password or -var; do not commit a real value."
  type        = string
  default     = ""
  sensitive   = true
}

variable "mlflow_s3_bucket" {
  description = "S3 bucket for MLflow artifacts"
  type        = string
  default     = ""
}

variable "mlflow_iam_role_arn" {
  description = "IAM role ARN for MLflow service account"
  type        = string
  default     = ""
}
