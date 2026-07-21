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

variable "vpc_id" {
  description = "VPC ID for FSx security group rules"
  type        = string
}

variable "node_security_group_id" {
  description = "EKS node security group ID for FSx ingress"
  type        = string
}

variable "target_subnet_ids" {
  description = "JSON-encoded list of subnet IDs in the target AZ"
  type        = string
}

variable "fsx_storage_capacity" {
  description = "FSx for Lustre storage capacity in GiB (minimum 1200)"
  type        = number
  default     = 2400
}

variable "fsx_throughput_per_unit" {
  description = "FSx for Lustre throughput per unit of storage (MB/s/TiB)"
  type        = number
  default     = 500
}

variable "namespace" {
  description = "Kubernetes namespace for the PVC"
  type        = string
  default     = "rlinf"
}

variable "skip_kubernetes" {
  description = "Skip Kubernetes PV/PVC creation (use when cluster uses non-IAM auth, e.g., OIDC)"
  type        = bool
  default     = false
}
