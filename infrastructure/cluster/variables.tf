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
  default     = "rlinf-on-eks"
}

variable "cluster_version" {
  description = "EKS Kubernetes version"
  type        = string
  default     = "1.35"
}

variable "target_az" {
  description = "Target Availability Zone for ALL workload resources (system nodes, GPU nodes via Karpenter, FSx). Set this to the AZ of your Capacity Block reservation or where your desired instance type has availability. Ensures co-locality to avoid cross-AZ latency and data transfer costs. Defaults to the first AZ if empty."
  type        = string
  default     = ""
}

variable "training_s3_bucket_name" {
  description = "S3 bucket name for training data and checkpoints. Used in IAM policies. Defaults to <cluster_name>-fsx-data if empty."
  type        = string
  default     = ""
}
