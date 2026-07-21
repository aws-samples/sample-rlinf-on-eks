# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "Cluster name (used for resource naming)"
  type        = string
  default     = "rlinf-on-eks"
}

variable "ecr_repository_name" {
  description = "ECR repository name for the training container image"
  type        = string
  default     = "rlinf-on-eks"
}

variable "codebuild_project_name" {
  description = "CodeBuild project name for building the training container"
  type        = string
  default     = "rlinf-on-eks"
}

variable "build_compute_type" {
  description = "CodeBuild compute type for container builds"
  type        = string
  default     = "BUILD_GENERAL1_2XLARGE"
}
