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

variable "gpu_instance_types" {
  description = "List of GPU instance types Karpenter can provision. If empty, Karpenter selects from gpu_instance_families based on pod resource requests."
  type        = list(string)
  default     = []
}

variable "gpu_instance_families" {
  description = "Instance families Karpenter can select from when gpu_instance_types is empty. Broadening this list improves capacity availability. For training on p-class GPUs, add p4d/p5/p5en."
  type        = list(string)
  default     = ["g6", "g6e", "g5"]
}

variable "gpu_limit" {
  description = "Maximum total GPUs Karpenter can provision across all nodes (cost guard)"
  type        = number
  default     = 8
}

variable "target_az" {
  description = "Target Availability Zone for GPU nodes"
  type        = string
  default     = ""
}

variable "fsx_az" {
  description = "AZ of the FSx filesystem (storage output fsx_az, passed by deploy.sh). The GPU NodePools are pinned here; empty = target_az."
  type        = string
  default     = ""
}

variable "gpu_any_az" {
  description = "Let the GPU NodePools launch in any VPC AZ instead of fsx_az. Use only to find capacity before moving FSx to that AZ (TF_VAR_fsx_az); GPU nodes outside fsx_az read FSx cross-AZ."
  type        = bool
  default     = false
}

variable "capacity_type" {
  description = "Capacity types Karpenter can use: on-demand, spot, reserved. Use reserved for Capacity Blocks (requires capacity_reservation_selector)."
  type        = list(string)
  default     = ["on-demand"]
}

variable "capacity_reservation_selector" {
  description = "Capacity reservation selector for EC2NodeClass. Required when capacity_type includes reserved. Example: {id = \"cr-0123456789abcdef0\"} or {tags = {purpose = \"ml-training\"}}."
  type = list(object({
    id   = optional(string)
    tags = optional(map(string))
  }))
  default = []
}

variable "node_security_group_id" {
  description = "EKS node security group ID (for EC2NodeClass)"
  type        = string
}

variable "node_role_name" {
  description = "IAM role name for Karpenter-provisioned GPU nodes. Must match the name referenced by the addons layer's Karpenter iam:PassRole grant. Defaults to <cluster_name>-gpu-training when empty."
  type        = string
  default     = ""
}

variable "enable_resource_limits" {
  description = "Enable ResourceQuota and LimitRange for the training namespace"
  type        = bool
  default     = true
}

# -------------------------------------------------------------------
# Optional RT-core simulation pool (gpu-sim): Isaac Sim / IsaacLab etc.
# -------------------------------------------------------------------
variable "enable_sim_pool" {
  description = "Create the gpu-sim NodePool/EC2NodeClass for RT-core simulator GPUs (e.g. Isaac Sim 5.1, which does not support A100/H100/H200)"
  type        = bool
  default     = false
}

variable "sim_instance_types" {
  description = "Instance types for the gpu-sim pool (RT-core GPUs). g7e (RTX PRO 6000) is recommended, g6e (L40S) the fallback; add g5 where neither is offered. Several sizes widen the net for scarce on-demand capacity."
  type        = list(string)
  default     = ["g7e.24xlarge", "g7e.48xlarge", "g6e.24xlarge", "g6e.48xlarge"]
}

variable "sim_capacity_type" {
  description = "Capacity types for the gpu-sim pool. On-demand is safer: losing one node kills the whole Ray job."
  type        = list(string)
  default     = ["on-demand"]
}

variable "sim_gpu_limit" {
  description = "Maximum total GPUs Karpenter can provision in the gpu-sim pool (cost guard)"
  type        = number
  default     = 16
}
