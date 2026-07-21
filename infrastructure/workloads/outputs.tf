# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

output "gpu_nodepool_name" {
  description = "Karpenter NodePool name for GPU training"
  value       = kubernetes_manifest.gpu_nodepool.manifest.metadata.name
}

output "gpu_ec2nodeclass_name" {
  description = "Karpenter EC2NodeClass name"
  value       = kubernetes_manifest.gpu_ec2nodeclass.manifest.metadata.name
}

output "karpenter_node_role_arn" {
  description = "IAM role ARN for Karpenter-provisioned GPU nodes"
  value       = aws_iam_role.karpenter_node.arn
}
