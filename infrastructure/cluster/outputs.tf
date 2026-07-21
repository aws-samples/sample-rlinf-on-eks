# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "region" {
  description = "AWS region"
  value       = var.region
}

output "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority_data" {
  description = "EKS cluster CA certificate (base64)"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "node_security_group_id" {
  description = "EKS node security group ID (for FSx and Karpenter SG rules)"
  value       = module.eks.node_security_group_id
}

output "target_az" {
  description = "Target Availability Zone for workload resources"
  value       = local.effective_az
}

output "target_subnet_ids" {
  description = "Subnet IDs in the target AZ (JSON-encoded list)"
  value       = jsonencode(local.target_subnet_ids)
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.region}"
}

output "training_iam_role_arn" {
  description = "Training pod IAM role ARN"
  value       = aws_iam_role.training.arn
}

output "mlflow_iam_role_arn" {
  description = "MLflow tracking server IAM role ARN"
  value       = aws_iam_role.mlflow.arn
}

output "mlflow_s3_bucket" {
  description = "S3 bucket for MLflow artifacts"
  value       = local.training_s3_bucket_name
}
