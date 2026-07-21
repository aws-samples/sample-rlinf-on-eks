# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

output "ecr_repository_url" {
  description = "ECR repository URL for the training container image"
  value       = aws_ecr_repository.training.repository_url
}

output "codebuild_project_name" {
  description = "CodeBuild project name"
  value       = aws_codebuild_project.training.name
}

output "codebuild_source_bucket" {
  description = "S3 bucket for CodeBuild source archives"
  value       = aws_s3_bucket.codebuild_source.id
}

output "ecr_kms_key_arn" {
  description = "ARN of the customer-managed KMS key encrypting the ECR repository"
  value       = aws_kms_key.ecr.arn
}
