# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

output "fsx_filesystem_id" {
  description = "FSx for Lustre filesystem ID"
  value       = aws_fsx_lustre_file_system.training.id
}

output "fsx_mount_name" {
  description = "FSx for Lustre mount name"
  value       = aws_fsx_lustre_file_system.training.mount_name
}

output "fsx_dns_name" {
  description = "FSx for Lustre DNS name"
  value       = aws_fsx_lustre_file_system.training.dns_name
}

output "s3_bucket_name" {
  description = "S3 bucket name for training data"
  value       = aws_s3_bucket.fsx_data.id
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.fsx_data.arn
}

output "fsx_kms_key_arn" {
  description = "ARN of the customer-managed KMS key encrypting the FSx for Lustre filesystem"
  value       = aws_kms_key.fsx.arn
}
