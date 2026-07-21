# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

output "karpenter_installed" {
  description = "Whether Karpenter was installed"
  value       = true
}

output "gpu_operator_namespace" {
  description = "Namespace where GPU Operator is installed"
  value       = helm_release.gpu_operator.namespace
}
