# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Partial backend config: bucket + region are supplied at init time via
  # -backend-config=infrastructure/backend.hcl (see backend.hcl.example).
  # deploy.sh passes this automatically. Keeps account-specific values out of
  # version control.
  backend "s3" {
    key          = "build/terraform.tfstate"
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
}
