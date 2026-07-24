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

  backend "s3" {
    bucket       = "REPLACE-WITH-YOUR-TF-STATE-BUCKET"
    key          = "build/terraform.tfstate"
    region       = "us-west-2"
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
}
