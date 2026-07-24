# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# Addons Layer — Helm Releases for EKS Cluster
#
# Installs Karpenter, NVIDIA GPU Operator, monitoring stack, MPI Operator,
# and optional KubeRay/MLflow. Applied after cluster layer is fully ready.
# =============================================================================

# --- ECR Public auth token (for Karpenter OCI chart) ---
# Must query us-east-1 regardless of deployment region (ECR Public is us-east-1 only)
provider "aws" {
  alias  = "virginia"
  region = "us-east-1"
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

data "aws_ecrpublic_authorization_token" "token" {
  provider = aws.virginia
}

# --- Configure kubectl for null_resource provisioners ---
resource "null_resource" "update_kubeconfig" {
  provisioner "local-exec" {
    command = "aws eks update-kubeconfig --name ${var.cluster_name} --region ${var.region}"
  }
}

# --- Karpenter ---
# Node autoscaler replacing EKS managed GPU node groups.
# Provisions GPU nodes on-demand based on pending pod requests.

# Karpenter Controller IAM Role
data "aws_iam_policy_document" "karpenter_assume" {
  statement {
    actions = ["sts:AssumeRole", "sts:TagSession"]
    principals {
      type        = "Service"
      identifiers = ["pods.eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "karpenter" {
  name               = "${var.cluster_name}-karpenter"
  assume_role_policy = data.aws_iam_policy_document.karpenter_assume.json
}

# Karpenter node role + instance profile are created in the workloads layer.
# Their name is centralized via the node_role_name variable (shared default with
# the workloads layer), so this iam:PassRole grant stays in sync without a
# cross-layer data source (workloads is downstream of addons).
locals {
  node_role_name                  = var.node_role_name != "" ? var.node_role_name : "${var.cluster_name}-gpu-training"
  karpenter_node_role_arn         = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:role/${local.node_role_name}"
  karpenter_instance_profile_arns = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:instance-profile/*"
}

resource "aws_iam_policy" "karpenter" {
  name = "${var.cluster_name}-karpenter"

  # checkov:skip=CKV_AWS_355: The KarpenterReadOnly and KarpenterEC2Provisioning
  #   statements require Resource="*". The EC2 Describe*, pricing:GetProducts, and
  #   ssm:GetParameter actions do not support resource-level permissions. The
  #   ec2:RunInstances/CreateFleet/CreateLaunchTemplate/CreateTags/DeleteLaunchTemplate
  #   actions target resources Karpenter provisions dynamically, whose ARNs are not
  #   known at policy-creation time, so this statement uses Resource="*" WITHOUT a
  #   tag condition (only the separate ConditionalEC2Termination statement is
  #   tag-scoped to karpenter.sh/nodepool). This mirrors the AWS-published Karpenter
  #   controller policy; a tighter per-resource-type / aws:RequestTag+ec2:CreateAction
  #   variant exists upstream but requires live-cluster provisioning validation to
  #   adopt safely. The genuine privilege-escalation vectors (iam:PassRole and the
  #   instance-profile actions) ARE scoped below.
  # checkov:skip=CKV_AWS_286: iam:PassRole is scoped to the Karpenter node role ARN
  #   with an iam:PassedToService=ec2.amazonaws.com condition, and instance-profile
  #   mutations are scoped to this account's instance profiles. No wildcard PassRole.
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "KarpenterReadOnly"
        Effect = "Allow"
        Action = [
          "ec2:DescribeAvailabilityZones",
          "ec2:DescribeImages",
          "ec2:DescribeInstances",
          "ec2:DescribeInstanceTypeOfferings",
          "ec2:DescribeInstanceTypes",
          "ec2:DescribeLaunchTemplates",
          "ec2:DescribeSecurityGroups",
          "ec2:DescribeSpotPriceHistory",
          "ec2:DescribeSubnets",
          "eks:DescribeCluster",
          "pricing:GetProducts",
          "ssm:GetParameter",
        ]
        Resource = "*"
      },
      {
        Sid    = "KarpenterEC2Provisioning"
        Effect = "Allow"
        Action = [
          "ec2:CreateFleet",
          "ec2:CreateLaunchTemplate",
          "ec2:CreateTags",
          "ec2:DeleteLaunchTemplate",
          "ec2:RunInstances",
        ]
        Resource = "*"
      },
      {
        Sid      = "ConditionalEC2Termination"
        Effect   = "Allow"
        Action   = "ec2:TerminateInstances"
        Resource = "*"
        Condition = {
          StringLike = {
            "ec2:ResourceTag/karpenter.sh/nodepool" = "*"
          }
        }
      },
      {
        Sid    = "KarpenterInstanceProfile"
        Effect = "Allow"
        Action = [
          "iam:AddRoleToInstanceProfile",
          "iam:CreateInstanceProfile",
          "iam:DeleteInstanceProfile",
          "iam:GetInstanceProfile",
          "iam:RemoveRoleFromInstanceProfile",
          "iam:TagInstanceProfile",
        ]
        Resource = local.karpenter_instance_profile_arns
      },
      {
        Sid      = "KarpenterPassNodeRole"
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = local.karpenter_node_role_arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "ec2.amazonaws.com"
          }
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "karpenter" {
  role       = aws_iam_role.karpenter.name
  policy_arn = aws_iam_policy.karpenter.arn
}

resource "aws_eks_pod_identity_association" "karpenter" {
  cluster_name    = var.cluster_name
  namespace       = "kube-system"
  service_account = "karpenter"
  role_arn        = aws_iam_role.karpenter.arn
}

resource "helm_release" "karpenter" {
  name       = "karpenter"
  namespace  = "kube-system"
  repository = "oci://public.ecr.aws/karpenter"
  chart      = "karpenter"
  version    = "1.11.1"
  wait       = true
  atomic     = true

  depends_on = [aws_eks_pod_identity_association.karpenter]

  set {
    name  = "settings.clusterName"
    value = var.cluster_name
  }

  set {
    name  = "settings.clusterEndpoint"
    value = var.cluster_endpoint
  }

  set {
    name  = "controller.resources.requests.cpu"
    value = "1"
  }

  set {
    name  = "controller.resources.requests.memory"
    value = "1Gi"
  }

  set {
    name  = "controller.resources.limits.memory"
    value = "1Gi"
  }
}

# --- NVIDIA GPU Operator ---
# Replaces standalone device plugin + DCGM exporter with a unified operator.
# Manages: device plugin, DCGM exporter, GPU Feature Discovery, container toolkit.
# Driver installation is disabled because EKS GPU AMI already includes NVIDIA drivers.
resource "helm_release" "gpu_operator" {
  name             = "gpu-operator"
  repository       = "https://helm.ngc.nvidia.com/nvidia"
  chart            = "gpu-operator"
  namespace        = "gpu-operator"
  create_namespace = true
  version          = "v24.9.2"

  # Disable driver install -- EKS GPU AMI already has NVIDIA drivers
  set {
    name  = "driver.enabled"
    value = "false"
  }

  # Enable container toolkit (required for GPU access in containers)
  set {
    name  = "toolkit.enabled"
    value = "true"
  }

  # Enable DCGM exporter (GPU metrics for Prometheus)
  set {
    name  = "dcgmExporter.enabled"
    value = "true"
  }

  # Enable GPU Feature Discovery (auto-labels nodes with GPU model, memory, etc.)
  set {
    name  = "gfd.enabled"
    value = "true"
  }

  # Tolerate GPU node taints so operator components schedule on GPU nodes
  set {
    name  = "daemonsets.tolerations[0].key"
    value = "nvidia.com/gpu"
  }
  set {
    name  = "daemonsets.tolerations[0].operator"
    value = "Exists"
  }
  set {
    name  = "daemonsets.tolerations[0].effect"
    value = "NoSchedule"
  }
}

# --- EFA Device Plugin ---
resource "helm_release" "efa_device_plugin" {
  name       = "efa-device-plugin"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-efa-k8s-device-plugin"
  namespace  = "kube-system"
  version    = "0.5.25"

  set {
    name  = "tolerations[0].key"
    value = "nvidia.com/gpu"
  }
  set {
    name  = "tolerations[0].operator"
    value = "Exists"
  }
  set {
    name  = "tolerations[0].effect"
    value = "NoSchedule"
  }
}

# --- kube-prometheus-stack (Prometheus + Grafana) ---
resource "helm_release" "prometheus" {
  count = var.enable_monitoring ? 1 : 0

  name             = "monitoring"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  namespace        = "monitoring"
  create_namespace = true
  version          = "82.18.0"

  set {
    name  = "grafana.adminPassword"
    value = var.grafana_password
  }
  set {
    name  = "prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues"
    value = "false"
  }

  lifecycle {
    precondition {
      condition     = var.grafana_password != ""
      error_message = "grafana_password must be set when enable_monitoring = true (e.g. export TF_VAR_grafana_password=... or pass -var grafana_password=...). Do not commit a real value."
    }
  }
}

# --- DCGM Exporter ServiceMonitor ---
# The NVIDIA GPU Operator creates the DCGM exporter DaemonSet and Service
# (nvidia-dcgm-exporter on port 9400) but does NOT create a ServiceMonitor.
# Without this, Prometheus never discovers or scrapes GPU metrics.
# We apply via kubectl (not kubernetes_manifest) because kubernetes_manifest
# requires CRD discovery at plan time, which fails on fresh clusters.
resource "null_resource" "dcgm_service_monitor" {
  count = var.enable_monitoring ? 1 : 0

  provisioner "local-exec" {
    command = <<-EOT
      kubectl apply -f - <<'EOF'
      apiVersion: monitoring.coreos.com/v1
      kind: ServiceMonitor
      metadata:
        name: nvidia-dcgm-exporter
        namespace: gpu-operator
        labels:
          app: nvidia-dcgm-exporter
      spec:
        selector:
          matchLabels:
            app: nvidia-dcgm-exporter
        namespaceSelector:
          matchNames:
            - gpu-operator
        endpoints:
          - port: gpu-metrics
            path: /metrics
            interval: "15s"
      EOF
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = "kubectl delete servicemonitor nvidia-dcgm-exporter -n gpu-operator --ignore-not-found"
  }

  depends_on = [helm_release.gpu_operator, helm_release.prometheus, null_resource.update_kubeconfig]
}

# --- KubeRay Operator (Optional) ---
resource "helm_release" "kuberay_operator" {
  count = var.enable_kuberay ? 1 : 0

  name             = "kuberay-operator"
  repository       = "https://ray-project.github.io/kuberay-helm/"
  chart            = "kuberay-operator"
  namespace        = "ray-system"
  create_namespace = true
  version          = "1.6.0"
}

# --- MLflow Tracking Server ---
# Self-hosted experiment tracking with SQLite backend on FSx and S3 artifact store.
# Training pods log metrics via MLFLOW_TRACKING_URI=http://mlflow.rlinf.svc.cluster.local
# Access UI via: kubectl port-forward svc/mlflow -n rlinf 5000:80
resource "helm_release" "mlflow" {
  count = var.enable_mlflow ? 1 : 0

  name             = "mlflow"
  repository       = "https://community-charts.github.io/helm-charts"
  chart            = "mlflow"
  version          = "1.8.1"
  namespace        = "rlinf"
  create_namespace = false

  # Pin to MLflow 3.1 (3.2+ added security middleware that rejects in-cluster
  # Host headers and is incompatible with the chart's gunicorn config)
  set {
    name  = "image.repository"
    value = "ghcr.io/mlflow/mlflow"
  }
  set {
    name  = "image.tag"
    value = "v3.1.0"
  }

  # SQLite backend store on FSx (persists across pod restarts)
  set {
    name  = "backendStore.databaseMigration"
    value = "true"
  }
  set {
    name  = "backendStore.postgres.enabled"
    value = "false"
  }
  set {
    name  = "backendStore.defaultSqlitePath"
    value = "/fsx/mlflow/mlflow.db"
  }

  # Artifacts on FSx
  set {
    name  = "extraArgs.defaultArtifactRoot"
    value = "/fsx/mlflow/artifacts"
  }
  set {
    name  = "extraEnvVars.AWS_DEFAULT_REGION"
    value = var.region
  }

  # Mount FSx PVC
  set {
    name  = "extraVolumes[0].name"
    value = "fsx"
  }
  set {
    name  = "extraVolumes[0].persistentVolumeClaim.claimName"
    value = "fsx-claim"
  }
  set {
    name  = "extraVolumeMounts[0].name"
    value = "fsx"
  }
  set {
    name  = "extraVolumeMounts[0].mountPath"
    value = "/fsx"
  }

  # Init container to create /fsx/mlflow directories on first deploy
  values = [yamlencode({
    initContainers = [{
      name    = "init-fsx-dirs"
      image   = "busybox:1.36"
      command = ["sh", "-c", "mkdir -p /fsx/mlflow/artifacts && chmod 777 /fsx/mlflow /fsx/mlflow/artifacts"]
      volumeMounts = [{
        name      = "fsx"
        mountPath = "/fsx"
      }]
    }]
  })]

  # Prometheus ServiceMonitor disabled -- ghcr.io/mlflow/mlflow:v3.1.0 lacks
  # prometheus_flask_exporter. MLflow experiment metrics are viewed via the MLflow UI;
  # Prometheus monitors infrastructure (GPU/node/K8s) not the MLflow server process.
  set {
    name  = "serviceMonitor.enabled"
    value = "false"
  }

  # Service account for EKS Pod Identity (S3 artifact access)
  set {
    name  = "serviceAccount.create"
    value = "true"
  }
  set {
    name  = "serviceAccount.name"
    value = "mlflow-tracking"
  }
}

# --- Kubeflow MPI Operator ---
# Required for MPIJob CRD used by NCCL tests and MPI-based distributed training.
# NOTE: The Helm chart repo returns 404. Install from GitHub release manifest.
resource "null_resource" "mpi_operator" {
  count = var.enable_mpi_operator ? 1 : 0

  provisioner "local-exec" {
    command = "kubectl apply --server-side --force-conflicts -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.5.0/deploy/v2beta1/mpi-operator.yaml"
  }

  provisioner "local-exec" {
    when    = destroy
    command = "kubectl delete -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.5.0/deploy/v2beta1/mpi-operator.yaml --ignore-not-found"
  }

  depends_on = [null_resource.update_kubeconfig]
}
