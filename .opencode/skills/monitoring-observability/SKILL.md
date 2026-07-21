---
name: monitoring-observability
description: Monitor GPU utilization, training progress, and costs using DCGM, Prometheus, Grafana, MLflow, and CloudWatch
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 15: Monitoring and Observability

## Purpose

Monitor GPU utilization, training progress, infrastructure health, and costs for RL training workloads running on EKS. This covers metrics collection, dashboards, alerting, and cost tracking.

## Monitoring Stack

```
+-------------------------------------------------------------------+
|                       Monitoring Architecture                      |
|                                                                    |
|  GPU Nodes                                                         |
|  ┌─────────────────┐                                              |
|  │ DCGM Exporter   │──► Prometheus ──► Grafana Dashboards         |
|  │ (GPU metrics)   │                                              |
|  ├─────────────────┤                                              |
|  │ Node Exporter   │──► Prometheus ──► Grafana Dashboards         |
|  │ (CPU/mem/disk)  │                                              |
|  ├─────────────────┤                                              |
|  │ Training Pod    │──► MLflow (training metrics)                  |
|  │ (RL framework)  │──► Ray Dashboard (cluster state)             |
|  └─────────────────┘                                              |
|                                                                    |
|  CloudWatch                                                        |
|  ┌─────────────────┐                                              |
|  │ EKS metrics     │──► CloudWatch Dashboards + Alarms            |
|  │ FSx metrics     │                                              |
|  │ EC2 metrics     │                                              |
|  └─────────────────┘                                              |
+-------------------------------------------------------------------+
```

## Layer 1: GPU Metrics (DCGM Exporter)

### Install DCGM Exporter

DCGM Exporter is managed by the NVIDIA GPU Operator (see Skill 01 `helm.tf`). The GPU Operator Helm release sets `dcgmExporter.enabled = true`, which deploys DCGM exporter pods on all GPU nodes automatically.

> **Discovery #59**: The NVIDIA GPU Operator creates the DCGM exporter DaemonSet and Service (`nvidia-dcgm-exporter` on port 9400) but does **NOT** create a ServiceMonitor. Without a ServiceMonitor, Prometheus never discovers or scrapes GPU metrics. Our `infrastructure/addons/main.tf` creates the ServiceMonitor via `null_resource.dcgm_service_monitor` (using `kubectl apply` because `kubernetes_manifest` requires CRD discovery at plan time, which fails on fresh clusters). The ServiceMonitor targets the `gpu-operator` namespace with label selector `app: nvidia-dcgm-exporter` and scrapes the `gpu-metrics` port at 15s intervals.

For manual standalone installation (not needed when using GPU Operator):

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install dcgm-exporter nvidia/dcgm-exporter \
  --namespace monitoring \
  --create-namespace \
  --set tolerations[0].key=nvidia.com/gpu \
  --set tolerations[0].operator=Exists \
  --set tolerations[0].effect=NoSchedule \
  --set nodeSelector.role=gpu-training
```

### Key GPU Metrics

| Metric | Prometheus Name | Healthy Range |
|--------|----------------|---------------|
| GPU Utilization | `DCGM_FI_DEV_GPU_UTIL` | >70% during update phase |
| GPU Memory Used | `DCGM_FI_DEV_FB_USED` | 60-75 GB on 80GB GPUs |
| GPU Temperature | `DCGM_FI_DEV_GPU_TEMP` | <85C |
| Power Usage | `DCGM_FI_DEV_POWER_USAGE` | <TDP (700W for H100) |
| SM Clock | `DCGM_FI_DEV_SM_CLOCK` | Near max (2.1 GHz H100) |
| Memory Clock | `DCGM_FI_DEV_MEM_CLOCK` | Near max |
| NVLink Bandwidth | `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | >0 during training |
| GPU Errors | `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` | 0 (alert if >0) |

## Layer 2: Cluster Metrics (Prometheus + Grafana)

### Install kube-prometheus-stack

The kube-prometheus-stack is installed as a Helm release in Terraform (see Skill 01 `helm.tf`). For manual installation:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=your-password \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

### Access Grafana

```bash
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80
# Open http://localhost:3000
```

### Custom Dashboard: RL Training Overview

> **Discovery #60**: Grafana sidecar dashboard provisioning works via ConfigMaps. The kube-prometheus-stack deploys a Grafana sidecar container that watches for ConfigMaps with the label `grafana_dashboard: "1"`. Any ConfigMap with this label in the `monitoring` namespace is automatically imported as a Grafana dashboard. Create a `kubernetes_config_map` resource in `infrastructure/addons/main.tf` with the dashboard JSON and the `grafana_dashboard: "1"` label.

Create a Grafana dashboard with these panels:

**GPU Section:**
```promql
# Average GPU utilization across training pods
avg(DCGM_FI_DEV_GPU_UTIL{pod=~"training.*"})

# GPU memory usage per GPU
DCGM_FI_DEV_FB_USED{pod=~"training.*"}

# GPU temperature
DCGM_FI_DEV_GPU_TEMP{pod=~"training.*"}
```

**Node Section:**
```promql
# CPU utilization (for env rendering)
100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage
node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes

# Network throughput (EFA)
rate(node_network_transmit_bytes_total{device=~"eth.*"}[5m])
```

**Storage Section:**
```promql
# FSx IOPS (from CloudWatch via YACE exporter)
# Or check via: kubectl exec <pod> -- lctl get_param osc.*.stats
```

## Layer 3: Training Metrics (MLflow)

The RL framework logs training metrics to MLflow:

### Key Metrics to Monitor

| Metric | What It Shows | Healthy Trend |
|--------|-------------|---------------|
| `success_rate` | Task success rate during validation | Increasing over time |
| `train/rewards` | Average binary rewards from rollout | Increasing (approaching 1.0) |
| `actor/loss` | Policy gradient loss | Decreasing, then stabilizing |
| `actor/entropy` | Action distribution entropy | Slowly decreasing |
| `timing/generate` | Time for rollout generation | Stable |
| `timing/update` | Time for policy update | Stable |
| `timing/total` | Total step time | Stable |
| `accuracy` | Fraction of mixed-outcome groups | 0.1-0.9 (dynamic sampling filter) |

### MLflow Configuration (Optional)

MLflow is an optional addon (disabled by default). Enable it with `enable_mlflow = true` in `infrastructure/addons/terraform.tfvars`. When enabled, configure the tracking URI in your training pod:

```bash
# In training pod environment (only when MLflow addon is enabled)
export MLFLOW_TRACKING_URI="http://mlflow.rlinf.svc.cluster.local"

# Framework-specific logger configuration
# (exact syntax depends on the RL framework)
```

> **Example (RLinf):** The RLinf reference uses Hydra overrides: `trainer.logger=['console','mlflow']`, `trainer.experiment_name=RLinf`. See `examples/AGENTS.md` for details.

### MLflow Alerts

MLflow does not include a built-in alerting system. Use Grafana alerting rules against Prometheus metrics for equivalent coverage:
- **Training stall**: Alert when `DCGM_FI_DEV_GPU_UTIL` drops to 0 for >30 minutes
- **NaN loss**: Configure a Grafana alert on the `actor/loss` metric panel
- **Success rate drop**: Alert on sudden decrease in validation success rate via Grafana
- **GPU OOM**: Alert when pod restarts detected via `kube_pod_container_status_restarts_total`

## Layer 4: CloudWatch (AWS Infrastructure)

### EKS Container Insights

```bash
# Enable Container Insights
aws eks create-addon \
  --cluster-name <cluster-name> \
  --addon-name amazon-cloudwatch-observability \
  --resolve-conflicts OVERWRITE
```

### FSx for Lustre Metrics

Monitor in CloudWatch:
- `DataReadBytes` / `DataWriteBytes` -- I/O throughput
- `FreeDataStorageCapacity` -- Available space
- `MetadataOperations` -- Metadata IOPS

### CloudWatch Alarms

```bash
# Alarm: FSx running low on space
aws cloudwatch put-metric-alarm \
  --alarm-name fsx-low-space \
  --metric-name FreeDataStorageCapacity \
  --namespace AWS/FSx \
  --statistic Average \
  --period 300 \
  --threshold 500000000000 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:<ACCOUNT>:alerts

# Alarm: GPU node not reporting
aws cloudwatch put-metric-alarm \
  --alarm-name gpu-node-missing \
  --metric-name node_count \
  --namespace ContainerInsights \
  --dimensions Name=ClusterName,Value=<cluster-name> \
  --statistic Minimum \
  --period 300 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:<ACCOUNT>:alerts
```

## Layer 5: Ray Dashboard

Ray provides a built-in dashboard for cluster monitoring:

```bash
# Port-forward Ray dashboard
kubectl port-forward <training-pod-0> 8265:8265
# Open http://localhost:8265
```

The Ray dashboard shows:
- Worker status (alive, dead, pending)
- Resource utilization (CPU, GPU, memory per worker)
- Task execution timeline
- Object store usage

## Cost Monitoring

### AWS Cost Explorer Tags

Tag all resources for cost tracking:

```yaml
tags:
  Project: <cluster-name>
  Environment: training
  Experiment: <experiment-name>
```

### Cost Estimation

```bash
# Quick cost estimate for a training run
# p4de.24xlarge: ~$41/hr per node
# 2 nodes x 100 hours = ~$8,200
# FSx 4.8 TiB PERSISTENT_2 at 125 MB/s: ~$696/month
# S3 storage: ~$23/TB/month
```

### Cost Optimization Alerts

```bash
# Budget alert: warn when monthly spend exceeds threshold
aws budgets create-budget \
  --account-id <ACCOUNT_ID> \
  --budget '{
    "BudgetName": "<cluster-name>-monthly",
    "BudgetLimit": {"Amount": "10000", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST",
    "CostFilters": {"TagKeyValue": ["user:Project$<cluster-name>"]}
  }' \
  --notifications-with-subscribers '[{
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "team@example.com"}]
  }]'
```

## Troubleshooting with Observability

| Symptom | Where to Look | What to Check |
|---------|--------------|---------------|
| Training slow | MLflow timing metrics | Is rollout or update the bottleneck? |
| GPU underutilized | DCGM Exporter | Is CPU bottleneck (env rendering)? |
| OOM errors | DCGM memory metrics + pod logs | Which GPU ran out? Reduce batch size |
| Training stalled | MLflow heartbeat + Ray dashboard | Worker crashed? NCCL timeout? |
| Poor success rate | MLflow success_rate + entropy | Entropy collapsed? Temperature too low? |
| High costs | Cost Explorer | Scale down idle nodes. Use Capacity Reservations |
| FSx slow | CloudWatch FSx metrics | Increase throughput tier or filesystem size |

## Validation Checklist

- [ ] DCGM Exporter running on all GPU nodes
- [ ] Prometheus scraping GPU and node metrics
- [ ] Grafana dashboard shows GPU utilization, memory, temperature
- [ ] MLflow receiving training metrics (loss, success rate, timing)
- [ ] CloudWatch Container Insights enabled
- [ ] CloudWatch alarms set for FSx space and node availability
- [ ] Cost tracking tags applied to all resources
- [ ] Budget alerts configured

## Related Skills

- [Skill 10: Kubernetes Native Features](kubernetes-native-features/SKILL.md) - NVIDIA GPU Operator
- [Skill 14: Benchmarking](benchmarking/SKILL.md) - Performance baseline metrics
- [Skill 01: EKS Cluster Provisioning](eks-cluster-provisioning/SKILL.md) - Cluster-level monitoring setup
