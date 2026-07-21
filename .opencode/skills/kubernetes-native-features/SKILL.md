---
name: kubernetes-native-features
description: Leverage Kubernetes-native features for GPU training including GPU Operator, topology scheduling, and priority classes
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 10: Kubernetes Native Features

## Purpose

Leverage Kubernetes-native features to optimize GPU training workloads: device plugins, topology-aware scheduling, priority classes, resource quotas, and pod disruption budgets.

## NVIDIA GPU Operator

The NVIDIA GPU Operator is the primary method for managing the GPU software stack on EKS. It is deployed as a Helm release in Terraform (see Skill 01 `helm.tf`) and manages:
- **Device plugin** -- exposes `nvidia.com/gpu` as a schedulable resource
- **DCGM exporter** -- GPU metrics for Prometheus (see Skill 15)
- **GPU Feature Discovery** -- auto-labels nodes with GPU model, memory, CUDA version, etc.
- **Container toolkit** -- GPU access in containers

```bash
# Verify GPU Operator pods
kubectl get pods -n gpu-operator

# Check GPU Feature Discovery labels
kubectl get nodes -l role=gpu-training -o json | \
  jq '.items[0].metadata.labels | to_entries | map(select(.key | startswith("nvidia"))) | from_entries'
```

Configuration in Terraform:

```hcl
resource "helm_release" "gpu_operator" {
  name    = "gpu-operator"
  chart   = "gpu-operator"
  version = "v24.9.2"

  set { name = "driver.enabled";       value = "false" }  # AMI has drivers
  set { name = "toolkit.enabled";      value = "true" }
  set { name = "dcgmExporter.enabled"; value = "true" }
  set { name = "gfd.enabled";          value = "true" }
}
```

> **Note**: `driver.enabled = false` because the EKS GPU AMI (`AL2_x86_64_GPU`) already includes NVIDIA drivers. The operator manages everything else.

### GPU Sharing (Not Recommended for Training)

For RL training, each pod should have exclusive access to all GPUs on a node. Do **not** use MIG or GPU time-slicing -- these add latency and reduce VRAM.

## Topology-Aware Scheduling

### Hostname-Based Spread (Default)

#### Pod Topology Spread Constraints

Ensure training pods spread across nodes (for multi-node) or pack onto a single node (for single-node):

```yaml
# For multi-node: one pod per node
spec:
  topologySpreadConstraints:
    - maxSkew: 1
      topologyKey: kubernetes.io/hostname
      whenUnsatisfiable: DoNotSchedule
      labelSelector:
        matchLabels:
          app: <reference-name>-multi
```

#### Pod Anti-Affinity (Ensure One Training Pod Per Node)

```yaml
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchExpressions:
              - key: app
                operator: In
                values:
                  - <reference-name>-multi
          topologyKey: kubernetes.io/hostname
```

### Network Topology-Aware Scheduling (3+ Nodes)

When scaling beyond 2 nodes or when a cluster placement group cannot guarantee co-location, use EC2 instance topology labels to schedule pods on physically proximate nodes. See [Skill 04: Networking - EFA](networking-efa/SKILL.md#topology-aware-scheduling-for-multi-node-training) for full details.

Nodes are labeled with their position in the datacenter network hierarchy:
- `topology.k8s.aws/network-node-layer-1` -- top-level switch
- `topology.k8s.aws/network-node-layer-2` -- mid-level switch
- `topology.k8s.aws/network-node-layer-3` -- bottom-level switch (closest to instance)

Use these labels to co-locate training pods under the same network switch:

```yaml
# Prefer pods on nodes sharing the same layer-2 network switch
spec:
  topologySpreadConstraints:
    - maxSkew: 1
      topologyKey: topology.k8s.aws/network-node-layer-2
      whenUnsatisfiable: ScheduleAnyway
      labelSelector:
        matchLabels:
          app: <reference-name>-multi
    # Also ensure one pod per node
    - maxSkew: 1
      topologyKey: kubernetes.io/hostname
      whenUnsatisfiable: DoNotSchedule
      labelSelector:
        matchLabels:
          app: <reference-name>-multi
```

A topology labeler DaemonSet applies these labels automatically. See `manifests/topology-labeler.yaml` and [Skill 04](networking-efa/SKILL.md) for setup.

### Node Affinity (Pin to GPU Nodes)

```yaml
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: role
                operator: In
                values:
                  - gpu-training
              - key: node.kubernetes.io/instance-type
                operator: In
                values:
                  - p5.48xlarge
                  - p4de.24xlarge
```

## Priority Classes

Define priority classes to ensure training jobs preempt lower-priority workloads:

```yaml
# priority-classes.yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: training-critical
value: 1000000
globalDefault: false
description: "Priority class for RL training jobs"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: evaluation-normal
value: 500000
globalDefault: false
description: "Priority class for evaluation/benchmarking jobs"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: data-prep-low
value: 100000
globalDefault: false
description: "Priority class for data preparation jobs"
```

Use in pods:

```yaml
spec:
  priorityClassName: training-critical
```

## Resource Quotas

Limit GPU usage per namespace to prevent runaway costs:

```yaml
# resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: training
spec:
  hard:
    requests.nvidia.com/gpu: "16"    # Max 16 GPUs (2 nodes)
    requests.cpu: "400"
    requests.memory: "4Ti"
    persistentvolumeclaims: "5"
```

## Pod Disruption Budgets

Protect long-running training from voluntary disruptions (node upgrades, autoscaler):

```yaml
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: training-pdb
spec:
  minAvailable: 2  # For 2-node training: never disrupt any pod
  selector:
    matchLabels:
      app: <reference-name>-multi
```

For single-node training:

```yaml
spec:
  maxUnavailable: 0
  selector:
    matchLabels:
      app: <reference-name>
```

## Taints and Tolerations

GPU nodes should be tainted to prevent non-GPU workloads from scheduling:

```hcl
# In Terraform (main.tf, eks_managed_node_groups)
taints = [
  {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }
]
```

Training pods tolerate this taint:

```yaml
spec:
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

## NVIDIA GPU Operator (Alternative to Manual Plugin)

> **Note**: The GPU Operator IS the deployed solution in this project (see `infrastructure/addons/main.tf`). The section above documents its configuration. For a managed experience consider the NVIDIA GPU Operator which automates device plugin, DCGM exporter, GPU Feature Discovery, and container toolkit deployment.

For manual standalone installation (not recommended when using the GPU Operator):

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \
  --set toolkit.enabled=true
```

> Set `driver.enabled=false` because the EKS GPU AMI already includes NVIDIA drivers.

## Karpenter (Alternative to Cluster Autoscaler)

For dynamic GPU node provisioning, Karpenter can replace the Cluster Autoscaler:

```yaml
# karpenter-nodepool.yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-training
spec:
  template:
    metadata:
      labels:
        role: gpu-training
    spec:
      requirements:
        - key: node.kubernetes.io/instance-type
          operator: In
          values:
            - p4de.24xlarge
            - p5.48xlarge
        - key: karpenter.sh/capacity-type
          operator: In
          values:
            - on-demand  # Never use spot for training
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: gpu-training
      taints:
        - key: nvidia.com/gpu
          value: "true"
          effect: NoSchedule
  limits:
    nvidia.com/gpu: 16  # Max 16 GPUs across all nodes
  disruption:
    consolidationPolicy: WhenEmpty  # Only remove nodes when no pods
    consolidateAfter: 5m
```

## Host Networking (Optional, for NCCL Performance)

For maximum NCCL performance, use host networking to bypass kube-proxy:

```yaml
spec:
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
```

> **Caution**: Host networking bypasses network policies and port isolation. Use only when NCCL performance is critical and you trust the workloads.

## Validation Checklist

- [ ] NVIDIA device plugin reports GPUs on all training nodes
- [ ] EFA device plugin reports EFA interfaces
- [ ] Priority classes are created and applied to training jobs
- [ ] PDB is protecting running training pods
- [ ] GPU taints prevent non-GPU workloads from scheduling on GPU nodes
- [ ] Training pods are anti-affined (one per node for multi-node)
- [ ] (3+ nodes) Topology labels applied to GPU nodes
- [ ] (3+ nodes) Training pods use topology-aware spread constraints

## Related Skills

- [Skill 01: EKS Cluster Provisioning](eks-cluster-provisioning/SKILL.md) - Cluster setup
- [Skill 09: Kubernetes Manifests](kubernetes-manifests/SKILL.md) - Manifest construction
- [Skill 15: Monitoring and Observability](monitoring-observability/SKILL.md) - GPU metrics
