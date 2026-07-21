---
name: rlinf-open-source-plugins
description: "RLinf: Integrate Ray, veRL, vLLM, and KubeRay into the RLinf training stack on EKS"
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 11: Open-Source Plugins

## Purpose

Integrate open-source frameworks -- Ray, RLinf, vLLM, and KubeRay -- into the Physical AI RL training stack on EKS. These provide distributed execution, RL training orchestration, and efficient inference.

## Component Overview

| Component | Role | Used By RLinf |
|-----------|------|---------------------|
| **Ray** | Distributed execution framework | Yes -- worker orchestration, resource pools |
| **RLinf** | RL training framework (FSDP + Ray) | Yes -- core training loop |
| **vLLM** | Fast LLM inference (optional rollout backend) | Optional -- `rollout.name=vllm` mode |
| **KubeRay** | Kubernetes operator for Ray clusters | Optional -- alternative to manual Ray setup |

## Ray

### Role in RLinf

Ray manages distributed workers within the training pipeline:

```
main_ppo.py
  └── ray.init()
        └── RayResourcePool (maps GPUs to placement groups)
              └── RayWorkerGroup
                    └── RobActorRolloutRefWorker (per GPU group)
                          ├── Actor (FSDP training)
                          ├── Rollout (env interaction)
                          └── Ref (reference model)
```

### Configuration

RLinf initializes Ray locally within the pod. For single-node:

```python
# In main_ppo.py
ray.init(runtime_env=runtime_env)  # Local Ray cluster
```

For multi-node, Ray head starts on pod-0 and workers connect:

```bash
# Pod 0 (head)
ray start --head --port=6379

# Pod 1+ (workers)
ray start --address=training-0.training-svc:6379
```

### Ray Resource Allocation

```python
# Resource pool: maps GPU groups to workers
# Format: {pool_id: [gpus_per_node] * num_nodes}
resource_pool_spec = {
    global_pool_id: [num_gpus_per_node] * num_nodes
}
# Example: 2 nodes x 8 GPUs = {pool: [8, 8]}
```

### Ray Environment Variables

```json
{
  "env_vars": {
    "MLFLOW_TRACKING_URI": "http://mlflow.rlinf.svc.cluster.local",  // Optional: only if MLflow addon is enabled
    "NCCL_DEBUG": "WARN",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "FI_PROVIDER": "efa",
    "FI_EFA_USE_DEVICE_RDMA": "1"
  }
}
```

## RLinf

### Role

RLinf provides the core RL training infrastructure:
- FSDP model wrapping and sharding
- PPO/GRPO policy gradient computation
- Hybrid actor-critic-rollout worker management
- Checkpoint saving/loading with FSDP
- Integration with Ray for distributed execution
- Macro-to-micro flow transformation for throughput optimization

### RLinf Extensions

RLinf includes VLA-specific components built on its flow transformation architecture:

| RLinf Component | Extension | File |
|---------------|----------------------|------|
| ActorRolloutRefWorker | **RobActorRolloutRefWorker** | `verl/workers/fsdp_workers.py` |
| DataParallelPPOActor | **RobDataParallelPPOActor** | `verl/workers/actor/dp_rob.py` |
| HFRollout | **RobHFRollout** | `verl/workers/rollout/rob_rollout.py` |
| core_algos.py | Added asymmetric clipping | `verl/trainer/ppo/core_algos.py` |
| RayTrainer | Added dynamic sampling filter | `verl/trainer/ppo/ray_trainer.py` |

### RLinf Installation

RLinf v0.2 is installed in the container image (see Skill 07):

```bash
git clone https://github.com/RLinf/RLinf.git
cd RLinf && pip install -e .
```

RLinf's `verl/` directory **overrides** the upstream modules. The `PYTHONPATH` must prioritize RLinf:

```bash
export PYTHONPATH="/workspace:${PYTHONPATH}"
```

## vLLM (Optional)

### Role

vLLM provides optimized LLM inference for the rollout phase. RLinf supports two rollout backends:

| Backend | Config | Speed | Compatibility |
|---------|--------|-------|---------------|
| **HuggingFace** (default) | `rollout.name=hf` | Slower | Full VLA support |
| **vLLM** | `rollout.name=vllm` | Faster | Requires vLLM VLA support |

The HF backend is the default and recommended for VLA models because vLLM's VLA support is still maturing.

### When to Use vLLM

- When rollout speed is the bottleneck (many environments, short episodes)
- When vLLM has added support for the specific VLA architecture (check vLLM release notes)
- For evaluation-only runs where rollout speed matters most

### vLLM Configuration (if used)

```yaml
actor_rollout_ref.rollout.name: vllm
actor_rollout_ref.rollout.tensor_model_parallel_size: 2
actor_rollout_ref.rollout.gpu_memory_utilization: 0.5
actor_rollout_ref.rollout.max_num_batched_tokens: 8192
actor_rollout_ref.rollout.enforce_eager: True
```

## KubeRay (Optional)

### Role

KubeRay is a Kubernetes operator that manages Ray clusters as native Kubernetes resources. It provides:
- Automatic Ray head/worker lifecycle management
- Kubernetes-native scaling
- Built-in health checks and recovery
- Integration with Kubernetes RBAC

### When to Use KubeRay

| Scenario | Manual Ray (default) | KubeRay |
|----------|---------------------|---------|
| Single-node training | Simpler, no operator needed | Overhead not justified |
| Multi-node training | Manual head/worker management | Auto-manages Ray cluster |
| Fault tolerance | Manual restart | Auto-recovery of workers |
| Multiple concurrent experiments | Manual coordination | CRD-based management |

### KubeRay Installation

KubeRay is installed as a Helm release in Terraform (see Skill 01 `helm.tf`):

```hcl
resource "helm_release" "kuberay_operator" {
  count = var.enable_kuberay ? 1 : 0

  name             = "kuberay-operator"
  repository       = "https://ray-project.github.io/kuberay-helm/"
  chart            = "kuberay-operator"
  namespace        = "ray-system"
  create_namespace = true
}
```

For manual installation:

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

helm install kuberay-operator kuberay/kuberay-operator \
  --namespace ray-system \
  --create-namespace
```

### RayCluster CRD for RLinf

```yaml
# raycluster.yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: rlinf-cluster
spec:
  rayVersion: "2.38.0"
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        serviceAccountName: training-sa
        nodeSelector:
          role: gpu-training
        tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
        containers:
          - name: ray-head
              image: <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/rlinf-on-eks/rlinf:latest
            resources:
              requests:
                nvidia.com/gpu: 8
                vpc.amazonaws.com/efa: 4
                cpu: "90"
                memory: "1000Gi"
              limits:
                nvidia.com/gpu: 8
                vpc.amazonaws.com/efa: 4
            volumeMounts:
              - name: fsx
                mountPath: /fsx
              - name: dshm
                mountPath: /dev/shm
            env:
              - name: MLFLOW_TRACKING_URI  # Optional: only if MLflow addon is enabled
                value: "http://mlflow.rlinf.svc.cluster.local"
        volumes:
          - name: fsx
            persistentVolumeClaim:
              claimName: fsx-claim
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: "64Gi"

  workerGroupSpecs:
    - replicas: 1  # Additional worker nodes
      groupName: gpu-workers
      rayStartParams: {}
      template:
        spec:
          serviceAccountName: training-sa
          nodeSelector:
            role: gpu-training
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
          containers:
            - name: ray-worker
            image: <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/rlinf-on-eks/rlinf:latest
              resources:
                requests:
                  nvidia.com/gpu: 8
                  vpc.amazonaws.com/efa: 4
                  cpu: "90"
                  memory: "1000Gi"
                limits:
                  nvidia.com/gpu: 8
                  vpc.amazonaws.com/efa: 4
              volumeMounts:
                - name: fsx
                  mountPath: /fsx
                - name: dshm
                  mountPath: /dev/shm
          volumes:
            - name: fsx
              persistentVolumeClaim:
                claimName: fsx-claim
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: "64Gi"
```

### Submit Training Job to RayCluster

```bash
# Apply the RayCluster
kubectl apply -f raycluster.yaml

# Wait for cluster to be ready
kubectl get raycluster rlinf-cluster

# Submit job via RayJob CRD
kubectl apply -f rayjob.yaml
```

```yaml
# rayjob.yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: rlinf-training
spec:
  entrypoint: "bash /workspace/scripts/run_training_eks.sh"
  runtimeEnvYAML: |
    env_vars:
      NCCL_DEBUG: WARN
      FI_PROVIDER: efa
  clusterSelector:
    ray.io/cluster: rlinf-cluster
```

## Validation Checklist

- [ ] Ray initializes and discovers all GPUs across nodes
- [ ] RLinf `RobActorRolloutRefWorker` creates successfully
- [ ] FSDP wrapping completes without OOM
- [ ] Rollout generates trajectories from simulation environments
- [ ] (If KubeRay) RayCluster CRD shows Ready state
- [ ] (If vLLM) vLLM backend loads model and generates actions

## Related Skills

- [Skill 08: PyTorch Training Scripts](pytorch-training-scripts/SKILL.md) - Scripts using these frameworks
- [Skill 09: Kubernetes Manifests](kubernetes-manifests/SKILL.md) - K8s resources for these plugins
- [Skill 04: Networking - EFA](networking-efa/SKILL.md) - EFA networking for NCCL
