---
name: kubernetes-manifests
description: Construct Kubernetes Jobs, StatefulSets, and Services for GPU-accelerated RL training workloads on EKS
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Skill 09: Kubernetes Manifests

## Purpose

Construct Kubernetes manifests for deploying RL training workloads on EKS. This includes Job definitions, headless Services for multi-node discovery, MPIJobs for NCCL tests, Secrets, and resource requests for GPUs, EFA, and shared storage.

> **Source of truth**: The actual deployed manifests live in `examples/<example>/manifests/`. The content below explains the patterns and design decisions. If this skill and the manifests ever diverge, the manifests are authoritative.

> **Reference manifests**: `container-test.yaml`, `training-statefulset.yaml`, `training-service.yaml` in `infrastructure/manifests/`. Per-example manifests in `examples/<example>/manifests/`.
>
> **Example (RLinf):** See `examples/<example>/manifests/` for the RLinf-specific manifests and `examples/AGENTS.md` for RLinf-specific configuration values (CONFIG_NAME, VENV_NAME, MODEL_PATH, etc.).

> **NCCL test manifest**: [`infrastructure/manifests/nccl-tests-mpijob.yaml`](infrastructure/manifests/nccl-tests-mpijob.yaml) -- MPIJob pattern from awslabs/awsome-distributed-ai. Uses pre-built public ECR image.

## Architecture: Single-Node vs Multi-Node vs MPIJob

### Single-Node (1 pod, 8 GPUs)

```
+-------------------------------------------+
| Kubernetes Job (completions: 1)           |
|                                           |
|  +-------------------------------------+ |
|  | Pod: training-0                      | |
|  | - 8x nvidia.com/gpu                  | |
|  | - 4x vpc.amazonaws.com/efa           | |
|  | - /fsx mounted                       | |
|  | - Ray head (local)                   | |
|  | - FSDP across 8 GPUs                 | |
|  +-------------------------------------+ |
+-------------------------------------------+
```

### Multi-Node (2 pods, 16 GPUs)

```
+--------------------------------------------------+
| Headless Service: training-svc                    |
|                                                   |
|  +---------------------+  +---------------------+|
|  | Pod: training-0     |  | Pod: training-1     ||
|  | (Ray head + worker) |  | (Ray worker)        ||
|  | 8x GPU, 4x EFA      |  | 8x GPU, 4x EFA      ||
|  | /fsx mounted        |  | /fsx mounted        ||
|  +---------------------+  +---------------------+|
|         NCCL over EFA (allreduce)                 |
+--------------------------------------------------+
```

## Single-Node Training Job

The single-node Job runs the training script directly -- no init container is needed because the training script handles model preparation internally. The image URI uses `${ECR_URI}` which is substituted at deploy time via `envsubst`.

```yaml
# training-job.yaml
#
# Before applying, substitute the ECR image URI:
#   export ECR_URI=<account>.dkr.ecr.<region>.amazonaws.com/rlinf-on-eks/rlinf
#   envsubst < manifests/training-job.yaml | kubectl apply -f -
#
# Or manually replace ${ECR_URI} below.
apiVersion: batch/v1
kind: Job
metadata:
  name: rlinf-training  # Replace with your reference name
  labels:
    app: rlinf  # Replace with your reference name
    reference: rlinf
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: rlinf
    spec:
      restartPolicy: Never
      serviceAccountName: training-sa
      nodeSelector:
        role: gpu-training
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule

      containers:
        - name: training
          image: ${ECR_URI}:latest  # Replace or use envsubst
          command:
            - bash
            - /workspace/eks/scripts/run_training_eks.sh  # Replace with your launch script
          resources:
            requests:
              nvidia.com/gpu: 8
              vpc.amazonaws.com/efa: 4
              cpu: "90"
              memory: "1000Gi"
            limits:
              nvidia.com/gpu: 8
              vpc.amazonaws.com/efa: 4
              cpu: "96"
              memory: "1100Gi"
          env:
            - name: MLFLOW_TRACKING_URI  # Optional: only if MLflow addon is enabled
              value: "http://mlflow.rlinf.svc.cluster.local"
            - name: NUM_GPUS
              value: "8"
            - name: NUM_NODES
              value: "1"
            - name: CONFIG_NAME
              value: "maniskill_openvla_ppo"  # Replace with your config name
            - name: VENV_NAME
              value: "openvla"  # Replace with your venv name
            - name: MODEL_PATH
              value: "/fsx/models/openvla-7b"  # Replace with your model path
            - name: EXPERIMENT_NAME
              value: "maniskill-openvlaoft-ppo"  # Replace with your experiment name
            - name: NVIDIA_DRIVER_CAPABILITIES
              value: "all"
            - name: NCCL_DEBUG
              value: "WARN"
            - name: FI_PROVIDER
              value: "efa"
            - name: FI_EFA_USE_DEVICE_RDMA
              value: "1"
            - name: FI_EFA_FORK_SAFE
              value: "1"
            - name: PYTORCH_CUDA_ALLOC_CONF
              value: "expandable_segments:True"
            - name: TOKENIZERS_PARALLELISM
              value: "true"
            - name: TORCH_NCCL_AVOID_RECORD_STREAMS
              value: "1"
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

### Key design decisions (single-node)

| Decision | Rationale |
|----------|-----------|
| No init container | The training script (`<launch-script>`) handles checkpoint preparation internally, removing the need for a separate pre-step. |
| `${ECR_URI}:latest` image | Uses `envsubst` at deploy time so the same manifest works across accounts/regions without editing. |
| `CONFIG_NAME`, `VENV_NAME`, `MODEL_PATH`, `EXPERIMENT_NAME` env vars | Parameterized so the same manifest works for different examples (swap values per experiment). |
| EFA/NCCL env vars | `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`, `FI_EFA_FORK_SAFE=1` configure libfabric for RDMA; `TORCH_NCCL_AVOID_RECORD_STREAMS=1` reduces CUDA memory fragmentation. |

## Multi-Node Training (StatefulSet + Headless Service)

### Headless Service

The headless Service provides DNS-based pod discovery. It must be applied **before** the StatefulSet so that DNS records are available when pods start.

```yaml
# training-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: training-svc
  labels:
    app: rlinf-multi  # Replace with your reference name
spec:
  clusterIP: None  # Headless -- pods discover each other via DNS
  selector:
    app: rlinf-multi
  ports:
    - name: ray-head
      port: 6379
      targetPort: 6379
    - name: ray-dashboard
      port: 8265
      targetPort: 8265
    - name: ray-client
      port: 10001
      targetPort: 10001
```

### StatefulSet

The StatefulSet deploys N pods (one per GPU node). Pod-0 starts a Ray head, waits for all workers to connect, then launches the training script. Pod-1+ are Ray workers that connect to the head and block. The image URI uses `${ECR_URI}` substituted via `envsubst` at deploy time.

```yaml
# training-statefulset.yaml
#
# Before applying, substitute the ECR image URI:
#   export ECR_URI=<account>.dkr.ecr.<region>.amazonaws.com/rlinf-on-eks/rlinf
#   envsubst < manifests/training-statefulset.yaml | kubectl apply -f -
#
# Or manually replace ${ECR_URI} below.
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: training
  labels:
    app: rlinf-multi  # Replace with your reference name
    reference: rlinf
spec:
  serviceName: training-svc
  replicas: 2  # Number of nodes
  podManagementPolicy: Parallel  # Start all pods simultaneously
  selector:
    matchLabels:
      app: rlinf-multi
  template:
    metadata:
      labels:
        app: rlinf-multi
    spec:
      serviceAccountName: training-sa
      nodeSelector:
        role: gpu-training
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      terminationGracePeriodSeconds: 120

      # Topology-aware scheduling:
      # 1. Prefer nodes under the same layer-2 network switch (lowest NCCL latency)
      # 2. Enforce one pod per node (anti-affinity)
      # If topology labels are absent, ScheduleAnyway falls back gracefully.
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.k8s.aws/network-node-layer-2
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: rlinf-multi

      # One pod per node
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchExpressions:
                  - key: app
                    operator: In
                    values:
                      - rlinf-multi
              topologyKey: kubernetes.io/hostname

      containers:
        - name: training
          image: ${ECR_URI}:latest  # Replace or use envsubst
          command:
            - bash
            - -c
            - |
              set -euo pipefail
              ORDINAL=${HOSTNAME##*-}
              # Use Downward API namespace, fallback to default
              NAMESPACE="${POD_NAMESPACE:-default}"
              HEAD_ADDR="training-0.training-svc.${NAMESPACE}.svc.cluster.local"

              if [ "$ORDINAL" = "0" ]; then
                echo "=== Starting Ray head node ==="
                ray start --head --port=6379 --dashboard-host=0.0.0.0

                echo "Waiting for $NUM_NODES Ray nodes..."
                TIMEOUT=300
                ELAPSED=0
                while [ "$(python3 -c 'import ray; ray.init(address="auto"); print(len(ray.nodes()))' 2>/dev/null || echo 0)" -lt "$NUM_NODES" ]; do
                  sleep 5
                  ELAPSED=$((ELAPSED + 5))
                  if [ $ELAPSED -ge $TIMEOUT ]; then
                    echo "ERROR: Timed out waiting for Ray workers after ${TIMEOUT}s"
                    exit 1
                  fi
                done
                echo "All $NUM_NODES Ray nodes connected. Starting training..."

                bash /workspace/eks/scripts/run_training_eks.sh  # Replace with your launch script
              else
                echo "=== Starting Ray worker, connecting to $HEAD_ADDR ==="
                ray start --address="${HEAD_ADDR}:6379" --block
              fi
          env:
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
            - name: NUM_NODES
              value: "2"
            - name: NUM_GPUS
              value: "8"
            - name: CONFIG_NAME
              value: "maniskill_openvla_ppo"  # Replace with your config name
            - name: VENV_NAME
              value: "openvla"  # Replace with your venv name
            - name: MODEL_PATH
              value: "/fsx/models/openvla-7b"  # Replace with your model path
            - name: EXPERIMENT_NAME
              value: "maniskill-openvlaoft-ppo"  # Replace with your experiment name
            - name: MLFLOW_TRACKING_URI  # Optional: only if MLflow addon is enabled
              value: "http://mlflow.rlinf.svc.cluster.local"
            - name: NVIDIA_DRIVER_CAPABILITIES
              value: "all"
            - name: NCCL_DEBUG
              value: "WARN"
            - name: FI_PROVIDER
              value: "efa"
            - name: FI_EFA_USE_DEVICE_RDMA
              value: "1"
            - name: FI_EFA_FORK_SAFE
              value: "1"
            - name: PYTORCH_CUDA_ALLOC_CONF
              value: "expandable_segments:True"
            - name: TOKENIZERS_PARALLELISM
              value: "true"
            - name: TORCH_NCCL_AVOID_RECORD_STREAMS
              value: "1"
          resources:
            requests:
              nvidia.com/gpu: 8
              vpc.amazonaws.com/efa: 4
              cpu: "90"
              memory: "1000Gi"
            limits:
              nvidia.com/gpu: 8
              vpc.amazonaws.com/efa: 4
              cpu: "96"
              memory: "1100Gi"
          ports:
            - containerPort: 6379
              name: ray-head
            - containerPort: 8265
              name: ray-dashboard
          # Liveness: check Ray head is responding
          livenessProbe:
            exec:
              command:
                - bash
                - -c
                - "ray status > /dev/null 2>&1 || exit 0"
            initialDelaySeconds: 120
            periodSeconds: 60
            failureThreshold: 5
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

### Key design decisions (multi-node)

| Decision | Rationale |
|----------|-----------|
| `POD_NAMESPACE` via Downward API | The head address FQDN includes the namespace (`training-0.training-svc.<ns>.svc.cluster.local`). Using `fieldRef: metadata.namespace` makes the manifest namespace-portable rather than hardcoding `default`. |
| `topologySpreadConstraints` on `topology.k8s.aws/network-node-layer-2` | Prefers co-locating pods under the same layer-2 switch for lowest NCCL latency. `ScheduleAnyway` means it is best-effort -- the cluster still works if the label is absent. |
| `podAntiAffinity` on `kubernetes.io/hostname` | Ensures exactly one training pod per physical node. Two pods on the same node would contend for GPUs. |
| Ray worker detection via `python3 -c 'import ray; ...'` | Counts connected Ray nodes programmatically with a 300-second timeout. More reliable than parsing `ray status` output with grep. |
| No separate init container | The training script (`<launch-script>`) handles all setup internally via `VENV_NAME` and `CONFIG_NAME` env vars. |
| Liveness probe on `ray status` | Detects Ray head crashes. The `|| exit 0` ensures workers (which do not run a head) do not get killed by the probe. `initialDelaySeconds: 120` gives Ray time to start before probing begins. |
| `${ECR_URI}:latest` image | Same `envsubst` pattern as the single-node Job for account/region portability. |

## Supporting Resources

### Resource Sizing Guide

| Instance Type | GPU Request | EFA Request | CPU Request | Memory Request |
|--------------|-------------|-------------|-------------|----------------|
| p4de.24xlarge | 8 | 4 | 90 | 1000Gi |
| p5.48xlarge | 8 | 32 | 180 | 1800Gi |

### /dev/shm Sizing

PyTorch DataLoader workers and NCCL use shared memory. Always mount a large `emptyDir` with `medium: Memory`:

```yaml
volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: "64Gi"  # At least 64Gi for multi-GPU training
```

Without this, training will fail with "Bus error" or "Insufficient shared memory".

## YAML/Bash Interaction Pitfalls

Bash scripts embedded in YAML `|` block scalars are a common source of subtle failures in training manifests. YAML parsers can misinterpret:

- **Semicolons (`;`)**: Can terminate YAML values. `if test -f foo; then echo yes; fi` may fail.
- **Curly braces (`{}`)**: Interpreted as YAML flow mappings. `${ }` and `{ echo ...; }` can cause parse errors.
- **`&&`/`||` chains**: Long chains with variable assignments may lose variable scope (subshell execution).
- **`for` loops with `;`**: `for x in a b; do ... done` -- the `;` before `do` breaks in many YAML parsers.
- **Colons (`:`)**: Bare colons (e.g., in image tags `$ECR_URI:latest`) are parsed as YAML key-value separators.

### Prescriptive Approach

1. **Keep inline bash simple** -- sequential commands, basic `if/then/else/fi` (with `then` on the same line using newline, not `;`).
2. **For complex logic, use a ConfigMap-mounted script:**

```yaml
volumes:
  - name: scripts
    configMap:
      name: training-scripts
      defaultMode: 0755
containers:
  - name: training
    command: ["/scripts/run.sh"]
    volumeMounts:
      - name: scripts
        mountPath: /scripts
```

3. **For test suites, use Python instead of bash** -- Python has no YAML interaction issues when mounted via ConfigMap.
4. **Always validate**: `kubectl apply --dry-run=client -f manifest.yaml` before deploying.

## Image Pull Considerations

Physical AI RL container images are typically 15-20 GB compressed. Image pull time impacts pod startup and job deadlines.

| Scenario | Pull Time | Recommendation |
|----------|-----------|----------------|
| First pull (cold) | 5-10 min | Account for this in `activeDeadlineSeconds` |
| Same image, same node (cached) | 0s | Default `IfNotPresent` policy |
| Updated `:latest` tag | 5-10 min | Only if `imagePullPolicy: Always` |
| Specific version tag (cached) | 0s | Best practice for production |

### Best Practices

- **Do NOT set `imagePullPolicy: Always`** for large training images. The default (`IfNotPresent`) is correct.
- **Tag images with specific versions** (e.g., `:20260401-085549`), not just `:latest`. Update the tag in manifests when the image changes.
- **Add 10 minutes to `activeDeadlineSeconds`** to account for cold image pulls. A 1-hour training test should use `activeDeadlineSeconds: 4200` (70 min).
- **Pre-pull images** on GPU nodes using a DaemonSet if predictable startup time is critical.

## Deployment

All manifests use `${ECR_URI}` as a placeholder for the container image. Substitute it at deploy time with `envsubst`:

```bash
# Set your ECR image URI
export ECR_URI=<account>.dkr.ecr.<region>.amazonaws.com/rlinf-on-eks/rlinf  # Replace with your values

# Single-node
envsubst < manifests/training-job.yaml | kubectl apply -f -

# Multi-node
kubectl apply -f manifests/training-service.yaml
envsubst < manifests/training-statefulset.yaml | kubectl apply -f -

# Monitor
kubectl logs -f job/rlinf-training       # single-node
kubectl logs -f training-0                      # multi-node head
kubectl logs -f training-1                      # multi-node worker
```

## Validation Checklist

- [ ] Pod schedules on GPU node with all resources allocated
- [ ] `nvidia-smi` inside pod shows 8 GPUs
- [ ] `/fsx` is mounted and readable/writable
- [ ] `/dev/shm` is mounted with sufficient size
- [ ] EFA devices are available (`ls /sys/class/infiniband/`)
- [ ] Training starts and logs appear in MLflow
- [ ] For multi-node: Ray workers connect to head, FSDP initializes across all ranks
- [ ] For multi-node: `ray status` on head pod shows all nodes connected
- [ ] For multi-node: liveness probe does not restart pods during training

## Related Skills

- [Skill 07: Container Image Building](container-image-building/SKILL.md) - Image referenced in manifests
- [Skill 08: PyTorch Training Scripts](pytorch-training-scripts/SKILL.md) - Scripts launched by these manifests
- [Skill 10: Kubernetes Native Features](kubernetes-native-features/SKILL.md) - Scheduling and topology
