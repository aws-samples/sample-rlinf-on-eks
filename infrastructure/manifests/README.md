<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Infrastructure Manifests

Kubernetes manifests for validating and operating the cluster infrastructure. These are **not** training workloads — they verify that GPUs (Graphics Processing Units), EFA (Elastic Fabric Adapter) networking, storage, and the container image are working correctly before you launch training.

## When to Use

Apply these after `infrastructure/deploy.sh --action apply --layer all` completes and you have `kubectl` access to the cluster.

## Manifests

| File | Purpose | Namespace | Prerequisites |
|------|---------|-----------|---------------|
| `smoke-test.yaml` | Validates GPU, EFA devices, CUDA (Compute Unified Device Architecture), FSx mount on a single node | `default` | GPU node provisioned, FSx PVC available |
| `container-test.yaml` | Tests the built container image (10 checks: CUDA, EFA libs, multi-venv, ManiSkill, Ray, OpenMPI, FSx) | `rlinf` | Container image built and pushed to ECR |
| `topology-labeler.yaml` | DaemonSet that applies EC2 network topology labels to GPU nodes for topology-aware scheduling | `kube-system` | GPU nodes running |
| `nccl-tests-mpijob.yaml` | MPIJob that measures NCCL (NVIDIA Collective Communications Library) allreduce bandwidth over EFA; verifies inter-node communication | `default` | MPI Operator addon enabled, 2+ GPU nodes |
| `training-service.yaml` | Headless Service for Ray pod discovery (used by L6 multi-node validation) | `rlinf` | — |
| `training-statefulset.yaml` | Multi-node Ray head/worker StatefulSet (2 replicas, 8 GPUs each) | `rlinf` | Container image in ECR, FSx PVC, 2+ GPU nodes |

## Usage

```bash
# Set ECR_URI for manifests that reference the training container image
export ECR_URI=$(terraform -chdir=infrastructure/build output -raw ecr_repository_url)

# L0: Infrastructure smoke test (no container image needed — uses NGC (NVIDIA GPU Cloud) PyTorch)
kubectl apply -f infrastructure/manifests/smoke-test.yaml
kubectl logs -f job/smoke-test

# L2: Container validation (after CodeBuild completes)
envsubst < infrastructure/manifests/container-test.yaml | kubectl apply -f -
kubectl logs -f job/container-test -n rlinf

# Topology labeler (once, enables topology-aware scheduling for multi-node)
kubectl apply -f infrastructure/manifests/topology-labeler.yaml

# L3: NCCL bandwidth test (requires MPI (Message Passing Interface) Operator addon)
export NCCL_TEST_IMAGE=public.ecr.aws/hpc-cloud/nccl-tests:cuda12.8.1-efa1.43.2-ofiv1.16.3-ncclv2.27.7-1-testsv2.16.9
envsubst '${NCCL_TEST_IMAGE}' < infrastructure/manifests/nccl-tests-mpijob.yaml | kubectl apply -f -
```

### NCCL Test: Instance Type Configuration

The NCCL test manifest defaults to **g6.8xlarge** settings (1 GPU per worker, 1 EFA device, 2 total GPUs). For other instance types, edit the manifest before applying:

| Instance | `slotsPerWorker` | `-np` | `-N` | GPU limit | EFA limit |
|----------|-----------------|-------|------|-----------|-----------|
| g6.8xlarge | 1 | 2 | 1 | 1 | 1 |
| p4de.24xlarge | 8 | 16 | 8 | 8 | 4 |
| p5.48xlarge | 8 | 16 | 8 | 8 | 32 |
| p5en.48xlarge | 8 | 16 | 8 | 8 | 16 |

Expected bandwidth (2 nodes, 1GB allreduce): ~437 GB/s for p5.48xlarge, ~2-3 GB/s for g6.8xlarge.

## Cleanup

```bash
# Delete after validation
kubectl delete job smoke-test
kubectl delete job container-test -n rlinf
kubectl delete mpijob nccl-tests
kubectl delete statefulset training -n rlinf
kubectl delete service training-svc -n rlinf

# Topology labeler (only if no longer needed)
kubectl delete -f infrastructure/manifests/topology-labeler.yaml
```

## Automated Validation

All of these are run automatically by `./validate.sh --mode cluster`:

| Level | Manifest | Timeout |
|-------|----------|---------|
| L0 | `smoke-test.yaml` | 10 min |
| L2 | `container-test.yaml` | 20 min |
| L3 | `nccl-tests-mpijob.yaml` | 10 min |
| L6 | `training-service.yaml` + `training-statefulset.yaml` | 20 min |
