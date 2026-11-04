<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Capacity scripts: catch, keep, run, release

On-demand GPU capacity for RT-core instances (g6e, g7e) and small p-class instances is
often unavailable in a given AZ for hours. These scripts take whatever capacity appears,
hold it between runs, and size each run to the node they caught, so measurements are
not pinned to one instance type.

| Script | What it does |
|---|---|
| `hold-sim.sh` | 1-GPU placeholder pod for any instance type in `SIM_TYPES`; Karpenter keeps retrying until one launches |
| `keep-node.sh` | CPU-only keeper on the caught node (stops empty-node consolidation), then deletes the placeholder; `release` lets the node go |
| `run-on-held-node.sh` | Renders the IsaacLab GR00T training Job for the held node: GPUs, CPU and memory from the node, matching placement |
| `fsx-move.sh` | Moves FSx to the node's AZ: FSx + PV + PVC re-created (refilled from S3) and the GPU pool pin, then prefetches FSx. System nodes stay in `target_az`; no tracked file is edited |
| `pipeline.sh` | Unattended loop over a queue file: catch, keep, move FSx if needed, run matching jobs, release |

Settings come from the environment (`common.sh`): `KUBECONFIG`, `NAMESPACE`, `IMAGE`,
`SIM_TYPES`, `WORK_DIR`, and for `fsx-move.sh` `FSX_CAPACITY`,
`WORKLOADS_TF_VARS`. Nothing account-specific is hard-coded.

```bash
export KUBECONFIG=~/.kube/<cluster> IMAGE=<ecr repo>:isaaclab-gr00t-latest \
       SIM_TYPES=g6e.24xlarge,g6e.48xlarge QUEUE_FILE=infrastructure/capacity/queue-isaaclab-g6e.txt \
       AUTO_FSX_MOVE=1 WORKLOADS_TF_VARS="TF_VAR_enable_sim_pool=true TF_VAR_gpu_any_az=true ..."
nohup bash infrastructure/capacity/pipeline.sh > /dev/null 2>&1 < /dev/null &
tail -f "${TMPDIR:-/tmp}/rlinf-capacity/pipeline.log"
```

Notes:
- **Keep the node between runs.** The `gpu-sim` NodePool consolidates an empty node after 10 minutes; without the keeper a scarce node is lost between two Jobs.
- **FSx follows the node, not the reverse.** Catch the GPU first (`gpu_any_az=true` in the workloads layer), then move FSx to its AZ. FSx refills from S3, but a fresh filesystem is lazily loaded: the prefetch step avoids a 20-minute first Isaac Sim boot.
- **Run the pipeline detached** (`nohup`). A pipeline tied to an interactive shell was interrupted mid-`terraform apply` once, which left FSx half-replaced.
- `fsx-move.sh` runs Terraform applies (storage, workloads). Only run the pipeline with `AUTO_FSX_MOVE=1` if unattended FSx moves are acceptable. It checks first that no pod uses `fsx-claim`.
