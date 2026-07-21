---
name: rlinf-functional-testing
description: "RLinf: Validate RLinf training examples end-to-end from container tests to single-step training"
---

<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# RLinf Functional Testing

## Purpose

RLinf-specific test configurations for the validation harness (L2-L6). These complement the generic testing framework in the shared `functional-testing` skill with concrete RLinf values.

## L2: Container Validation

The container test (`infrastructure/manifests/container-test.yaml`) runs 10 tests verifying:
- Multi-venv availability (openvla, openvla-oft, openpi)
- EFA libraries (libfabric, aws-ofi-nccl)
- CUDA/GPU detection
- RLinf framework imports (`import verl`, `from flash_attn import flash_attn_func`)
- MuJoCo/ManiSkill rendering (`MUJOCO_GL=osmesa`)

## L4: Model Download

The model download jobs (`examples/<example>/manifests/model-download.yaml`) download models to FSx:

| Model | FSx Path | Size |
|-------|----------|------|
| `gen-robot/openvla-7b-rlvla-warmup` | `/fsx/models/openvla-7b-rlvla-warmup` | ~14 GB |
| `RLinf/Openvla-oft-SFT-libero10-trajall` | `/fsx/models/openvla-oft-sft-libero10` | ~14 GB |
| `RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal` | `/fsx/models/rlinf-pi0-sft-spatial` | ~14 GB |

## L5: Per-Example Training Step Tests

Each example runs 1 PPO training step via `HYDRA_OVERRIDES="runner.max_steps=1 runner.save_interval=-1 runner.val_check_interval=-1"`.

### Example Configurations

Defined in `tests/lib/cluster.sh` L5_EXAMPLE_LIST:

| Example Name | `VENV_NAME` | `CONFIG_NAME` | `MODEL_PATH` |
|-------------|-------------|---------------|--------------|
| `maniskill-openvla` | `openvla` | `maniskill_ppo_openvla_quickstart` | `/fsx/models/openvla-7b-rlvla-warmup` |
| `maniskill-openvlaoft` | `openvla-oft` | `maniskill_ppo_openvlaoft_quickstart` | `/fsx/models/openvla-oft-sft-libero10` |
| `libero-pi0` | `openpi` | `libero_spatial_ppo_openpi_quickstart` | `/fsx/models/rlinf-pi0-sft-spatial` |

### Running a Specific Example

```bash
./validate.sh --mode cluster --level 5 --skip-to 5 --example maniskill-openvla
```

### Expected Success Indicators

In the pod logs, look for:
- `TRAINING_STEP_TEST_PASSED` -- explicit success marker
- `Global Step: 1/1` -- training step completed
- No `Traceback`, `ERROR:`, or `FAILED` in output

### Common Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM during FSDP init | GPU too small for 7B model | Use p-class instances (not g6) |
| `ModuleNotFoundError: verl` | Wrong venv activated | Check `VENV_NAME` matches example |
| `input()` hang (no output) | LIBERO import without `LIBERO_DATASET_DIR` | Set env var and pre-create config |
| `FileNotFoundError: config.json` | Model not downloaded | Run L4 first |

## L6: Multi-Node Training

Uses `infrastructure/manifests/training-statefulset.yaml` + `training-service.yaml`. Requires 2+ GPU nodes. Ray head starts on pod-0, workers connect via headless service `training-svc`.

The test patches `HYDRA_OVERRIDES` onto the StatefulSet for single-step validation.

## Related Skills

- `functional-testing` -- generic testing framework and validation levels
- `rlinf-pytorch-training-scripts` -- Hydra configs and training entrypoints
- `rlinf-dataset-preparation` -- model download details

## Validation Checklist

- [ ] L2 container test passes all 10 tests
- [ ] L4 all 3 models downloaded and verified on FSx
- [ ] L5 each example completes 1 training step without OOM or error
- [ ] L6 multi-node Ray cluster forms and completes 1 training step (if 2+ GPU nodes)
