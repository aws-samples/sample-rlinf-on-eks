# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Isaac Sim / IsaacLab smoke test for an RT-core GPU node (no training).

Mirrors rlinf/envs/sim/isaaclab/tasks/stack_cube.py:make_env_isaaclab: headless
AppLauncher with cameras, the RLinf stack-cube task, N envs at 256x256. It then steps
random actions and reports timing, GPU memory, and whether the camera frames are real
renders (not all-black or constant).

Run inside the gr00t venv after `source $ISAAC_PATH/setup_conda_env.sh`:
  python isaaclab_smoke.py --num-envs 16 --steps 100 --out /fsx/isaacsim-smoke
"""

import argparse
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--task", default="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0")
parser.add_argument("--out", default="/fsx/isaacsim-smoke")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
result = {"num_envs": args.num_envs, "steps": args.steps, "task": args.task}

os.environ.pop("DISPLAY", None)
t0 = time.time()
from isaaclab.app import AppLauncher  # noqa: E402

sim_app = AppLauncher(headless=True, enable_cameras=True).app
result["app_launch_s"] = round(time.time() - t0, 1)
print(f"AppLauncher up in {result['app_launch_s']} s", flush=True)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401  (registers tasks, incl. RLinf/IsaacLab's Rewarded stack-cube)
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.seed = 0
cfg.scene.num_envs = args.num_envs
for cam in ("wrist_cam", "table_cam"):
    getattr(cfg.scene, cam).height = 256
    getattr(cfg.scene, cam).width = 256

t1 = time.time()
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array").unwrapped
obs, _ = env.reset()
result["env_create_reset_s"] = round(time.time() - t1, 1)
print(f"env created + reset in {result['env_create_reset_s']} s", flush=True)

act_shape = env.action_space.shape
t2 = time.time()
for _ in range(args.steps):
    action = torch.rand(act_shape, device=env.device) * 2 - 1
    obs, rew, term, trunc, info = env.step(action)
torch.cuda.synchronize()
dt = time.time() - t2
result["step_s_total"] = round(dt, 2)
result["env_steps_per_s"] = round(args.steps * args.num_envs / dt, 1)

for cam in ("table_cam", "wrist_cam"):
    img = obs["policy"][cam][0].detach().float().cpu().numpy()
    result[f"{cam}_shape"] = list(img.shape)
    result[f"{cam}_mean"] = round(float(img.mean()), 3)
    result[f"{cam}_std"] = round(float(img.std()), 3)
    try:
        from PIL import Image

        arr = img if img.max() > 1.5 else img * 255.0
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)[..., :3]).save(
            os.path.join(args.out, f"{cam}_n{args.num_envs}.png")
        )
    except Exception as e:  # PIL missing is not a failure of the smoke test
        result[f"{cam}_png_error"] = repr(e)

result["gpu_mem_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
free, total = torch.cuda.mem_get_info()
result["gpu_mem_used_gb"] = round((total - free) / 1e9, 2)
result["gpu_mem_total_gb"] = round(total / 1e9, 2)
result["gpu_name"] = torch.cuda.get_device_name(0)
result["render_ok"] = all(result[f"{c}_std"] > 1.0 for c in ("table_cam", "wrist_cam"))
print("SMOKE_RESULT " + json.dumps(result), flush=True)
with open(os.path.join(args.out, f"result_n{args.num_envs}.json"), "w") as f:
    json.dump(result, f, indent=1)
env.close()
sim_app.close()
