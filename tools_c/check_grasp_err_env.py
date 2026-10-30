# Check that env obs["policy"]["grasp_err" / "grasp_err_ee" / "place_err_ee"] equal the offline functions of the same-step policy obs
# (imitation_pland.md 6.1). Replays the actions of one human demo so the arm actually moves.
#
#   ./isaaclab.sh -p tools_c/check_grasp_err_env.py --dataset datasets/team10_lee_s3.hdf5 --headless
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0")
parser.add_argument("--dataset", default="datasets/team10_lee_s3.hdf5")
parser.add_argument("--demo", default="demo_0")
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import h5py
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.place.mdp import grasp_feat as gf
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env_cfg.terminations.time_out = None
env_cfg.terminations.success = None
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
pol = env_cfg.observations.policy

with h5py.File(args_cli.dataset, "r") as f:
    actions = torch.as_tensor(f["data"][args_cli.demo]["actions"][:], dtype=torch.float32, device=env.device)

obs, _ = env.reset()
max_err, n = 0.0, 0
for t in range(min(args_cli.steps, len(actions))):
    o = obs["policy"]
    pose = (o["eef_pos"], o["eef_quat"], o["toy_truck_positions"], o["toy_truck_orientations"])
    refs = {
        "grasp_err": gf.grasp_err_from_poses(*pose, **pol.grasp_err.params),
        "grasp_err_ee": gf.grasp_err_from_poses(*pose, **pol.grasp_err_ee.params),
        "place_err_ee": gf.place_err_from_poses(o["eef_pos"], o["eef_quat"], o["box_positions"], **pol.place_err_ee.params),
    }
    for name, ref in refs.items():
        max_err = max(max_err, (o[name] - ref).abs().max().item())
    n += 1
    if t % 50 == 0:
        print(f"t={t:4d} grasp_err_ee={o['grasp_err_ee'][0].cpu().numpy().round(3)} place_err_ee={o['place_err_ee'][0].cpu().numpy().round(3)}", flush=True)
    obs, *_ = env.step(actions[t : t + 1])

shapes = {k: tuple(obs["policy"][k].shape) for k in ("grasp_err", "grasp_err_ee", "place_err_ee")}
print(f"\n[check] shapes={shapes}, steps={n}, max |env - ref| = {max_err:.2e}", flush=True)
print("[check] PASS" if max_err < 1e-4 else "[check] FAIL", flush=True)
env.close()
simulation_app.close()
