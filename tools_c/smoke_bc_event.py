# CPU smoke test for the "bc_event" algo without Isaac Sim (train.py launches the simulator, which uses the GPU):
# config -> dataset -> a few training steps -> checkpoint -> reload -> rollout-style get_action -> threshold calibration.
#
#   CUDA_VISIBLE_DEVICES= python tools_c/smoke_bc_event.py --config <bc_event_*.json> --data datasets/mimic_500_evt.hdf5
import argparse
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bc_event  # noqa: F401, E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
import robomimic.utils.train_utils as TrainUtils  # noqa: E402
from robomimic.algo import algo_factory  # noqa: E402
from robomimic.config import config_factory  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--out", default="/tmp/bc_event_smoke")
A = ap.parse_args()
torch.set_num_threads(6)
torch.manual_seed(0)

ext = json.load(open(A.config))
config = config_factory(ext["algo_name"])
with config.values_unlocked():
    config.update(ext)
    config.train.data = A.data
    config.train.cuda = False
    config.train.num_data_workers = 0
    config.train.batch_size = 16
ObsUtils.initialize_obs_utils_with_config(config)
shape_meta = FileUtils.get_shape_metadata_from_dataset(dataset_path=A.data, all_obs_keys=config.all_obs_keys, verbose=False)
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=A.data)
dev = torch.device("cpu")
model = algo_factory(config.algo_name, config, obs_key_shapes=shape_meta["all_shapes"], ac_dim=shape_meta["ac_dim"], device=dev)
print(type(model).__name__, "| GMM ac_dim", model.nets["policy"].ac_dim, "| event in", model.nets["event"].mu.numel() // 2,
      "| pos_weight", model.nets["event"].pos_weight.tolist(), "| optimizers", list(model.optimizers))

trainset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
loader = DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
b = next(iter(loader))
print("batch:", {k: tuple(v.shape) for k, v in b.items() if torch.is_tensor(v)}, "| obs keys", sorted(b["obs"]))
log = TrainUtils.run_epoch(model=model, data_loader=loader, epoch=1, num_steps=A.steps)
print("epoch log:", {k: round(v, 4) for k, v in log.items() if k in ("Loss", "Event_Loss", "Log_Likelihood")})

os.makedirs(os.path.join(A.out, "models"), exist_ok=True)
ckpt = os.path.join(A.out, "models", "model_epoch_1.pth")
TrainUtils.save_model(model=model, config=config, env_meta=env_meta, shape_meta=shape_meta, ckpt_path=ckpt)
policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=dev, verbose=False)
same = all(torch.equal(a, b) for a, b in zip(model.nets["event"].state_dict().values(), policy.policy.nets["event"].state_dict().values()))
print("reloaded:", type(policy.policy).__name__, "| event weights identical:", same)

# rollout-style calls on a demo: actions must be 7D with a +-1 gripper from the event head
policy.start_episode()
with h5py.File(A.data, "r") as f:
    g = f["data/demo_0"]
    acts, ps = [], []
    for t in range(len(g["actions"])):
        acts.append(policy({k: g["obs"][k][t] for k in config.all_obs_keys}))
        ps.append(policy.policy.last_event_p)
acts = np.array(acts)
print("rollout actions", acts.shape, "| gripper values", np.unique(acts[:, 6]), "| event p range", np.nanmin(ps), np.nanmax(ps),
      "| switches", int(np.abs(np.diff((acts[:, 6] < 0).astype(int))).sum()))
policy.start_episode()
print("after reset: closed =", policy.policy._closed, "| prev_x is None:", policy.policy._prev_x is None)

subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrate_event_tau.py"), "--ckpt", ckpt], check=True)
print("apply_event_tau ->", bc_event.apply_event_tau(policy, ckpt))
