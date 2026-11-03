# Write a copy of a bc_event checkpoint whose stored config already holds the calibrated gripper thresholds (<run>/event_tau.json)
# and the close rule, so the stock scripts/imitation_learning/robomimic/play.py runs the policy exactly as tools_c/play_record.py
# evaluated it (play_record applies the same two settings from event_tau.json and --close_rule). Weights are unchanged. CPU only.
#
#   python tools_c/make_play_ckpt.py --ckpt <run>/models/model_epoch_2000.pth --close_rule 0.04,0.02 --action_frame ee \
#       --out <run>/models/model_eez2.pth      (eez2: run with the stock play.py in ...-RmpFlow-v0)
import argparse
import json
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--close_rule", default=None, help="'xy_max,z_max' in m; omit to keep closing with the close head")
ap.add_argument("--action_frame", default=None, choices=["base", "ee"],
                help="'ee': the policy rotates its EE-axes arm action to base axes itself -> run it in ...-RmpFlow-v0 (not -EEAct-v0)")
ap.add_argument("--out", required=True)
A = ap.parse_args()

ck = torch.load(A.ckpt, map_location="cpu", weights_only=False)
cfg = json.loads(ck["config"])
assert cfg["algo_name"] == "bc_event", f"not a bc_event checkpoint: {cfg['algo_name']}"
tau_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(A.ckpt))), "event_tau.json")
with open(tau_path) as f:
    cfg["algo"]["event"]["tau"] = list(json.load(f)["tau"])
cfg["algo"]["event"]["close_rule"] = [float(v) for v in A.close_rule.split(",")] if A.close_rule else []
if A.action_frame:
    cfg["algo"]["event"]["action_frame"] = A.action_frame
ck["config"] = json.dumps(cfg, indent=4)
torch.save(ck, A.out)
print(f"{A.out}: tau (close, open) = {cfg['algo']['event']['tau']}, close_rule = {cfg['algo']['event']['close_rule']}, "
      f"action_frame = {cfg['algo']['event'].get('action_frame', 'base')}")
