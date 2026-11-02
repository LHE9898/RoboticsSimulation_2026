# Pick the bc_event gripper thresholds on the TRAINING demos (CPU only) and write <run>/event_tau.json,
# which tools_c/play_record.py applies automatically (bc_event.apply_event_tau).
#
# Teacher-forced along every demo: for each true switch s, the rule (p > tau for `consecutive` steps) is
#   on time  if it first fires inside the labeled window [first positive step before s, s + TOL]
#   early    if it fires before that window, miss otherwise.
# tau per head = argmax on-time rate over GRID (ties -> fewer early).
#
#   CUDA_VISIBLE_DEVICES= python tools_c/calibrate_event_tau.py --ckpt <run>/models/model_epoch_2000.pth [--data <hdf5>]
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bc_event  # noqa: F401, E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402

GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99]
TOL = 10


def fires(p, tau, m):
    f = (p > tau).astype(int)
    return np.convolve(f, np.ones(m, int), "full")[: len(f)] >= m


ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", default=None, help="defaults to train.data of the checkpoint config")
ap.add_argument("--threads", type=int, default=6)
A = ap.parse_args()
torch.set_num_threads(A.threads)

policy, ckpt = FileUtils.policy_from_checkpoint(ckpt_path=A.ckpt, device=torch.device("cpu"), verbose=False)
algo = policy.policy
label_key, m = algo.algo_config.event.label_key, algo.algo_config.event.consecutive
data = A.data or os.path.expanduser(algo.global_config.train.data)

demos = []
with h5py.File(data, "r") as f:
    for name in f["data"]:
        g = f["data"][name]
        obs = {k: torch.as_tensor(g["obs"][k][()], dtype=torch.float32) for k in algo.event_keys}
        x = algo._flat(obs)
        dx = torch.cat([torch.zeros_like(x[:1]), x[1:] - x[:-1]])
        c = g["grip_state"][:, 0]
        p = algo.event_prob(x, dx, torch.as_tensor(c)).numpy()
        demos.append((p, c.astype(bool), g[label_key][:, 0] > 0))


def outcomes(tau):
    res = {0: [], 1: []}
    for p, c, lab in demos:
        sw = np.where(np.diff(c.astype(int)) != 0)[0]  # c changes after step s -> the switch command was at s
        for i, s in enumerate(sw):
            h = int(c[s])
            lo = sw[i - 1] + 1 if i else 0
            hi = min(s + TOL, sw[i + 1] if i + 1 < len(sw) else len(p) - 1)
            w = s
            while w - 1 >= lo and lab[w - 1]:
                w -= 1
            tr = np.where(fires(p[lo : hi + 1], tau[h], m))[0]
            res[h].append("miss" if not len(tr) else ("early" if lo + tr[0] < w else "on"))
    return res


best, table = [], {}
for h, name in ((0, "close"), (1, "open")):
    rows = []
    for g in GRID:
        r = outcomes((g, g))[h]
        rows.append((g, np.mean([o == "on" for o in r]), np.mean([o == "early" for o in r]), np.mean([o == "miss" for o in r])))
    table[name] = rows
    best.append(max(rows, key=lambda t: (t[1], -t[2]))[0])
    print(f"{name}: " + ", ".join(f"{g}: on {on:.2f} / early {e:.2f} / miss {mi:.2f}" for g, on, e, mi in rows))

final = outcomes(tuple(best))
out = {
    "tau": best, "consecutive": m, "label_key": label_key, "data": data, "n_demos": len(demos),
    "on_time": {n: float(np.mean([o == "on" for o in final[h]])) for h, n in ((0, "close"), (1, "open"))},
    "early": {n: float(np.mean([o == "early" for o in final[h]])) for h, n in ((0, "close"), (1, "open"))},
    "miss": {n: float(np.mean([o == "miss" for o in final[h]])) for h, n in ((0, "close"), (1, "open"))},
    "grid": table,
}
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(A.ckpt))), "event_tau.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"tau (close, open) = {best} -> {path}")
