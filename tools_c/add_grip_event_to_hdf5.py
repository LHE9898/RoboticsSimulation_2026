# Add gripper state / event labels to a robomimic hdf5 for the "bc_event" algo (tools_c/bc_event.py). CPU only.
#
# Per demo (shape (T, 1), float32, stored next to "actions" so robomimic loads them through train.dataset_keys):
#   grip_state              commanded gripper state before step t (0 open, 1 closed; episodes start open)
#   grip_event_switch       1 for t in [s-5, s] before every switch s (stage A label)
#   grip_event_ready        switch label, extended back while the EE is already where it switches
#                           close: |e_G,p| <= |e_G,p(s)| + 1 cm; open: EE-box xy <= xy(s) + 1 cm and EE height within 1 cm of z(s)
#                           (at most READY_MAX steps, never across the previous switch)
#   grip_event_ready_inbox  ready label, but open positives only for releases with truck-box xy < 10 cm
# File attrs: grip_event_obs_keys, grip_event_mu, grip_event_sd (normalization of [x, dx]) and <label>_pos_weight (per head).
#
#   python tools_c/add_grip_event_to_hdf5.py --input datasets/mimic_500_rel.hdf5 --output datasets/mimic_500_evt.hdf5
import argparse
import json
import shutil

import h5py
import numpy as np

KEYS = sorted(["eef_pos", "eef_quat", "gripper_pos", "toy_truck_positions", "toy_truck_orientations",
               "box_positions", "box_orientations", "grasp_err_ee", "place_err_ee"])  # gepl obs, sorted like bc_event.py
WIN, READY_MAX, SLACK, IN_BOX = 5, 40, 0.01, 0.10
LABELS = ["grip_event_switch", "grip_event_ready", "grip_event_ready_inbox"]


def labels(g):
    a, o = g["actions"][()], g["obs"]
    cmd = a[:, 6] < 0
    c = np.r_[False, cmd[:-1]]
    sw = np.where(cmd != c)[0]
    ep = np.linalg.norm(o["grasp_err_ee"][:, :3], axis=1)
    eef, box, tr = o["eef_pos"][()], o["box_positions"][()], o["toy_truck_positions"][()]
    xy_eb = np.linalg.norm(box[:, :2] - eef[:, :2], axis=1)
    xy_tb = np.linalg.norm(tr[:, :2] - box[:, :2], axis=1)
    out = {k: np.zeros(len(a), np.float32) for k in LABELS}
    for i, s in enumerate(sw):
        lo = sw[i - 1] + 1 if i else 0
        base = np.arange(max(lo, s - WIN), s + 1)
        out["grip_event_switch"][base] = 1
        t = s
        if not c[s]:  # close event
            ok = lambda u: ep[u] <= ep[s] + SLACK  # noqa: E731
        else:  # open event
            ok = lambda u: xy_eb[u] <= xy_eb[s] + SLACK and abs(eef[u, 2] - eef[s, 2]) <= SLACK  # noqa: E731
        while t - 1 >= max(lo, s - READY_MAX) and ok(t - 1):
            t -= 1
        ready = np.arange(min(t, base[0]), s + 1)
        out["grip_event_ready"][ready] = 1
        if not c[s] or xy_tb[s] < IN_BOX:
            out["grip_event_ready_inbox"][ready] = 1
    x = np.concatenate([o[k][()].reshape(len(a), -1) for k in KEYS], 1)
    dx = np.diff(x, axis=0, prepend=x[:1])
    return c.astype(np.float32), out, np.c_[x, dx]


ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
A = ap.parse_args()
shutil.copyfile(A.input, A.output)
Z, S, L = [], [], {k: [] for k in LABELS}
with h5py.File(A.output, "r+") as f:
    for name in f["data"]:
        g = f["data"][name]
        c, lab, z = labels(g)
        for k in ["grip_state"] + LABELS:
            if k in g:
                del g[k]
        g.create_dataset("grip_state", data=c[:, None])
        for k, v in lab.items():
            g.create_dataset(k, data=v[:, None])
            L[k].append(v)
        Z.append(z)
        S.append(c)
    Z, S = np.concatenate(Z), np.concatenate(S).astype(bool)
    f["data"].attrs["grip_event_obs_keys"] = json.dumps(KEYS)
    f["data"].attrs["grip_event_mu"] = Z.mean(0)
    f["data"].attrs["grip_event_sd"] = Z.std(0) + 1e-6
    print(f"{A.output}: {len(f['data'])} demos, {len(S)} steps, obs dim {Z.shape[1] // 2}, closed {S.mean():.3f}")
    for k in LABELS:
        y = np.concatenate(L[k]).astype(bool)
        pw = [float(((~y) & (S == h)).sum() / max((y & (S == h)).sum(), 1)) for h in (False, True)]
        f["data"].attrs[f"{k}_pos_weight"] = np.array(pw)
        print(f"  {k}: positive rate close-head {y[~S].mean():.4f}, open-head {y[S].mean():.4f} -> pos_weight {pw[0]:.1f}, {pw[1]:.1f}")
