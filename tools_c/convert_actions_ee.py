# Express the arm actions of a robomimic hdf5 in end-effector axes (for Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-EEAct-v0).
# CPU only.
#
# Dataset actions are RMPFlow relative deltas in base axes, applied to the pose observed at the same step:
#   p_des = p_t + dp,  R_des = exp(w) R_t   (obs[t] is the state before action t; checked on the demos: cos(dp_t, p_{t+1}-p_t)
#   0.67 vs 0.18 for the previous move, and w_t agrees with R_{t+1} R_t^T, not R_t^T R_{t+1})
# so the EE-axes action is dp_E = R_t^T dp, w_E = R_t^T w (mdp/ee_frame_actions.py applies R_t back). Gripper column unchanged.
# Everything else (obs, labels, file attrs) is copied as is.
#
#   python tools_c/convert_actions_ee.py --input datasets/mimic_500_evt.hdf5 --output datasets/mimic_500_evt_ee.hdf5
import argparse
import shutil

import h5py
import numpy as np


def quat2mat(q):
    """(w, x, y, z) -> rotation matrix, shape (..., 3, 3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def to_ee(actions, eef_quat, inverse=False):
    R = quat2mat(eef_quat)
    if not inverse:
        R = np.swapaxes(R, -1, -2)
    out = actions.copy()
    out[:, 0:3] = np.einsum("tij,tj->ti", R, actions[:, 0:3])
    out[:, 3:6] = np.einsum("tij,tj->ti", R, actions[:, 3:6])
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
A = ap.parse_args()

shutil.copyfile(A.input, A.output)
err = 0.0
with h5py.File(A.output, "r+") as f:
    for k in f["data"]:
        g = f["data"][k]
        a, q = g["actions"][()], g["obs"]["eef_quat"][()]
        ae = to_ee(a, q)
        err = max(err, float(np.abs(to_ee(ae, q, inverse=True) - a).max()))
        g["actions"][...] = ae.astype(g["actions"].dtype)
    f["data"].attrs["action_frame"] = "ee"
    n = len(f["data"])
print(f"{n} demos -> {A.output} (round trip max |a - R R^T a| = {err:.2e})")
