# Compute the grasp target frame G (fixed to the toy truck) from human demos.
#
#   ./isaaclab.sh -p tools_c/compute_grasp_offset.py --input datasets/team10_lee_s3.hdf5
#
# Output (default): source/isaaclab_tasks/.../place/config/agibot/grasp_offset.json
#   p_bar = ^T p_G, R_bar = ^T R_G, approach_axis = a  (see imitation_pland.md 4.1)
import argparse
import json
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(
    REPO,
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/grasp_offset.json",
)


def Rq(q):
    return Rot.from_quat(np.asarray(q, float)[..., [1, 2, 3, 0]]).as_matrix()  # (w,x,y,z) -> scipy (x,y,z,w)


def geo(A, B):
    return float(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


def flip(R, a):
    F = -np.eye(3)
    F[a, a] = 1
    return R @ F


def svd_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, 0))
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def grasp_step(g, z, lift=0.02, ahead=100):
    """First open(+1) -> close(-1) switch after which the truck is actually lifted."""
    for t in np.where((g[:-1] > 0) & (g[1:] < 0))[0] + 1:
        if z[t : t + ahead].max() - z[t] > lift:
            return int(t)


ap = argparse.ArgumentParser()
ap.add_argument("--input", nargs="+", required=True, help="human demo hdf5 file(s)")
ap.add_argument("--out", default=DEFAULT_OUT)
A = ap.parse_args()

P, Rs, Re, used = [], [], [], []
for fn in A.input:
    with h5py.File(fn, "r") as f:
        for k in sorted(f["data"], key=lambda s: int(s.split("_")[1])):
            o = f["data"][k]["obs"]
            t = grasp_step(f["data"][k]["actions"][:, 6], o["toy_truck_positions"][:, 2])
            if t is None:
                print(f"[skip] {os.path.basename(fn)}/{k}: no lifted grasp found")
                continue
            Ro, Ree = Rq(o["toy_truck_orientations"][t]), Rq(o["eef_quat"][t])
            P.append(Ro.T @ (o["eef_pos"][t] - o["toy_truck_positions"][t]))
            Rs.append(Ro.T @ Ree)
            Re.append(Ree)
            used.append(f"{os.path.basename(fn)}/{k}@{t}")
P, Rs, Re = map(np.array, (P, Rs, Re))

# approach axis: E axis most parallel to base z at the grasp moment
a = int(np.argmax(np.abs(Re[:, 2, :]).mean(0)))

# fold the 180 deg symmetry about the approach axis, then average
ref = Rs[0]
for _ in range(5):
    fl = np.array([geo(ref, flip(R, a)) < geo(ref, R) for R in Rs])
    F = np.array([flip(R, a) if x else R for R, x in zip(Rs, fl)])
    ref = svd_mean(F)

out = {
    "p_bar": P.mean(0).tolist(),
    "p_std": P.std(0).tolist(),
    "R_bar": ref.tolist(),
    "approach_axis": a,
    "n_used": len(P),
    "minority_orientation": int(min(fl.sum(), (~fl).sum())),
    "ang_raw_deg": np.degrees([geo(ref, R) for R in Rs]).round(1).tolist(),
    "ang_folded_deg": np.degrees([geo(ref, R) for R in F]).round(1).tolist(),
    "sources": used,
}
os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
with open(A.out, "w") as f:
    json.dump(out, f, indent=2)
print(f"saved -> {A.out}")
print({k: out[k] for k in ["p_bar", "p_std", "approach_axis", "n_used", "minority_orientation"]})
print("ang_raw_deg   :", out["ang_raw_deg"])
print("ang_folded_deg:", out["ang_folded_deg"])
