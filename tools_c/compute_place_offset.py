# Compute the place target P (fixed to the box) from human demos: P = box center + h * z_B.
#
#   ./isaaclab.sh -p tools_c/compute_place_offset.py --input datasets/team10_lee_s3.hdf5
#
# Output (default): source/isaaclab_tasks/.../place/config/agibot/place_offset.json
#   h = mean EE height above the box at the release moment. The xy target is the box center itself
#   (the success term is defined w.r.t. the box center), the measured xy spread is only reported.
import argparse
import json
import os

import h5py
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(
    REPO,
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/place_offset.json",
)


def release_step(g, z, lift=0.02):
    """First open(+1) command after the truck has been lifted."""
    lifted = np.where(z - z[0] > lift)[0]
    if not len(lifted):
        return None
    op = np.where(g[lifted[0] :] > 0)[0]
    return int(lifted[0] + op[0]) if len(op) else None


ap = argparse.ArgumentParser()
ap.add_argument("--input", nargs="+", required=True, help="human demo hdf5 file(s)")
ap.add_argument("--out", default=DEFAULT_OUT)
A = ap.parse_args()

D, used = [], []
for fn in A.input:
    with h5py.File(fn, "r") as f:
        for k in sorted(f["data"], key=lambda s: int(s.split("_")[1])):
            o = f["data"][k]["obs"]
            t = release_step(f["data"][k]["actions"][:, 6], o["toy_truck_positions"][:, 2])
            if t is None:
                print(f"[skip] {os.path.basename(fn)}/{k}: no release after lift found")
                continue
            D.append(o["eef_pos"][t] - o["box_positions"][t])  # base frame
            used.append(f"{os.path.basename(fn)}/{k}@{t}")
D = np.array(D)

out = {
    "h": float(D[:, 2].mean()),
    "h_std": float(D[:, 2].std()),
    "xy_mean": D[:, :2].mean(0).tolist(),
    "xy_std": D[:, :2].std(0).tolist(),
    "n_used": len(D),
    "sources": used,
}
os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
with open(A.out, "w") as f:
    json.dump(out, f, indent=2)
print(f"saved -> {A.out}")
print({k: out[k] for k in ["h", "h_std", "xy_mean", "xy_std", "n_used"]})
