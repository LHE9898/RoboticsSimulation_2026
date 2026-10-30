# How well the EE reaches the two relative-coordinate targets (grasp target G*, place target P), seen from the EE.
# Same event definitions for human demos, Mimic data and policy rollouts:
#   grasp moment t_g = last open->close switch at most 100 steps before the truck first rises > 2 cm
#   release moment t_r = first open(+1) command after the truck was lifted > 2 cm
#
#   python tools_c/target_reach.py --hdf5 datasets/team10_lee_s3.hdf5 datasets/team10_lee_s3_mimic_500.hdf5 \
#                                  --rollouts outputs/rollouts/geee_s101 outputs/rollouts/geee_s111 ...
import argparse
import glob
import importlib.util
import os

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "grasp_feat", "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/mdp/grasp_feat.py"
)
gf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gf)
CFG = "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot"
LIFT, WIN = 0.02, 100


def events(a, z):
    z = z - z[0]
    lifted = np.where(z > LIFT)[0]
    # the close that actually lifted the truck: last open->close switch before lift onset, at most WIN steps before
    closes = np.where(np.diff(np.r_[1.0, a[:, 6]]) < 0)[0]
    closes = closes[(closes <= lifted[0]) & (closes >= lifted[0] - WIN)] if len(lifted) else []
    t_g = int(closes[-1]) if len(closes) else None
    op = np.where(a[lifted[0] :, 6] > 0)[0] if len(lifted) else []
    t_r = int(lifted[0] + op[0]) if len(op) else None
    return t_g, t_r


def episode(o, a, success):
    T = lambda k: torch.tensor(np.asarray(o[k]), dtype=torch.float64)  # noqa: E731
    ge = gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                 os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()
    pe = gf.place_err_from_poses(T("eef_pos"), T("eef_quat"), T("box_positions"),
                                 os.path.join(CFG, "place_offset.json"), "ee").numpy()
    pb = gf.place_err_from_poses(T("eef_pos"), T("eef_quat"), T("box_positions"),
                                 os.path.join(CFG, "place_offset.json"), "base").numpy()
    t_g, t_r = events(np.asarray(a), np.asarray(o["toy_truck_positions"])[:, 2])
    return {
        "success": bool(success),
        "g_p": ge[t_g, :3] if t_g is not None else None,
        "g_R": float(np.degrees(np.linalg.norm(ge[t_g, 3:]))) if t_g is not None else None,
        "g_min": float(np.linalg.norm(ge[:, :3], axis=1).min()),  # closest approach over the episode
        "p_p": pe[t_r] if t_r is not None else None,
        "p_xy": float(np.linalg.norm(pb[t_r, :2])) if t_r is not None else None,  # horizontal distance (base xy)
        "p_z": float(pb[t_r, 2]) if t_r is not None else None,  # height error (base z)
    }


def load_hdf5(p):
    import h5py

    with h5py.File(p, "r") as f:
        return [episode(g["obs"], g["actions"][()], g.attrs.get("success", True)) for g in f["data"].values()]


def load_rollouts(p):
    return [episode(d, d["actions"], d["success"]) for d in (dict(np.load(f)) for f in sorted(glob.glob(f"{p}/rollout_*.npz")))]


def report(name, eps):
    n = len(eps)
    G = [e for e in eps if e["g_p"] is not None]
    P = [e for e in eps if e["p_p"] is not None]
    gp = np.array([e["g_p"] for e in G]) * 100
    pp = np.array([e["p_p"] for e in P]) * 100
    med = lambda x: float(np.median(x)) if len(x) else np.nan  # noqa: E731
    print(f"| {name} | {n} | {len(G) / n:.2f} | {np.mean(np.linalg.norm(gp, axis=1) < 3) if len(G) else np.nan:.2f}"
          f" | {med(np.linalg.norm(gp, axis=1)):.1f} | ({', '.join(f'{v:+.1f}' for v in gp.mean(0))})"
          f" | {med([e['g_R'] for e in G]):.1f} | {len(P) / n:.2f}"
          f" | {np.mean([e['p_xy'] < 0.10 for e in P]) if P else np.nan:.2f} | {med([e['p_xy'] * 100 for e in P]):.1f}"
          f" | {med([e['p_z'] * 100 for e in P]):+.1f} | ({', '.join(f'{v:+.1f}' for v in pp.mean(0))}) |")


ap = argparse.ArgumentParser()
ap.add_argument("--hdf5", nargs="*", default=[])
ap.add_argument("--rollouts", nargs="*", default=[])
ap.add_argument("--split", action="store_true", help="also report success / failure rollouts separately")
A = ap.parse_args()

hdr = ["source", "n", "grasped", "‖e_G,p‖<3cm @t_g", "‖e_G,p‖ @t_g (cm, med)", "mean e_G,p^E @t_g (cm)", "‖e_G,R‖ @t_g (deg, med)",
       "released", "xy(E,P)<10cm @t_r", "xy(E,P) @t_r (cm, med)", "z(E,P) @t_r (cm, med)", "mean e_P^E @t_r (cm)"]
print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
for p in A.hdf5:
    report(os.path.basename(p), load_hdf5(p))
groups = {}
for p in A.rollouts:
    groups.setdefault(os.path.basename(os.path.normpath(p)).rsplit("_s", 1)[0], []).extend(load_rollouts(p))
for name, eps in groups.items():
    report(name, eps)
    if A.split:
        report(f"{name} success", [e for e in eps if e["success"]])
        report(f"{name} failure", [e for e in eps if not e["success"]])
