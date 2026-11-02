# Gripper decision diagnosis (CPU only): when does the gripper switch, and where is the EE relative to the targets then.
#   1) switches per episode / short close-reopen flickers   2) EE->grasp target at every close event
#   3) EE->place target at every release                    4) overlap with failure classes
#   5) data side: closed/open balance and how rare the switching steps are
#
#   CUDA_VISIBLE_DEVICES= python tools_c/gripper_diag.py --hdf5 datasets/team10_lee_s3.hdf5 datasets/team10_lee_s3_mimic_500.hdf5 \
#       --rollouts outputs/rollouts/base_s101 outputs/rollouts/base_s111 outputs/rollouts/gepl_s101 outputs/rollouts/gepl_s111
import argparse
import glob
import os

import numpy as np

# reuse event / target definitions from the other analyzers
exec(open("tools_c/target_reach.py").read().split("def events")[0])  # gf, CFG, LIFT, WIN
exec(open("tools_c/analyze_rollouts.py").read().split("def load_run")[0].split("import numpy as np")[1])  # analyze_rollout

FLICKER = 10  # a closed segment shorter than this (0.5 s at 20 Hz) that never lifts the truck
NEAR, FAR = 0.03, 0.10


def segments(closed):
    """(start, end) of every closed run, end exclusive."""
    d = np.diff(np.r_[0, closed.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def episode(o, a, success):
    a = np.asarray(a)
    T = lambda k: torch.tensor(np.asarray(o[k]), dtype=torch.float64)  # noqa: E731
    ge = gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                 os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()
    pb = gf.place_err_from_poses(T("eef_pos"), T("eef_quat"), T("box_positions"),
                                 os.path.join(CFG, "place_offset.json"), "base").numpy()
    tr = np.asarray(o["toy_truck_positions"])
    z = tr[:, 2] - tr[0, 2]
    xy_tb = np.linalg.norm(tr[:, :2] - np.asarray(o["box_positions"])[:, :2], axis=1)
    closed = a[:, 6] < 0
    segs = segments(closed)
    closes = []
    for s, e in segs:
        lifted = bool((z[s : min(e + 1, len(z))] > LIFT).any())
        closes.append({"t": int(s), "len": int(e - s), "lift": lifted, "ep": float(np.linalg.norm(ge[s, :3])),
                       "eR": float(np.degrees(np.linalg.norm(ge[s, 3:])))})
    # releases = close->open switches while the truck is lifted
    rel = [{"t": int(e), "xy_EP": float(np.linalg.norm(pb[e, :2])), "z_EP": float(pb[e, 2]), "xy_TB": float(xy_tb[e])}
           for s, e in segs if e < len(z) and z[e] > LIFT]
    return {"success": bool(success), "n_switch": int(np.abs(np.diff(closed.astype(int))).sum()), "closes": closes,
            "rel": rel, "steps": len(a), "closed_frac": float(closed.mean()),
            "first_closed": bool(closed[0])}


def load_hdf5(p):
    import h5py

    with h5py.File(p, "r") as f:
        return [episode(g["obs"], g["actions"][()], True) for g in f["data"].values()]


def load_rollouts(p):
    out = []
    for f in sorted(glob.glob(f"{p}/rollout_*.npz")):
        d = dict(np.load(f))
        e = episode(d, d["actions"], d["success"])
        r = analyze_rollout(d)
        e["cls"], e["release_out"] = r["cls"], bool(r["release_out"])
        out.append(e)
    return out


def pct(x):
    return f"{100 * np.mean(x):.0f}%" if len(x) else "-"


def summarize(name, eps):
    C = [c for e in eps for c in e["closes"]]
    flick = [c for c in C if not c["lift"] and c["len"] < FLICKER]
    miss = [c for c in C if not c["lift"]]
    eff = [c for c in C if c["lift"]]
    R = [r for e in eps for r in e["rel"]]
    first_rel = [e["rel"][0] for e in eps if e["rel"]]
    sw = [e["n_switch"] for e in eps]
    print(f"\n### {name} (n={len(eps)})")
    print(f"- switches/episode: median {np.median(sw):.0f}, mean {np.mean(sw):.1f}, max {max(sw)}; "
          f"episodes with >2 switches {pct([s > 2 for s in sw])}")
    print(f"- close events {len(C)} ({len(C) / len(eps):.2f}/ep): lifting {len(eff)}, not lifting {len(miss)} "
          f"(flicker <{FLICKER} steps {len(flick)}); episodes with >=1 non-lifting close {pct([any(not c['lift'] for c in e['closes']) for e in eps])}")
    for lab, L in [("lifting", eff), ("not lifting", miss)]:
        if L:
            ep = np.array([c["ep"] for c in L])
            print(f"  - {lab}: |e_G,p| median {np.median(ep) * 100:.1f} cm | <3cm {pct(ep < NEAR)} 3-10cm {pct((ep >= NEAR) & (ep < FAR))}"
                  f" >10cm {pct(ep >= FAR)} | |e_G,R| median {np.median([c['eR'] for c in L]):.1f} deg"
                  f" | closed length median {np.median([c['len'] for c in L]):.0f} steps")
    if first_rel:
        xy = np.array([r["xy_TB"] for r in first_rel])
        print(f"- first release after lift (n={len(first_rel)}): truck-box xy <10cm {pct(xy < 0.10)}, median {np.median(xy) * 100:.1f} cm;"
              f" EE-P xy median {np.median([r['xy_EP'] for r in first_rel]) * 100:.1f} cm, z median {np.median([r['z_EP'] for r in first_rel]) * 100:+.1f} cm;"
              f" releases/episode {len(R) / len(eps):.2f}")
    print(f"- closed-step fraction {np.mean([e['closed_frac'] for e in eps]):.2f}; switching steps / all steps "
          f"{sum(sw) / sum(e['steps'] for e in eps) * 100:.2f}%")
    if "cls" in eps[0]:
        print("- by outcome: | class | n | >=1 non-lifting close | non-lifting closes/ep | first close |e_G,p| median | release_out |")
        for cls in ["success", "approach", "grasp", "transport", "place"]:
            E = [e for e in eps if e["cls"] == cls]
            if not E:
                continue
            fc = [e["closes"][0]["ep"] for e in E if e["closes"]]
            print(f"  | {cls} | {len(E)} | {pct([any(not c['lift'] for c in e['closes']) for e in E])}"
                  f" | {np.mean([sum(not c['lift'] for c in e['closes']) for e in E]):.2f}"
                  f" | {np.median(fc) * 100 if fc else np.nan:.1f} cm | {pct([e['release_out'] for e in E])} |")
        E = [e for e in eps if not e["success"]]
        g_err = [e for e in E if e["cls"] in ("grasp",) or (e["cls"] == "approach" and any(c["ep"] < FAR for c in e["closes"]))]
        r_err = [e for e in E if e["release_out"]]
        print(f"- failures {len(E)}: gripper-timing related = grasp class or approach with a close <10cm "
              f"({len(g_err)}) + released outside box ({len(r_err)}) -> {len(g_err) + len(r_err)} ({(len(g_err) + len(r_err)) / len(E) * 100:.0f}% of failures)")


ap = argparse.ArgumentParser()
ap.add_argument("--hdf5", nargs="*", default=[])
ap.add_argument("--rollouts", nargs="*", default=[])
A = ap.parse_args()
for p in A.hdf5:
    summarize(os.path.basename(p), load_hdf5(p))
groups = {}
for p in A.rollouts:
    groups.setdefault(os.path.basename(os.path.normpath(p)).rsplit("_s", 1)[0], []).extend(load_rollouts(p))
for name, eps in groups.items():
    summarize(name, eps)
