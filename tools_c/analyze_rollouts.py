# Failure classification + metrics + pre-registered judgement (imitation_pland.md 8) from play_record.py output.
#
#   python tools_c/analyze_rollouts.py outputs/rollouts/base_s101                     # one run (baseline diagnosis)
#   python tools_c/analyze_rollouts.py --baseline outputs/rollouts/base_s101 outputs/rollouts/base_s111 \
#                                      --exp outputs/rollouts/ge_s101 outputs/rollouts/ge_s111 [--criteria grasp|frame|place]
import argparse
import glob
import os

import numpy as np

APPROACH_EP = 0.03  # approach failure: min ||e_p|| > 3 cm
LIFT = 0.02  # truck counts as lifted if it rises > 2 cm
BOX_XY = 0.10  # same xy threshold as the success term
IDLE = 0.01  # step counts as "stopped" if max |a[:6]| < IDLE (keyboard step = 0.05)
CLASSES = ["success", "approach", "grasp", "transport", "place"]


def analyze_rollout(d):
    a, e = d["actions"], d["grasp_err"]
    z = d["toy_truck_positions"][:, 2] - d["toy_truck_positions"][0, 2]
    xy = np.linalg.norm(d["toy_truck_positions"][:, :2] - d["box_positions"][:, :2], axis=1)
    ep, eR = np.linalg.norm(e[:, :3], axis=1), np.linalg.norm(e[:, 3:], axis=1)
    close = np.where(a[:, 6] < 0)[0]
    t_c = int(close[0]) if len(close) else None
    lifted = np.where(z > LIFT)[0]
    # release = first open command after the truck was lifted
    opened = np.where(a[lifted[0] :, 6] > 0)[0] if len(lifted) else []
    t_r = int(lifted[0] + opened[0]) if len(opened) else None

    if bool(d["success"]):
        cls = "success"
    elif ep.min() > APPROACH_EP:
        cls = "approach"
    elif t_c is None or z[t_c:].max() < LIFT:
        cls = "grasp"
    elif not (xy[lifted[0] :] < BOX_XY).any():
        cls = "transport"
    else:
        cls = "place"

    # approach-phase alignment between commanded translation and position error (both in base frame)
    pre = slice(0, t_c if t_c is not None else len(a))
    ax, epv = a[pre, :3], e[pre, :3]
    m = (np.linalg.norm(ax, axis=1) > 1e-3) & (np.linalg.norm(epv, axis=1) > 1e-3)
    cos = (ax[m] * epv[m]).sum(1) / (np.linalg.norm(ax[m], axis=1) * np.linalg.norm(epv[m], axis=1))
    return {
        "cls": cls,
        "eR0_deg": float(np.degrees(eR[0])),
        "ep_close": float(ep[t_c]) if t_c is not None else np.nan,
        "eR_close_deg": float(np.degrees(eR[t_c])) if t_c is not None else np.nan,
        "cos_approach": float(cos.mean()) if m.any() else np.nan,
        "idle": float((np.abs(a[:, :6]).max(1) < IDLE).mean()),
        # released outside the success radius (and not a success)
        "release_out": float(cls != "success" and t_r is not None and xy[t_r] >= BOX_XY),
        "xy_rel": float(xy[t_r]) if t_r is not None else np.nan,
        # EE xy speed over the 10 steps (0.5 s at 20 Hz) before release
        "v_rel": float(np.linalg.norm(d["eef_pos"][t_r, :2] - d["eef_pos"][max(t_r - 10, 0), :2]) * 2)
        if t_r is not None
        else np.nan,
    }


def load_run(path):
    files = sorted(glob.glob(os.path.join(path, "rollout_*.npz")))
    assert files, f"no rollout_*.npz in {path}"
    return [analyze_rollout(dict(np.load(f))) for f in files]


def summarize(rows):
    n = len(rows)
    s = {c: sum(r["cls"] == c for r in rows) / n for c in CLASSES}
    for k in ["ep_close", "eR_close_deg", "cos_approach", "idle", "release_out"]:
        s[k] = float(np.nanmean([r[k] for r in rows]))
    for k in ["xy_rel", "v_rel"]:  # medians: heavy-tailed
        s[k] = float(np.nanmedian([r[k] for r in rows]))
    s["n"] = n
    return s


def print_runs(runs):
    hdr = ["run", "n", "succ", "appr", "grasp", "transp", "place", "|e_p|@close cm", "|e_R|@close deg", "cos_appr", "idle",
           "rel_out", "xy@rel cm (med)", "v@rel cm/s (med)"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for name, rows in runs.items():
        s = summarize(rows)
        print(
            f"| {name} | {s['n']} | {s['success']:.2f} | {s['approach']:.2f} | {s['grasp']:.2f} | {s['transport']:.2f}"
            f" | {s['place']:.2f} | {s['ep_close'] * 100:.2f} | {s['eR_close_deg']:.1f} | {s['cos_approach']:.3f}"
            f" | {s['idle']:.2f} | {s['release_out']:.2f} | {s['xy_rel'] * 100:.1f} | {s['v_rel'] * 100:.1f} |"
        )


def bin_success(rows, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["eR0_deg"] < hi]
        out.append((len(sel), np.mean([r["cls"] == "success" for r in sel]) if sel else np.nan))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="*", help="rollout dirs to summarize")
ap.add_argument("--baseline", nargs="+", default=[])
ap.add_argument("--exp", nargs="+", default=[])
ap.add_argument("--criteria", default="grasp", choices=["grasp", "frame", "place"],
                help="grasp: grasp_err(base) vs 23D | frame: grasp_err_ee vs grasp_err | place: +place_err_ee vs grasp_err_ee")
A = ap.parse_args()

dirs = A.runs + A.baseline + A.exp
runs = {os.path.basename(os.path.normpath(p)): load_run(p) for p in dirs}
print_runs(runs)

all_rows = [r for rows in runs.values() for r in rows]
# the start EE pose is fixed, so ||e_R(0)|| only spans a narrow range -> bin by pooled quartiles
edges = np.percentile([r["eR0_deg"] for r in all_rows], [0, 25, 50, 75, 100])
edges[-1] += 1e-6
print("\nsuccess rate by initial ||e_R(0)|| quartile (deg):")
print("| run | " + " | ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(edges[:-1], edges[1:])) + " |\n|" + "---|" * len(edges))
for name, rows in runs.items():
    print(f"| {name} | " + " | ".join(f"{s:.2f} (n={n})" for n, s in bin_success(rows, edges)) + " |")

if A.baseline and A.exp:
    B = [summarize(runs[os.path.basename(os.path.normpath(p))]) for p in A.baseline]
    E = [summarize(runs[os.path.basename(os.path.normpath(p))]) for p in A.exp]
    mean = lambda S, k: float(np.mean([s[k] for s in S]))  # noqa: E731
    std = lambda S, k: float(np.std([s[k] for s in S]))  # noqa: E731
    print(f"\nbaseline success {mean(B, 'success'):.3f} ± {std(B, 'success'):.3f} | "
          f"grasp_err success {mean(E, 'success'):.3f} ± {std(E, 'success'):.3f}")

    # 3) improvement concentrated where the initial rotation error is large (split at pooled median)
    med = float(np.median([r["eR0_deg"] for r in all_rows]))
    def half_rate(paths, hi):  # noqa: E301
        rows = [r for p in paths for r in runs[os.path.basename(os.path.normpath(p))]]
        sel = [r["cls"] == "success" for r in rows if (r["eR0_deg"] >= med) == hi]
        return float(np.mean(sel)) if sel else np.nan
    d_hi = half_rate(A.exp, True) - half_rate(A.baseline, True)
    d_lo = half_rate(A.exp, False) - half_rate(A.baseline, False)

    beats = min(s["success"] for s in E) > max(s["success"] for s in B)
    gain = mean(E, "success") - mean(B, "success")
    crit = [
        ("1. success +10%p and both exp runs > both baseline runs", gain >= 0.10 and beats),
        ("2. grasp-failure ratio down and |e_R|@close down",
         mean(E, "grasp") < mean(B, "grasp") and mean(E, "eR_close_deg") < mean(B, "eR_close_deg")),
        (f"3. gain larger for ||e_R(0)|| >= {med:.0f} deg ({d_hi:+.3f}) than below ({d_lo:+.3f})", d_hi > d_lo),
        ("4. approach-failure ratio down and cos_approach up (only if approach was the baseline cause)",
         mean(E, "approach") < mean(B, "approach") and mean(E, "cos_approach") > mean(B, "cos_approach")),
    ]
    if A.criteria == "frame":  # imitation_pland_v2.md 3.1
        crit = [
            ("F1. success +5%p and both exp runs > both baseline runs", gain >= 0.05 and beats),
            ("F2. approach-failure ratio down", mean(E, "approach") < mean(B, "approach")),
            ("F3. |e_R|@close down", mean(E, "eR_close_deg") < mean(B, "eR_close_deg")),
        ]
    elif A.criteria == "place":  # imitation_pland_v2.md 3.2
        crit = [
            ("P1. success +5%p and both exp runs > both baseline runs", gain >= 0.05 and beats),
            ("P2. released-outside-box ratio down", mean(E, "release_out") < mean(B, "release_out")),
            ("P3. median truck-box xy at release down", mean(E, "xy_rel") < mean(B, "xy_rel")),
            ("P4. median EE speed at release down", mean(E, "v_rel") < mean(B, "v_rel")),
        ]
    print(f"\npre-registered criteria ({A.criteria}):")
    for txt, ok in crit:
        print(f"  [{'PASS' if ok else 'FAIL'}] {txt}")
