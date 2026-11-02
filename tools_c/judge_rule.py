# Judgement for evr + geometric close rule (pre-registered in results/event_clf/plan_XY.md). CPU only.
#
#   CUDA_VISIBLE_DEVICES= python tools_c/judge_rule.py --gepl outputs/rollouts/gepl_s101 outputs/rollouts/gepl_s111 \
#       --evr outputs/rollouts/evr_s101 outputs/rollouts/evr_s111 --exp outputs/rollouts/evrz_s101 outputs/rollouts/evrz_s111
import argparse
import glob
import os
import sys

import numpy as np

sys.argv, _argv = sys.argv[:1], sys.argv
exec(open("tools_c/gripper_diag.py").read().split("ap = argparse.ArgumentParser()")[0])  # episode, analyze_rollout, gf, CFG
sys.argv = _argv


def metrics(run):
    rows, stuck_end = [], []
    for fn in sorted(glob.glob(f"{run}/rollout_*.npz")):
        d = dict(np.load(fn))
        e = episode(d, d["actions"], d["success"])
        T = lambda k: torch.tensor(np.asarray(d[k]), dtype=torch.float64)  # noqa: E731
        ge = gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                     os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()
        p3 = np.linalg.norm(ge[:, :3], axis=1)
        stuck = not e["closes"] and p3.min() < 0.03
        if stuck:  # where did the EE wait? (closest step)
            t = int(np.argmin(p3))
            stuck_end.append((np.linalg.norm(ge[t, :2]) * 100, ge[t, 2] * 100))
        rows.append({"success": e["success"], "lifted": any(c["lift"] for c in e["closes"]), "stuck": stuck,
                     "miss_far": sum((not c["lift"]) and c["ep"] >= 0.03 for c in e["closes"]),
                     "miss": sum(not c["lift"] for c in e["closes"]), "multi": e["n_switch"] > 2,
                     "cls": analyze_rollout(d)["cls"]})
    return rows, stuck_end


ap = argparse.ArgumentParser()
ap.add_argument("--gepl", nargs="+", required=True)
ap.add_argument("--evr", nargs="+", required=True)
ap.add_argument("--exp", nargs="+", required=True)
A = ap.parse_args()

runs = {name: {os.path.basename(os.path.normpath(p)): metrics(p) for p in paths}
        for name, paths in (("gepl", A.gepl), ("evr", A.evr), ("evrz", A.exp))}
cls_names = ["success", "approach", "grasp", "transport", "place"]
print("| run | n | success | lifted | stuck (3 cm 안, 안 닫음) | 못 든 닫힘 (3 cm 밖) | 못 든 닫힘 (전체) | 2번 넘게 바꿈 | "
      + " | ".join(cls_names) + " |")
print("|" + "---|" * (8 + len(cls_names)))
agg = {}
for name, rr in runs.items():
    for run, (rows, _) in rr.items():
        n = len(rows)
        print(f"| {run} | {n} | {np.mean([r['success'] for r in rows]):.3f} | {np.mean([r['lifted'] for r in rows]):.3f} | "
              f"{sum(r['stuck'] for r in rows)} | {sum(r['miss_far'] for r in rows)} | {sum(r['miss'] for r in rows)} | "
              f"{np.mean([r['multi'] for r in rows]):.2f} | " + " | ".join(str(sum(r['cls'] == c for r in rows)) for c in cls_names) + " |")
    allr = [r for rows, _ in rr.values() for r in rows]
    agg[name] = {"success": np.mean([r["success"] for r in allr]), "lifted": np.mean([r["lifted"] for r in allr]),
                 "stuck": sum(r["stuck"] for r in allr), "miss_far": sum(r["miss_far"] for r in allr),
                 "runs": [np.mean([r["success"] for r in rows]) for rows, _ in rr.values()]}
    print(f"| **{name} 합산** | {len(allr)} | {agg[name]['success']:.3f} | {agg[name]['lifted']:.3f} | {agg[name]['stuck']} | "
          f"{agg[name]['miss_far']} | {sum(r['miss'] for r in allr)} | {np.mean([r['multi'] for r in allr]):.2f} | "
          + " | ".join(str(sum(r['cls'] == c for r in allr)) for c in cls_names) + " |")

g, v, x = agg["gepl"], agg["evr"], agg["evrz"]
crit = [
    (f"G1. success {x['success']:.3f} > evr {v['success']:.3f}", x["success"] > v["success"]),
    (f"G2. lifted {x['lifted']:.3f} >= gepl {g['lifted']:.3f}", x["lifted"] >= g["lifted"]),
    (f"G3. stuck {x['stuck']} < evr {v['stuck']}", x["stuck"] < v["stuck"]),
    (f"G4. non-lifting closes beyond 3 cm {x['miss_far']} <= evr {v['miss_far']}", x["miss_far"] <= v["miss_far"]),
]
print()
for t, ok in crit:
    print(f"- [{'PASS' if ok else 'FAIL'}] {t}")
ref = x["success"] - g["success"] >= 0.05 and min(x["runs"]) > max(g["runs"])
print(f"- (참고, B1과 같은 잣대) success {x['success'] - g['success']:+.3f} vs gepl, both evrz runs > both gepl runs: {'PASS' if ref else 'FAIL'}")
se = [pt for _, st in runs["evrz"].values() for pt in st]
if se:
    print(f"- evrz stuck episodes: closest point e_xy / e_z (cm): " + ", ".join(f"({a:.1f}, {b:.1f})" for a, b in se))
