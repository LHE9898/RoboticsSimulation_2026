# Judgement for EE-axes arm actions (pre-registered in results/event_clf/plan_EE.md). CPU only.
#
#   CUDA_VISIBLE_DEVICES= python tools_c/judge_ee.py --ref outputs/rollouts/evrz_s101 outputs/rollouts/evrz_s111 \
#       --exp outputs/rollouts/eez_s101 outputs/rollouts/eez_s111
import argparse
import glob
import os
import sys

import numpy as np

sys.argv, _argv = sys.argv[:1], sys.argv
exec(open("tools_c/gripper_diag.py").read().split("ap = argparse.ArgumentParser()")[0])  # episode, analyze_rollout, gf, CFG
sys.argv = _argv


def metrics(run):
    rows = []
    for fn in sorted(glob.glob(f"{run}/rollout_*.npz")):
        d = dict(np.load(fn))
        T = lambda k: torch.tensor(np.asarray(d[k]), dtype=torch.float64)  # noqa: E731
        ge = gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                     os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()
        p = np.linalg.norm(ge[:, :3], axis=1) * 100
        r = np.degrees(np.linalg.norm(ge[:, 3:], axis=1))
        t = int(np.argmin(p))
        cls = analyze_rollout(d)["cls"]
        appr = cls == "approach"
        e = episode(d, d["actions"], d["success"])
        rows.append({"success": bool(d["success"]), "cls": cls, "lifted": any(c["lift"] for c in e["closes"]),
                     # approach-failure subtypes (results/event_clf/plan_EE.md)
                     "above": appr and p[t] > 10 and np.linalg.norm(ge[t, :2]) * 100 < 5 and ge[t, 2] * 100 > 10,
                     "rot": appr and np.median(r[-200:]) > 45,
                     "near": appr and p[t] <= 10})
    return rows


ap = argparse.ArgumentParser()
ap.add_argument("--ref", nargs="+", required=True, help="evrz runs")
ap.add_argument("--exp", nargs="+", required=True, help="eez runs")
A = ap.parse_args()

cls_names = ["success", "approach", "grasp", "transport", "place"]
print("| run | n | success | lifted | " + " | ".join(cls_names) + " | 접근: 위에서 멈춤 | 접근: 자세 못 맞춤 | 접근: 10 cm 안 |")
print("|" + "---|" * (4 + len(cls_names) + 3))
agg = {}
for name, paths in (("evrz", A.ref), ("eez", A.exp)):
    runs = {os.path.basename(os.path.normpath(p)): metrics(p) for p in paths}
    for run, rows in list(runs.items()) + [(f"**{name} 합산**", [r for rr in runs.values() for r in rr])]:
        cnt = {c: sum(r["cls"] == c for r in rows) for c in cls_names}
        print(f"| {run} | {len(rows)} | {np.mean([r['success'] for r in rows]):.3f} | {np.mean([r['lifted'] for r in rows]):.3f} | "
              + " | ".join(str(cnt[c]) for c in cls_names)
              + f" | {sum(r['above'] for r in rows)} | {sum(r['rot'] for r in rows)} | {sum(r['near'] for r in rows)} |")
    allr = [r for rr in runs.values() for r in rr]
    agg[name] = {"success": np.mean([r["success"] for r in allr]), "lifted": np.mean([r["lifted"] for r in allr]),
                 "approach": sum(r["cls"] == "approach" for r in allr),
                 "late": sum(r["cls"] in ("transport", "place") for r in allr),
                 "runs_appr": [sum(r["cls"] == "approach" for r in rr) for rr in runs.values()]}

v, x = agg["evrz"], agg["eez"]
crit = [
    (f"E1. approach failures {x['approach']} <= {v['approach'] - 10} (evrz {v['approach']} - 10, i.e. -5%p)", x["approach"] <= v["approach"] - 10),
    (f"E2. success {x['success']:.3f} > evrz {v['success']:.3f}", x["success"] > v["success"]),
    (f"E3. lifted {x['lifted']:.3f} >= evrz {v['lifted']:.3f} - 0.03", x["lifted"] >= v["lifted"] - 0.03),
    (f"E4. transport + place failures {x['late']} <= evrz {v['late']} + 5", x["late"] <= v["late"] + 5),
]
print()
for t, ok in crit:
    print(f"- [{'PASS' if ok else 'FAIL'}] {t}")
both = max(x["runs_appr"]) < min(v["runs_appr"])
print(f"- (참고) both eez runs have fewer approach failures than both evrz runs: {'PASS' if both else 'FAIL'} "
      f"(eez {x['runs_appr']}, evrz {v['runs_appr']})")
