# Judgement for the relative-only observation experiment (pre-registered in results/event_clf/plan_REL.md). CPU only.
#
#   CUDA_VISIBLE_DEVICES= python tools_c/judge_rel.py --ref outputs/rollouts/eez_s101 outputs/rollouts/eez_s111 \
#       --exp outputs/rollouts/eerel_s101 outputs/rollouts/eerel_s111
import argparse
import glob
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.argv, _argv_rel = sys.argv[:1], sys.argv
exec(open("tools_c/judge_ee.py").read().split("ap = argparse.ArgumentParser()\nap.add_argument(\"--ref\"")[0])  # metrics()
sys.argv = _argv_rel


def zone(q):
    y = np.degrees(Rot.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_euler("xyz")[2])
    return "G" if -30 <= y < 60 else ("B" if (-60 <= y < -30 or 60 <= y < 90) else "G'")


def rows(run):
    out = []
    for fn, m in zip(sorted(glob.glob(f"{run}/rollout_*.npz")), metrics(run)):
        d = np.load(fn)
        m["zone"] = zone(d["toy_truck_orientations"][0])
        m["far"] = bool(np.linalg.norm(d["toy_truck_positions"][0, :2]) >= 0.65)
        tilt = np.degrees(np.arccos(np.clip(Rot.from_quat(d["toy_truck_orientations"][:, [1, 2, 3, 0]]).as_matrix()[:, 2, 2], -1, 1)))
        m["A"] = m["rot"] and not (tilt > 30).any()
        out.append(m)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--ref", nargs="+", required=True)
ap.add_argument("--exp", nargs="+", required=True)
ap.add_argument("--ref_name", default="eez")
ap.add_argument("--exp_name", default="eerel")
A = ap.parse_args()

S = {}
cells = [("G'/far", lambda r: r["zone"] == "G'" and r["far"]), ("G'/near", lambda r: r["zone"] == "G'" and not r["far"]),
         ("G", lambda r: r["zone"] == "G"), ("boundary", lambda r: r["zone"] == "B"), ("all", lambda r: True)]
print("| run | n | success | lifted | approach | grasp | transport | place | A type | " + " | ".join(f"succ {c}" for c, _ in cells) + " |")
print("|" + "---|" * (9 + len(cells)))
for name, paths in ((A.ref_name, A.ref), (A.exp_name, A.exp)):
    per = {os.path.basename(os.path.normpath(p)): rows(p) for p in paths}
    allr = [r for rr in per.values() for r in rr]
    for run, rr in list(per.items()) + [(f"**{name} 합산**", allr)]:
        cnt = {c: sum(r["cls"] == c for r in rr) for c in ("approach", "grasp", "transport", "place")}
        cs = []
        for _, f in cells:
            sel = [r for r in rr if f(r)]
            cs.append(f"{np.mean([r['success'] for r in sel]):.2f} ({len(sel)})" if sel else "-")
        print(f"| {run} | {len(rr)} | {np.mean([r['success'] for r in rr]):.3f} | {np.mean([r['lifted'] for r in rr]):.3f} | "
              + " | ".join(str(cnt[c]) for c in ("approach", "grasp", "transport", "place"))
              + f" | {sum(r['A'] for r in rr)} | " + " | ".join(cs) + " |")
    S[name] = {"all": allr, "runs": per}


def succ(rr, f=lambda r: True):
    sel = [r for r in rr if f(r)]
    return np.mean([r["success"] for r in sel]) if sel else float("nan")


ref, exp = S[A.ref_name]["all"], S[A.exp_name]["all"]
far = cells[0][1]
g = cells[2][1]
crit = [
    (f"R1. G'/far success {succ(exp, far):.3f} >= {A.ref_name} {succ(ref, far):.3f} + 0.10", succ(exp, far) >= succ(ref, far) + 0.10),
    (f"R2. overall success {succ(exp):.3f} > {A.ref_name} {succ(ref):.3f}", succ(exp) > succ(ref)),
    (f"R3. G-zone success {succ(exp, g):.3f} >= {A.ref_name} {succ(ref, g):.3f} - 0.10 (no harm)", succ(exp, g) >= succ(ref, g) - 0.10),
    (f"R4. lifted {np.mean([r['lifted'] for r in exp]):.3f} >= {A.ref_name} {np.mean([r['lifted'] for r in ref]):.3f} - 0.03 (no harm)",
     np.mean([r["lifted"] for r in exp]) >= np.mean([r["lifted"] for r in ref]) - 0.03),
]
print()
for t, ok in crit:
    print(f"- [{'PASS' if ok else 'FAIL'}] {t}")
fr = [succ(rr, far) for rr in S[A.exp_name]["runs"].values()]
fe = [succ(rr, far) for rr in S[A.ref_name]["runs"].values()]
print(f"- (참고) both {A.exp_name} runs have higher G'/far success than both {A.ref_name} runs: "
      f"{'PASS' if min(fr) > max(fe) else 'FAIL'} ({A.exp_name} {np.round(fr, 2).tolist()}, {A.ref_name} {np.round(fe, 2).tolist()})")
