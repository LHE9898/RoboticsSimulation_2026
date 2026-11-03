# Judgement for the close-rule depth tolerance 3 cm -> 2 cm (pre-registered in results/event_clf/plan_Z2.md). CPU only.
#
#   CUDA_VISIBLE_DEVICES= python tools_c/judge_z2.py --ref outputs/rollouts/eez_s101 outputs/rollouts/eez_s111 \
#       --exp outputs/rollouts/eez2_s101 outputs/rollouts/eez2_s111
import argparse
import glob
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ref", nargs="+", required=True)
ap.add_argument("--exp", nargs="+", required=True)
ap.add_argument("--ref_name", default="eez")
ap.add_argument("--exp_name", default="eez2")
ARGS = ap.parse_args()

# metrics() and rows() (cells, A type) from judge_rel.py; episode() (drop detection) from diag_drop.py
_src = open("tools_c/judge_rel.py").read().split("ap = argparse.ArgumentParser()\nap.add_argument(\"--ref\"")[0]
sys.argv = sys.argv[:1]
exec(_src)
exec(open("tools_c/diag_drop.py").read().split("ap = argparse.ArgumentParser()")[0])
_dd = open("tools_c/diag_drop.py").read().split("A = ap.parse_args()")[1].split("E = []")[0]
exec(_dd.replace("def episode(d):", "def drop_episode(d):").replace("def zone(y):", "def _zone_y(y):").replace("cell=zone(", "cell=_zone_y("))  # keep judge_ee's episode()/judge_rel's zone()
OFF = glob.glob("source/**/agibot/grasp_offset.json", recursive=True)[0]


def run_rows(run):
    out = []
    files = sorted(glob.glob(f"{run}/rollout_*.npz"))
    for fn, m in zip(files, rows(run)):
        d = dict(np.load(fn))
        e = drop_episode(d)
        m["D"] = e is not None and e["t_dz"] is not None and not m["success"]
        m["depth"] = e["d_ref"][2] if e is not None else np.nan
        c = np.where(d["actions"][:, 6] < 0)[0]
        m["closed"] = bool(len(c))
        if len(c):
            t = c[0]
            T = lambda k: torch.tensor(d[k][t:t + 1], dtype=torch.float64)  # noqa: E731
            m["ez_c"] = float(gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                                      OFF, True, "ee")[0, 2]) * 100
        else:
            m["ez_c"] = np.nan
        out.append(m)
    return out


cells = [("G'/far", lambda r: r["zone"] == "G'" and r["far"]), ("G'/near", lambda r: r["zone"] == "G'" and not r["far"]),
         ("G", lambda r: r["zone"] == "G"), ("boundary", lambda r: r["zone"] == "B")]
S = {}
print("| run | n | success | lifted | approach | grasp | transport | place | dropped D | never closed | e_z@close median | depth median | "
      + " | ".join(f"succ {c}" for c, _ in cells) + " | " + " | ".join(f"D {c}" for c, _ in cells) + " |")
print("|" + "---|" * (12 + 2 * len(cells)))
for name, paths in ((ARGS.ref_name, ARGS.ref), (ARGS.exp_name, ARGS.exp)):
    per = {os.path.basename(os.path.normpath(p)): run_rows(p) for p in paths}
    allr = [r for rr in per.values() for r in rr]
    for run, rr in list(per.items()) + [(f"**{name} 합산**", allr)]:
        cnt = {c: sum(r["cls"] == c for r in rr) for c in ("approach", "grasp", "transport", "place")}
        sc, dc = [], []
        for _, f in cells:
            sel = [r for r in rr if f(r)]
            sc.append(f"{np.mean([r['success'] for r in sel]):.2f} ({len(sel)})" if sel else "-")
            dc.append(str(sum(r["D"] for r in sel)))
        print(f"| {run} | {len(rr)} | {np.mean([r['success'] for r in rr]):.3f} | {np.mean([r['lifted'] for r in rr]):.3f} | "
              + " | ".join(str(cnt[c]) for c in ("approach", "grasp", "transport", "place"))
              + f" | {sum(r['D'] for r in rr)} | {sum(not r['closed'] for r in rr)} | {np.nanmedian([r['ez_c'] for r in rr]):+.2f}"
              f" | {np.nanmedian([r['depth'] for r in rr]):+.2f} | " + " | ".join(sc) + " | " + " | ".join(dc) + " |")
    S[name] = allr

ref, exp = S[ARGS.ref_name], S[ARGS.exp_name]
nD = lambda rr: sum(r["D"] for r in rr)  # noqa: E731
succ = lambda rr: np.mean([r["success"] for r in rr])  # noqa: E731
ag = lambda rr: sum(r["cls"] in ("approach", "grasp") for r in rr)  # noqa: E731
lift = lambda rr: np.mean([r["lifted"] for r in rr])  # noqa: E731
crit = [
    (f"Z1. dropped D {nD(exp)} <= {ARGS.ref_name} {nD(ref)} / 2", nD(exp) <= nD(ref) / 2),
    (f"Z2. overall success {succ(exp):.3f} > {ARGS.ref_name} {succ(ref):.3f}", succ(exp) > succ(ref)),
    (f"Z3. approach + grasp failures {ag(exp)} <= {ARGS.ref_name} {ag(ref)} + 10 (no harm)", ag(exp) <= ag(ref) + 10),
    (f"Z4. lifted {lift(exp):.3f} >= {ARGS.ref_name} {lift(ref):.3f} - 0.03 (no harm)", lift(exp) >= lift(ref) - 0.03),
]
print()
for t, ok in crit:
    print(f"- [{'PASS' if ok else 'FAIL'}] {t}")
for name, rr in ((ARGS.ref_name, ref), (ARGS.exp_name, exp)):
    ez = np.array([r["ez_c"] for r in rr if r["closed"]])
    print(f"- ({name}) e_z@close quantiles 10/25/50/75/90 = {np.round(np.percentile(ez, [10, 25, 50, 75, 90]), 2).tolist()} cm,"
          f" > 2 cm {np.mean(ez > 2):.2f}")
