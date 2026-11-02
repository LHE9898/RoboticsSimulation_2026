# CPU check of a geometric close rule (pre-registered in results/event_clf/plan_R45.md):
#   close when |e_G,p| < R (and e_z < ez_max if given) for M consecutive steps while open. Replayed on recorded rollouts (first close only).
# --lateral: use the lateral error e_xy = |(e_x, e_y)| instead of |e_p| (|e_p| already contains e_z), i.e. close when
#   e_xy < R and e_z < ez_max.
# Expected lift = empirical lift rate of the (distance, e_z) cell of the close, from all rollout closes (gepl + evr + evi).
#
#   CUDA_VISIBLE_DEVICES= python tools_c/check_region_rule.py --radius 0.045 --rollouts outputs/rollouts/evr_s101 outputs/rollouts/evr_s111 \
#       --ref outputs/rollouts/gepl_s101 outputs/rollouts/gepl_s111 --table outputs/rollouts/{gepl,evr,evi}_s1{01,11}
import argparse
import glob
import os
import sys

import numpy as np

sys.argv, _argv = sys.argv[:1], sys.argv
exec(open("tools_c/gripper_diag.py").read().split("ap = argparse.ArgumentParser()")[0])  # gf, CFG, LIFT, torch
sys.argv = _argv

EP_BINS = [0, 2, 3, 4, 5, 7, 10, np.inf]  # cm
EZ_BINS = [-np.inf, -1, 0, 1, 2, np.inf]  # cm
LATERAL = "--lateral" in _argv
if LATERAL:
    EP_BINS = [0, 1, 2, 3, 4, 6, np.inf]
    EZ_BINS = [-np.inf, -2, -1, 0, 1, 2, 3, 5, np.inf]


def load(run):
    out = []
    for fn in sorted(glob.glob(f"{run}/rollout_*.npz")):
        d = dict(np.load(fn))
        T = lambda k: torch.tensor(np.asarray(d[k]), dtype=torch.float64)  # noqa: E731
        ge = gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                     os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()
        closed = d["actions"][:, 6] < 0
        tz = d["toy_truck_positions"][:, 2] - d["toy_truck_positions"][0, 2]
        dd = np.diff(np.r_[0, closed.astype(int), 0])
        segs = list(zip(np.where(dd == 1)[0], np.where(dd == -1)[0]))
        p3 = np.linalg.norm(ge[:, :3], axis=1) * 100
        out.append({"ep": np.linalg.norm(ge[:, :2], axis=1) * 100 if LATERAL else p3, "p3": p3, "ez": ge[:, 2] * 100, "closed": closed,
                    "closes": [(s, bool((tz[s : min(e + 1, len(tz))] > LIFT).any())) for s, e in segs],
                    "success": bool(d["success"])})
    return out


def cell(ep, ez):
    return np.searchsorted(EP_BINS, ep, side="right") - 1, np.searchsorted(EZ_BINS, ez, side="right") - 1


def lift_table(eps):
    n = np.zeros((len(EP_BINS) - 1, len(EZ_BINS) - 1))
    k = np.zeros_like(n)
    for e in eps:
        for s, lift in e["closes"]:
            i, j = cell(e["ep"][s], e["ez"][s])
            n[i, j] += 1
            k[i, j] += lift
    row = k.sum(1) / np.maximum(n.sum(1), 1)
    return np.where(n > 0, k / np.maximum(n, 1), row[:, None]), n


def replay(eps, R, M, table, ez_max=np.inf):
    stuck, lifted, new, wait = [], [], [], []
    for e in eps:
        t_close = e["closes"][0][0] if e["closes"] else len(e["ep"])
        inside = ((e["ep"][:t_close] < R) & (e["ez"][:t_close] < ez_max)).astype(int)
        f = np.convolve(inside, np.ones(M, int), "full")[:t_close] >= M
        t = int(np.argmax(f)) if f.any() else None
        first_lift = bool(e["closes"] and e["closes"][0][1])
        if not e["closes"] and e["p3"].min() < 3:  # same 13 stuck episodes whatever the rule metric
            stuck.append(t is not None)
        if first_lift:
            tt = t if t is not None else t_close
            i, j = cell(e["ep"][tt], e["ez"][tt])
            i0, j0 = cell(e["ep"][t_close], e["ez"][t_close])
            lifted.append((e["ep"][tt], e["ez"][tt], table[i, j], table[i0, j0], t is not None and t < t_close, t is None))
            # rule had not fired and the EE was outside the region at the recorded close -> the rule would still be waiting
            wait.append(t is None and not (e["ep"][t_close] < R and e["ez"][t_close] < ez_max))
        elif t is not None and not any(l for _, l in e["closes"]):
            new.append(e["ep"][t])
    L = np.array(lifted)
    return {"stuck_fired": int(np.sum(stuck)), "stuck_n": len(stuck), "lifted_n": len(L),
            "fire_ep_median": float(np.median(L[:, 0])), "earlier_than_recorded": float(np.mean(L[:, 4])),
            "exp_lift_rule": float(np.mean(L[:, 2])), "exp_lift_recorded": float(np.mean(L[:, 3])),
            "risky": float(np.mean((L[:, 1] >= 2) & (L[:, 0] >= 3))), "low_cell": float(np.mean(L[:, 2] < 0.8)),
            "no_fire_by_recorded": float(np.mean(L[:, 5])), "wait_outside": float(np.mean(wait)),
            "new_close_in_nonlift_eps": len(new)}


ap = argparse.ArgumentParser()
ap.add_argument("--radius", type=float, default=0.045)
ap.add_argument("--consecutive", type=int, default=3)
ap.add_argument("--ez_max", type=float, default=None, help="m; also require e_z below this")
ap.add_argument("--lateral", action="store_true", help="rule and table on e_xy = |(e_x, e_y)| instead of |e_p|")
ap.add_argument("--rollouts", nargs="+", required=True)
ap.add_argument("--ref", nargs="*", default=[])
ap.add_argument("--table", nargs="+", required=True)
ap.add_argument("--out", default="results/event_clf/results_R45.md")
A = ap.parse_args()
R = A.radius * 100
Z = A.ez_max * 100 if A.ez_max is not None else np.inf

table, n = lift_table([e for r in A.table for e in load(r)])
main = replay([e for r in A.rollouts for e in load(r)], R, A.consecutive, table, Z)
ref = replay([e for r in A.ref for e in load(r)], R, A.consecutive, table, Z) if A.ref else None

crit = [
    ("R1. stuck episodes: closes in >= 12 of 13", main["stuck_fired"] >= 12),
    ("R2. expected lift (rule) not more than 2%p below the recorded closes",
     main["exp_lift_rule"] >= main["exp_lift_recorded"] - 0.02),
    ("R3. first closes with e_z >= 2 cm and |e_p| >= 3 cm <= 15%", main["risky"] <= 0.15),
]
if A.ez_max is not None:  # plan_R45.md addendum: R3 is 0 by construction, R4 guards against a new stuck mode
    crit = crit[:2] + [("R4. lifted episodes where the rule never fired by the recorded close <= 10%", main["no_fire_by_recorded"] <= 0.10)]
if A.lateral:  # plan_R45.md addendum 5: R4' counts only the episodes where the rule would still be waiting
    crit = crit[:2] + [("R3'. first closes in low-lift cells (< 0.8) <= 10%", main["low_cell"] <= 0.10),
                       ("R4'. lifted episodes where the rule would still be waiting at the recorded close <= 10%", main["wait_outside"] <= 0.10)]
names = {"stuck_fired": "버틴 에피소드 중 닫음", "stuck_n": "버틴 에피소드", "lifted_n": "트럭을 든 에피소드",
         "fire_ep_median": "규칙이 처음 닫는 거리 중앙값 (cm)", "earlier_than_recorded": "기록보다 먼저 닫는 비율",
         "exp_lift_rule": "들어 올림 기대값: 규칙", "exp_lift_recorded": "들어 올림 기대값: 기록된 닫힘",
         "risky": "위험 구역(e_z≥2, |e_p|≥3)에서 닫는 비율", "low_cell": "들어 올린 비율 < 0.8인 칸에서 닫는 비율",
         "no_fire_by_recorded": "기록된 닫힘까지 한 번도 발동 안 한 비율", "wait_outside": "기록된 닫힘 순간 영역 밖이라 계속 기다릴 비율", "new_close_in_nonlift_eps": "못 든 에피소드 중 규칙이라면 닫았을 건수"}
cols = [("evr (" + ", ".join(os.path.basename(r) for r in A.rollouts) + ")", main)]
if ref:
    cols.append(("참고: gepl (" + ", ".join(os.path.basename(r) for r in A.ref) + ")", ref))
fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
L = [f"# 닫기 영역 규칙 {'e_xy' if A.lateral else '|e_p|'} < {R:.1f} cm" + (f" 그리고 e_z < {Z:.1f} cm" if A.ez_max is not None else "") + f", {A.consecutive} step 연속 (자동 생성: tools_c/check_region_rule.py, 계획: plan_R45.md)", "",
     "| 지표 | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
L += [f"| {names[k]} | " + " | ".join(fmt(m[k]) for _, m in cols) + " |" for k in names]
L += ["", "- 들어 올림 기대값: 닫은 순간의 (|e_p|, e_z) 칸(--lateral이면 (e_xy, e_z) 칸)에서 실제로 들어 올린 비율(gepl+evr+evi 닫힘)의 평균. 규칙과 기록된 닫힘에 같은 잣대를 쓴다.", ""]
L += [f"- [{'PASS' if ok else 'FAIL'}] {t}" for t, ok in crit]
open(A.out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
