# Diagnose "dropped during transport" failures: truck lifted (> 2 cm), gripper still commanded closed, truck falls back
# (< 1 cm above its start) before the first open command. CPU only.
#
# Per episode, from close (t_c), lift (t_l = first z > 2 cm) and release (t_r = first open command after t_l):
#   hold pose  : truck pose in EE axes, d(t) = R_E^T (p_T - p_E), R_rel(t) = R_E^T R_T; reference at t_l + 5
#   slip       : |d(t) - d_ref| [cm] and angle(R_rel(t), R_rel_ref) [deg]; t_slip = first step with slip > 2 cm
#   grasp      : |e_p|, |e_R| at t_c (grasp_err), finger readings (gripper_pos) at t_l + 5
#   motion     : EE linear speed [cm/step], acceleration [cm/step^2], angular speed [deg/step]
#   place      : truck height above start, truck-box horizontal distance
# Successes (S) are measured over their transport (t_l + 5 .. t_r).
#
#   CUDA_VISIBLE_DEVICES= python tools_c/diag_drop.py --runs evrz:outputs/rollouts/evrz_s101 ... > results/event_clf/diag_drop.txt
import argparse
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="name:dir")
A = ap.parse_args()
LIFT, DROP, SLIP = 0.02, 0.01, 0.02


def q2m(q):
    return Rot.from_quat(np.asarray(q)[:, [1, 2, 3, 0]]).as_matrix()


def yaw(q):
    return np.degrees(Rot.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_euler("xyz")[2])


def zone(y):
    return "G" if -30 <= y < 60 else ("B" if (-60 <= y < -30 or 60 <= y < 90) else "G'")


def first(c):
    i = np.where(c)[0]
    return int(i[0]) if len(i) else None


def episode(d):
    a, ep, eq, tp, tq, bp = (d[k] for k in ("actions", "eef_pos", "eef_quat", "toy_truck_positions", "toy_truck_orientations", "box_positions"))
    T = len(a)
    z = tp[:, 2] - tp[0, 2]
    t_c = first(a[:, 6] < 0)
    t_l = first(z > LIFT)
    if t_c is None or t_l is None or t_l + 6 >= T:
        return None
    t_r = first(a[t_l:, 6] > 0)
    t_r = t_l + t_r if t_r is not None else None
    end = t_r if t_r is not None else T
    t_dz = first(z[t_l:end] < DROP)
    t_dz = t_l + t_dz if t_dz is not None else None
    RE, RT = q2m(eq), q2m(tq)
    dl = np.einsum("tji,tj->ti", RE, tp - ep)  # truck position in EE axes
    Rr = np.einsum("tji,tjk->tik", RE, RT)
    ref = t_l + 5
    slip_p = np.linalg.norm(dl - dl[ref], axis=1) * 100
    slip_R = np.degrees(Rot.from_matrix(np.einsum("ji,tjk->tik", Rr[ref], Rr)).magnitude())
    t_s = first(slip_p[ref:end] > SLIP * 100)
    t_s = ref + t_s if t_s is not None else None
    v = np.r_[np.linalg.norm(np.diff(ep, axis=0), axis=1), 0] * 100
    acc = np.r_[np.abs(np.diff(v)), 0]
    w = np.r_[np.degrees(Rot.from_matrix(np.einsum("tji,tjk->tik", RE[:-1], RE[1:])).magnitude()), 0]
    xy_box = np.linalg.norm(tp[:, :2] - bp[:, :2], axis=1) * 100
    tilt_T = np.degrees(np.arccos(np.clip(RT[:, 2, 2], -1, 1)))
    down = np.degrees(np.arccos(np.clip(-RE[:, 2, 2], -1, 1)))  # EE approach axis vs world down
    e = d["grasp_err"]
    return dict(T=T, t_c=t_c, t_l=t_l, t_r=t_r, t_dz=t_dz, t_s=t_s, z=z * 100, slip_p=slip_p, slip_R=slip_R, v=v, acc=acc, w=w,
                xy_box=xy_box, tilt_T=tilt_T, down=down, grip=d["gripper_pos"], d_ref=dl[ref] * 100, a=a,
                ep_c=np.linalg.norm(e[t_c, :3]) * 100, eR_c=np.degrees(np.linalg.norm(e[t_c, 3:])), cell=zone(yaw(tq[0])),
                far=np.linalg.norm(tp[0, :2]) >= 0.65, success=bool(d["success"]), ee_h=(ep[:, 2] - bp[:, 2]) * 100)


E = []
for spec in A.runs:
    name, run = spec.split(":")
    for fn in sorted(glob.glob(f"{run}/rollout_*.npz")):
        r = episode(dict(np.load(fn)))
        if r is None:
            continue
        r["model"], r["run"], r["idx"] = name, os.path.basename(run), int(os.path.basename(fn).split("_")[1].split(".")[0])
        r["grp"] = "S" if r["success"] else ("D" if r["t_dz"] is not None else "other")
        E.append(r)

D = [r for r in E if r["grp"] == "D"]
S = [r for r in E if r["grp"] == "S"]
print(f"episodes with a lift: {len(E)} | dropped (D): {len(D)} | success (S): {len(S)}")
for m in dict.fromkeys(r["model"] for r in E):
    print(f"  {m}: D {sum(r['grp'] == 'D' and r['model'] == m for r in E)}")

print("\n## D 개별: 잡기→들어 올림→미끄러짐(>2cm)→바닥 step | 들어 올린 최고 높이 | 미끄러지기 전 10 step: EE 속도/가속/각속도 | 그때 트럭–박스 xy, EE 하방각 | 미끄러짐 때 트럭 기울기")
for r in D:
    ts = r["t_s"] if r["t_s"] is not None else r["t_dz"]
    sl = slice(max(r["t_l"] + 5, ts - 10), ts + 1)
    print(f"{r['model']:5s} {r['run']:10s} #{r['idx']:02d} {r['cell']:2s}{'/far ' if r['far'] else '/near'} | c {r['t_c']} l {r['t_l']} s {r['t_s']} drop {r['t_dz']}"
          f" (hold {ts - r['t_l']}) | zmax {r['z'][r['t_l']:r['t_dz']].max():.0f}cm | v {r['v'][sl].max():.1f} a {r['acc'][sl].max():.2f} w {r['w'][sl].max():.1f}"
          f" | xy_box {r['xy_box'][ts]:.0f}cm EE-box h {r['ee_h'][ts]:.0f}cm down {r['down'][ts]:.0f}° | tilt {r['tilt_T'][ts]:.0f}°"
          f" | grip@l+5 {np.round(r['grip'][r['t_l'] + 5], 2).tolist()} | e_p@c {r['ep_c']:.1f}cm e_R@c {r['eR_c']:.0f}° | d_ref {np.round(r['d_ref'], 1).tolist()}")


def pct(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    return f"{np.median(x):6.2f} [{np.percentile(x, 25):6.2f},{np.percentile(x, 75):6.2f}]" if len(x) else "  -"


def hold_slice(r):
    if r["grp"] == "D":
        ts = r["t_s"] if r["t_s"] is not None else r["t_dz"]
        return slice(r["t_l"] + 5, max(ts + 1, r["t_l"] + 6))
    return slice(r["t_l"] + 5, max(r["t_r"] or r["T"], r["t_l"] + 6))


print("\n## D vs S 비교 (중앙값 [25%, 75%])")
feats = [
    ("잡을 때 |e_p| [cm]", lambda r: r["ep_c"]),
    ("잡을 때 |e_R| [deg]", lambda r: r["eR_c"]),
    ("손가락 1 @l+5", lambda r: r["grip"][r["t_l"] + 5, 0]),
    ("손가락 2 @l+5", lambda r: r["grip"][r["t_l"] + 5, 1]),
    ("손가락 합 @l+5", lambda r: r["grip"][r["t_l"] + 5].sum()),
    ("EE→트럭 거리 @l+5 [cm]", lambda r: np.linalg.norm(r["d_ref"])),
    ("EE→트럭 z(EE축) @l+5 [cm]", lambda r: r["d_ref"][2]),
    ("들어 올림→미끄러짐/놓기 [step]", lambda r: hold_slice(r).stop - r["t_l"]),
    ("운반 중 EE 최대 속도 [cm/step]", lambda r: r["v"][hold_slice(r)].max()),
    ("운반 중 EE 최대 가속 [cm/step²]", lambda r: r["acc"][hold_slice(r)].max()),
    ("운반 중 EE 최대 각속도 [deg/step]", lambda r: r["w"][hold_slice(r)].max()),
    ("운반 중 EE 누적 회전 [deg]", lambda r: r["w"][hold_slice(r)].sum()),
    ("운반 중 최대 미끄러짐 위치 [cm] (D는 2 cm 넘기 직전까지)", lambda r: r["slip_p"][hold_slice(r)][:-1].max() if hold_slice(r).stop - hold_slice(r).start > 1 else np.nan),
    ("운반 중 최대 미끄러짐 회전 [deg]", lambda r: r["slip_R"][hold_slice(r)][:-1].max() if hold_slice(r).stop - hold_slice(r).start > 1 else np.nan),
    ("운반 중 EE 하방각 최대 [deg]", lambda r: r["down"][hold_slice(r)].max()),
    ("미끄러짐/놓기 때 트럭–박스 xy [cm]", lambda r: r["xy_box"][hold_slice(r).stop - 1]),
    ("미끄러짐/놓기 때 트럭 높이 [cm]", lambda r: r["z"][hold_slice(r).stop - 1]),
    ("운반 중 트럭 최고 높이 [cm]", lambda r: r["z"][hold_slice(r)].max()),
]
print(f"{'':52s} {'D (n=' + str(len(D)) + ')':24s} S (n={len(S)})")
for lab, f in feats:
    print(f"{lab:50s} {pct([f(r) for r in D]):24s} {pct([f(r) for r in S])}")

print("\n## 셀별 D 비율 (들어 올린 에피소드 중)")
for c in ("G", "B", "G'"):
    for far in (False, True):
        sel = [r for r in E if r["cell"] == c and r["far"] == far]
        if sel:
            print(f"  {c:2s}/{'far ' if far else 'near'}: lifted {len(sel):3d}, D {sum(r['grp'] == 'D' for r in sel):2d} ({np.mean([r['grp'] == 'D' for r in sel]):.2f})")
