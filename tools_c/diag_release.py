# Diagnose "released outside the box" failures: truck lifted, first open command after the lift (t_r) with the truck
# >= 10 cm (xy) from the box center, not a success. CPU only.
#
# Groups:  O = released outside (not success);  S = success;  also the demos (mimic 500).
# At the release step t_r we compare
#   truck-box xy, EE-box xy, truck offset from the EE (horizontal, world axes), EE height above the box,
#   EE speed and the direction of motion (cos to the EE->box xy direction), whether the EE had stalled,
#   the open probability p before t_r (how it crossed tau), the closest truck-box xy reached before t_r,
#   time from lift to release, and where the truck ends up (final truck-box xy).
#
#   CUDA_VISIBLE_DEVICES= python tools_c/diag_release.py --runs eez2:outputs/rollouts/eez2_s101 ... > results/event_clf/diag_release.txt
import argparse
import glob
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="name:dir")
ap.add_argument("--demo", default="datasets/mimic_500_evt_ee.hdf5")
A = ap.parse_args()
LIFT, BOX = 0.02, 0.10


def yaw(q):
    return np.degrees(Rot.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_euler("xyz")[2])


def zone(y):
    return "G" if -30 <= y < 60 else ("B" if (-60 <= y < -30 or 60 <= y < 90) else "G'")


def first(c):
    i = np.where(c)[0]
    return int(i[0]) if len(i) else None


def release(a, ep, tp, tq, bp, p=None, success=None):
    z = tp[:, 2] - tp[0, 2]
    t_l = first(z > LIFT)
    if t_l is None:
        return None
    t_r = first(a[t_l:, 6] > 0)
    if t_r is None:
        return None
    t_r += t_l
    xy_t = np.linalg.norm(tp[:, :2] - bp[:, :2], axis=1) * 100
    xy_e = np.linalg.norm(ep[:, :2] - bp[:, :2], axis=1) * 100
    v = np.r_[np.diff(ep, axis=0), np.zeros((1, 3))]  # m/step
    w = slice(max(t_r - 10, 0), t_r + 1)
    vxy = v[w, :2].mean(0)
    to_box = (bp[t_r, :2] - ep[t_r, :2])
    cos = float(vxy @ to_box / (np.linalg.norm(vxy) * np.linalg.norm(to_box) + 1e-9))
    off = (tp[t_r, :2] - ep[t_r, :2]) * 100  # truck root minus EE, world xy
    # is the truck offset pointing away from the box (EE closer to the box than the truck)?
    away = float(off @ (to_box * 100) / (np.linalg.norm(off) * np.linalg.norm(to_box * 100) + 1e-9))
    r = dict(t_l=t_l, t_r=t_r, hold=t_r - t_l, xy_t=xy_t[t_r], xy_e=xy_e[t_r], xy_min=xy_t[t_l:t_r + 1].min(),
             t_min=int(np.argmin(xy_t[t_l:t_r + 1])), off=np.linalg.norm(off), off_away=away,
             h_e=(ep[t_r, 2] - bp[t_r, 2]) * 100, h_t=(tp[t_r, 2] - bp[t_r, 2]) * 100,
             speed=np.linalg.norm(v[w], axis=1).mean() * 100, cos=cos,
             xy_end=xy_t[-1], box_r=np.linalg.norm(bp[0, :2]) * 100, cell=zone(yaw(tq[0])), far=np.linalg.norm(tp[0, :2]) >= 0.65,
             approach_v=-(np.diff(xy_e[max(t_r - 10, 0):t_r + 1]).mean()))  # cm/step closing speed toward the box
    if p is not None:
        r["p_r"] = p[t_r]
        r["p_prev"] = p[max(t_r - 20, 0):max(t_r - 3, 1)].max() if t_r > 3 else np.nan
        cross = np.where(p[t_l:t_r + 1] > 0.5)[0]
        r["t_p50"] = t_r - (t_l + cross[0]) if len(cross) else np.nan  # steps between p first > 0.5 and release
    return r


E = []
for spec in A.runs:
    name, run = spec.split(":")
    for fn in sorted(glob.glob(f"{run}/rollout_*.npz")):
        d = dict(np.load(fn))
        r = release(d["actions"], d["eef_pos"], d["toy_truck_positions"], d["toy_truck_orientations"], d["box_positions"], d["event_p"])
        if r is None:
            continue
        r.update(model=name, run=os.path.basename(run), idx=int(os.path.basename(fn).split("_")[1].split(".")[0]), success=bool(d["success"]))
        r["grp"] = "S" if r["success"] else ("O" if r["xy_t"] >= BOX * 100 else "I")
        E.append(r)

Dm = []
with h5py.File(A.demo, "r") as f:
    for k in f["data"]:
        g = f["data"][k]
        o = g["obs"]
        r = release(g["actions"][()], o["eef_pos"][()], o["toy_truck_positions"][()], o["toy_truck_orientations"][()], o["box_positions"][()])
        if r is not None:
            r["grp"] = "demo"
            Dm.append(r)

O = [r for r in E if r["grp"] == "O"]
S = [r for r in E if r["grp"] == "S"]
print(f"released after a lift: {len(E)} | outside & failed (O): {len(O)} | success (S): {len(S)} | released inside but failed (I): {sum(r['grp'] == 'I' for r in E)} | demos: {len(Dm)}")
for m in dict.fromkeys(r["model"] for r in E):
    print(f"  {m}: O {sum(r['grp'] == 'O' and r['model'] == m for r in E)}, S {sum(r['grp'] == 'S' and r['model'] == m for r in E)}")

print("\n## O 개별")
for r in O:
    print(f"{r['model']:5s} {r['run']:10s} #{r['idx']:02d} {r['cell']:2s}{'/far ' if r['far'] else '/near'} | hold {r['hold']:3d} | 놓을 때 트럭–박스 {r['xy_t']:4.0f} EE–박스 {r['xy_e']:4.0f} 트럭–EE {r['off']:4.1f}cm (박스 반대쪽 cos {r['off_away']:+.2f})"
          f" | 최근접 {r['xy_min']:4.0f}cm | EE 높이 {r['h_e']:4.0f} 트럭 높이 {r['h_t']:4.0f} | 속도 {r['speed']:.1f}cm/step 박스쪽 cos {r['cos']:+.2f} 접근속도 {r['approach_v']:+.2f}"
          f" | p@r {r['p_r']:.2f} p50→놓기 {r['t_p50']} step | 끝 위치 {r['xy_end']:.0f}cm | 박스 반경 {r['box_r']:.0f}cm")


def pct(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return f"{np.median(x):7.2f} [{np.percentile(x, 25):6.2f},{np.percentile(x, 75):6.2f}]" if len(x) else "   -"


feats = [("들어 올림→놓기 [step]", "hold"), ("놓을 때 트럭–박스 xy [cm]", "xy_t"), ("놓을 때 EE–박스 xy [cm]", "xy_e"),
         ("놓을 때 트럭–EE 수평 거리 [cm]", "off"), ("트럭이 EE보다 박스 반대쪽 (cos)", "off_away"),
         ("놓기 전 트럭–박스 최근접 [cm]", "xy_min"), ("놓을 때 EE 높이 (박스 기준) [cm]", "h_e"), ("놓을 때 트럭 높이 (박스 기준) [cm]", "h_t"),
         ("놓기 전 10 step EE 속도 [cm/step]", "speed"), ("이동 방향과 EE→박스 방향 cos", "cos"), ("박스 쪽 접근 속도 [cm/step]", "approach_v"),
         ("열기 확률 p 0.5 넘고 놓기까지 [step]", "t_p50"), ("끝 트럭–박스 xy [cm]", "xy_end"), ("박스 수평 거리 (로봇 기준) [cm]", "box_r")]
print("\n## 비교 (중앙값 [25%, 75%])")
print(f"{'':40s} {'O (n=' + str(len(O)) + ')':26s} {'S (n=' + str(len(S)) + ')':26s} 데모 (n={len(Dm)})")
for lab, k in feats:
    print(f"{lab:38s} {pct([r.get(k) for r in O]):26s} {pct([r.get(k) for r in S]):26s} {pct([r.get(k) for r in Dm])}")

print("\n## 놓을 때 트럭–박스 xy 구간별: 성공 비율 (정책 rollout) / 데모 비율")
for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 200)):
    sel = [r for r in E if lo <= r["xy_t"] < hi]
    dm = np.mean([lo <= r["xy_t"] < hi for r in Dm])
    if sel:
        print(f"  [{lo:3d},{hi:3d}) cm: rollout n={len(sel):3d} 성공 {np.mean([r['success'] for r in sel]):.2f} | 데모 비율 {dm:.2f}")
print("\n## 셀별 O 비율 (놓은 에피소드 중)")
for c in ("G", "B", "G'"):
    for far in (False, True):
        sel = [r for r in E if r["cell"] == c and r["far"] == far]
        if sel:
            print(f"  {c:2s}/{'far ' if far else 'near'}: 놓음 {len(sel):3d}, O {sum(r['grp'] == 'O' for r in sel):2d} ({np.mean([r['grp'] == 'O' for r in sel]):.2f})")
