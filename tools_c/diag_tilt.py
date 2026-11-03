# Diagnose "type A" posture failures (wrist never pitches down; truck not tipped). CPU only.
#
# Groups (per rollout):  A = approach failure, late-200 median rotation error > 45 deg, truck never tipped > 30 deg
#                        B = same but truck tipped;  S = success
# Per step we compute
#   down   : angle between EE approach axis (z) and world down [deg]
#   ood    : distance of the policy-relevant state to the nearest demo frame (pre-lift frames, z-scored per dim)
#   cmd_p  : commanded translation |dp| [cm/step];  got_p : achieved |p_{t+1} - p_t| [cm/step]
#   useful : commanded rotation component that pitches the EE axis toward down [deg/step] (+ = toward down)
# and for paired episodes (same initial truck pose, < 1 mm) where one model failed (A) and the other succeeded,
# the first step where the two EE trajectories separate.
#
#   CUDA_VISIBLE_DEVICES= python tools_c/diag_tilt.py --runs evrz:base:outputs/rollouts/evrz_s101 ... --out results/figs/tilt_diag.png
import argparse
import glob
import os
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

sys.argv, _argv = sys.argv[:1], sys.argv
exec(open("tools_c/gripper_diag.py").read().split("ap = argparse.ArgumentParser()")[0])  # analyze_rollout, gf, CFG, torch
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="name:frame:dir  (frame = base|ee, the action frame)")
ap.add_argument("--demo", default="datasets/mimic_500_evt.hdf5")
ap.add_argument("--out", default="results/figs/tilt_diag.png")
A = ap.parse_args()
OFF = os.path.join(CFG, "grasp_offset.json")
DOWN = np.array([0, 0, -1.0])


def q2m(q):
    return Rot.from_quat(np.asarray(q)[:, [1, 2, 3, 0]]).as_matrix()


def canon(q):  # q and -q are the same rotation
    q = np.asarray(q, dtype=np.float64)
    return q * np.where(q[:, :1] < 0, -1, 1)


def ge_ee(ep, eq, tp, tq):
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)  # noqa: E731
    return gf.grasp_err_from_poses(T(ep), T(eq), T(tp), T(tq), OFF, True, "ee").numpy()


def feat(ep, eq, tp, tq):
    """State used for the out-of-distribution measure: EE pose, truck pose, grasp error in EE axes."""
    return np.concatenate([ep, canon(eq), tp, canon(tq), ge_ee(ep, eq, tp, tq)], 1)


# ---- demo reference (pre-lift frames only, i.e. the approach phase)
X = []
with h5py.File(A.demo, "r") as f:
    for k in f["data"]:
        o = f["data"][k]["obs"]
        tp = o["toy_truck_positions"][()]
        z = tp[:, 2] - tp[0, 2]
        tl = int(np.argmax(z > 0.02)) if (z > 0.02).any() else len(z)
        X.append(feat(o["eef_pos"][()][:tl], o["eef_quat"][()][:tl], tp[:tl], o["toy_truck_orientations"][()][:tl]))
X = np.concatenate(X)
mu, sd = X.mean(0), X.std(0) + 1e-6
tree = cKDTree((X - mu) / sd)
print(f"demo approach frames: {len(X)}")


def series(d, frame):
    ep, eq, tp, tq, a = d["eef_pos"], d["eef_quat"], d["toy_truck_positions"], d["toy_truck_orientations"], d["actions"]
    Ree = q2m(eq)
    ez = Ree[:, :, 2]
    down = np.degrees(np.arccos(np.clip(ez @ DOWN, -1, 1)))
    dp, w = a[:, :3], a[:, 3:6]
    if frame == "ee":
        dp, w = np.einsum("tij,tj->ti", Ree, dp), np.einsum("tij,tj->ti", Ree, w)
    ax = np.cross(ez, DOWN)
    ax /= np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9
    ood = tree.query((feat(ep, eq, tp, tq) - mu) / sd)[0]
    got = np.r_[np.linalg.norm(np.diff(ep, axis=0), axis=1), np.nan] * 100
    tilt_truck = np.degrees(np.arccos(np.clip(q2m(tq)[:, 2, 2], -1, 1)))
    gerr = ge_ee(ep, eq, tp, tq)
    return {"down": down, "ood": ood, "cmd_p": np.linalg.norm(dp, axis=1) * 100, "got_p": got,
            "useful": np.degrees((w * ax).sum(1)), "cmd_w": np.degrees(np.linalg.norm(w, axis=1)),
            "ep": np.linalg.norm(gerr[:, :3], axis=1) * 100, "h": (ep[:, 2] - tp[:, 2]) * 100,
            "truck_tilt": tilt_truck, "grip": a[:, 6], "ep_pos": ep, "ep_quat": eq, "tp0": tp[0]}


R = {}  # (name, seed_dir) -> list of (group, series)
for spec in A.runs:
    name, frame, run = spec.split(":")
    rows = []
    for fn in sorted(glob.glob(f"{run}/rollout_*.npz")):
        d = dict(np.load(fn))
        s = series(d, frame)
        cls = analyze_rollout(d)["cls"]
        late_rot = np.median(np.degrees(np.linalg.norm(ge_ee(d["eef_pos"], d["eef_quat"], d["toy_truck_positions"],
                                                             d["toy_truck_orientations"])[-200:, 3:], axis=1)))
        if bool(d["success"]):
            g = "S"
        elif cls == "approach" and late_rot > 45:
            g = "B" if (s["truck_tilt"] > 30).any() else "A"
        else:
            g = "other"
        rows.append((g, s))
    R[(name, run)] = rows

# ---- OOD threshold from successes (pre-lift part)
succ_ood = np.concatenate([s["ood"][:300] for rows in R.values() for g, s in rows if g == "S"])
TH = np.percentile(succ_ood, 99)
print(f"OOD threshold (99th pct of successes, first 300 steps): {TH:.2f}")


def first(cond):
    i = np.where(cond)[0]
    return int(i[0]) if len(i) else -1


print("\n## A 유형 개별 에피소드 (run, idx): down 최소 | 처음 OOD | 처음 '반대로 숙임' 지속(>20 step) | 50 step 단위 down / cmd_p / got_p / h")
stats = {"S": [], "A": []}
for (name, run), rows in R.items():
    for i, (g, s) in enumerate(rows):
        if g not in ("A", "S"):
            continue
        T = len(s["down"])
        t_ood = first(s["ood"][:T] > TH)
        neg = np.convolve((s["useful"] < 0).astype(float), np.ones(20), "valid") >= 20
        t_neg = first(neg)
        t_d45 = first(s["down"] < 45)
        stats[g].append((t_ood, t_neg, t_d45, s["down"][:150].min(), np.nanmean(s["cmd_p"][:100]), np.nanmean(s["got_p"][:100])))
        if g == "A":
            marks = range(0, min(T, 400), 50)
            line = " ".join(f"{s['down'][t]:.0f}/{s['cmd_p'][t]:.1f}/{s['got_p'][t]:.1f}/{s['h'][t]:.0f}" for t in marks)
            print(f"{name} {os.path.basename(run)} #{i:02d}: min {s['down'].min():.0f}° | ood@{t_ood} | neg@{t_neg} | {line}")

for g in ("S", "A"):
    v = np.array(stats[g], dtype=float)
    f = lambda c: v[v[:, c] >= 0, c]  # noqa: E731
    print(f"\n[{g}] n={len(v)} | 처음 OOD step median {np.median(f(0)) if len(f(0)) else -1:.0f} (OOD 된 비율 {np.mean(v[:, 0] >= 0):.2f})"
          f" | 반대로 숙임 지속 시작 median {np.median(f(1)) if len(f(1)) else -1:.0f} (비율 {np.mean(v[:, 1] >= 0):.2f})"
          f" | down<45° 도달 median {np.median(f(2)) if len(f(2)) else -1:.0f} (비율 {np.mean(v[:, 2] >= 0):.2f})"
          f" | 첫 150 step down 최소 median {np.median(v[:, 3]):.0f}° | 첫 100 step cmd_p {np.mean(v[:, 4]):.2f} got_p {np.mean(v[:, 5]):.2f} cm/step")

# ---- paired episodes: same initial truck pose, one model A, the other S
print("\n## 짝 비교 (같은 초기 상태, 한 모델 A / 다른 모델 성공): 두 EE 궤적이 갈라지는 step")
keys = list(R)
pairs = []
for ka in keys:
    for kb in keys:
        if ka == kb or ka[0] == kb[0] or ka[1].split("_s")[-1] != kb[1].split("_s")[-1]:
            continue
        for i, ((ga, sa), (gb, sb)) in enumerate(zip(R[ka], R[kb])):
            if ga == "A" and gb == "S" and np.abs(sa["tp0"] - sb["tp0"]).max() < 1e-3:
                T = min(len(sa["down"]), len(sb["down"]))
                dpos = np.linalg.norm(sa["ep_pos"][:T] - sb["ep_pos"][:T], axis=1) * 100
                dang = np.abs(sa["down"][:T] - sb["down"][:T])
                t_sep = first((dpos > 3) | (dang > 10))
                pairs.append(t_sep)
                print(f"{ka[0]} A vs {kb[0]} S, ep #{i:02d}: 갈라짐 step {t_sep} | 그때 A down {sa['down'][t_sep]:.0f}° S down {sb['down'][t_sep]:.0f}°"
                      f" | A h {sa['h'][t_sep]:.0f}cm S h {sb['h'][t_sep]:.0f}cm | step 0-10 Δpos {dpos[:10].max():.2f}cm")
if pairs:
    print(f"갈라짐 step median {np.median(pairs):.0f} (n={len(pairs)})")

# ---- figure
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_cjk):
    matplotlib.font_manager.fontManager.addfont(_cjk)
    plt.rcParams.update({"font.family": matplotlib.font_manager.FontProperties(fname=_cjk).get_name(), "axes.unicode_minus": False})
fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))
items = [("down", "EE 축과 수직 하방 사이 각 [deg]"), ("ood", "데모 최근접 거리 (z-score)"),
         ("cmd_p", "명령 이동 |dp| [cm/step]"), ("useful", "하방 쪽 회전 명령 [deg/step]")]
for ax, (k, lab) in zip(axs, items):
    for g, c in (("S", "#4C78A8"), ("A", "#E45756")):
        Y = [s[k][:400] for rows in R.values() for gg, s in rows if gg == g]
        L = min(len(y) for y in Y)
        Y = np.array([y[:L] for y in Y], dtype=float)
        if k in ("cmd_p", "useful"):
            Y = np.array([np.convolve(y, np.ones(10) / 10, "same") for y in Y])
        t = np.arange(L)
        ax.plot(t, np.nanmedian(Y, 0), color=c, lw=2, label=f"{'성공' if g == 'S' else 'A 유형 실패'} (n={len(Y)})")
        ax.fill_between(t, np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0), color=c, alpha=0.2, lw=0)
    if k == "ood":
        ax.axhline(TH, color="gray", ls="--", lw=1)
    if k == "useful":
        ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("step")
    ax.set_title(lab, fontsize=10)
    ax.grid(alpha=0.3)
axs[0].legend(fontsize=9)
fig.suptitle("A 유형 자세 실패 진단 (evrz + eez, 중앙값과 25–75% 구간)")
fig.tight_layout()
os.makedirs(os.path.dirname(A.out), exist_ok=True)
fig.savefig(A.out, dpi=130)
print(f"\nfigure -> {A.out}")
