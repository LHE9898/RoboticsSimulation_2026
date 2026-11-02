# Where does a close lift the truck? Separate lifting / non-lifting closes of gepl by the EE pose relative to the grasp target
# (CPU only). LDA / logistic regression for the boundary (figure: LDA), an ellipsoid region around the lifting closes for
# comparison. Fit on gepl, test on evr + evi (other policies), demo closes as a reference.
#
#   CUDA_VISIBLE_DEVICES= python tools_c/close_region.py --fit outputs/rollouts/gepl_s101 outputs/rollouts/gepl_s111 \
#       --test outputs/rollouts/evr_s101 outputs/rollouts/evr_s111 outputs/rollouts/evi_s101 outputs/rollouts/evi_s111 \
#       --demos datasets/mimic_500_evt.hdf5
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401

sys.argv, _argv = sys.argv[:1], sys.argv
exec(open("tools_c/plot_gripper_events.py").read().split("def load(run):")[0].split("import numpy as np  # noqa: E402", 1)[1])
sys.argv = _argv

FEAT = ["e_x (cm)", "e_y (cm)", "e_z (cm)", "r_x (deg)", "r_y (deg)", "r_z (deg)", "EE speed (cm/step)"]
NEAR = 0.10


def feats(ge, eef, t):
    sp = np.linalg.norm(eef[t] - eef[max(t - 3, 0)]) / max(min(t, 3), 1)
    return np.r_[ge[t, :3] * 100, np.degrees(ge[t, 3:]), sp * 100]


def ge_ee(d):
    T = lambda k: torch.tensor(np.asarray(d[k]), dtype=torch.float64)  # noqa: E731
    return gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                   os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()


def rollout_closes(runs):
    X, y = [], []
    for r in runs:
        for fn in sorted(glob.glob(f"{r}/rollout_*.npz")):
            d = dict(np.load(fn))
            ge, eef = ge_ee(d), np.asarray(d["eef_pos"])
            tz = d["toy_truck_positions"][:, 2] - d["toy_truck_positions"][0, 2]
            closed = d["actions"][:, 6] < 0
            dd = np.diff(np.r_[0, closed.astype(int), 0])
            for s, e in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
                X.append(feats(ge, eef, s))
                y.append(bool((tz[s : min(e + 1, len(tz))] > LIFT).any()))
    return np.array(X), np.array(y)


def demo_closes(path):
    X = []
    with h5py.File(path, "r") as f:
        for g in f["data"].values():
            c = g["grip_state"][:, 0]
            ge, eef = g["obs"]["grasp_err_ee"][()], g["obs"]["eef_pos"][()]
            X += [feats(ge, eef, s) for s in np.where(np.diff(c) > 0)[0]]  # command at s
    return np.array(X)


def auc(s, y):
    y = np.asarray(y, bool)
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def lda(X, y):
    m1, m0 = X[y].mean(0), X[~y].mean(0)
    Sw = np.cov(X[y].T) * (y.sum() - 1) + np.cov(X[~y].T) * ((~y).sum() - 1)
    w = np.linalg.solve(Sw + 1e-6 * np.eye(len(m1)), m1 - m0)
    return w / np.linalg.norm(w)


def logreg(X, y, l2=1e-2, it=2000):
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=it)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b)


def cv_auc(X, y, fit_score, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    s = np.zeros(len(y))
    for f in range(k):
        te = idx[f::k]
        tr = np.setdiff1d(idx, te)
        s[te] = fit_score(X[tr], y[tr], X[te])
    return auc(s, y)


class Ellipsoid:
    """Mahalanobis distance to the lifting-close cloud; close rule = inside radius r."""

    def __init__(self, X):
        self.m, self.P = X.mean(0), np.linalg.inv(np.cov(X.T) + 1e-6 * np.eye(X.shape[1]))

    def dist(self, X):
        D = X - self.m
        return np.sqrt(np.einsum("ij,jk,ik->i", D, self.P, D))


ap = argparse.ArgumentParser()
ap.add_argument("--fit", nargs="+", required=True)
ap.add_argument("--test", nargs="*", default=[])
ap.add_argument("--demos", default=None)
ap.add_argument("--out", default="results/figs")
ap.add_argument("--md", default="results/event_clf/close_region.md")
A = ap.parse_args()

Xf, yf = rollout_closes(A.fit)
Xt, yt = rollout_closes(A.test) if A.test else (np.zeros((0, len(FEAT))), np.zeros(0, bool))
Xd = demo_closes(A.demos) if A.demos else np.zeros((0, len(FEAT)))
near = lambda X: np.linalg.norm(X[:, :3], axis=1) < NEAR * 100  # noqa: E731
L = ["# 닫힘이 트럭을 드는 영역 (자동 생성: tools_c/close_region.py)", "",
     f"- 학습(fit): {', '.join(os.path.basename(r) for r in A.fit)} 닫힘 {len(yf)}건 (든 것 {yf.sum()}, 못 든 것 {(~yf).sum()})",
     f"- 시험(test): {', '.join(os.path.basename(r) for r in A.test)} 닫힘 {len(yt)}건 (든 것 {int(yt.sum())}, 못 든 것 {int((~yt).sum())})",
     f"- 시범 닫힘 {len(Xd)}건 (모두 성공, 참고용)",
     f"- 잡기 목표 10 cm 밖 닫힘은 거리만으로 100% 구분되므로(전부 실패) 아래 분석은 10 cm 안 닫힘만 쓴다: "
     f"fit {near(Xf).sum()}건 (못 든 것 {(~yf[near(Xf)]).sum()}), test {near(Xt).sum()}건 (못 든 것 {(~yt[near(Xt)]).sum()})", ""]
print("\n".join(L))
Xf, yf, Xt, yt = Xf[near(Xf)], yf[near(Xf)], Xt[near(Xt)], yt[near(Xt)]
mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
Z = lambda X: (X - mu) / sd  # noqa: E731

# 1) per feature
L += ["## 1. 특징별 비교 (10 cm 안 닫힘, fit)", "", "| 특징 | 든 닫힘 중앙값 | 못 든 닫힘 중앙값 | 시범 닫힘 중앙값 | 단변량 AUC (|값| 작을수록 든다) |", "|---|---|---|---|---|"]
for j, n in enumerate(FEAT):
    a = auc(-np.abs(Xf[:, j]), yf)
    L.append(f"| {n} | {np.median(Xf[yf, j]):.2f} | {np.median(Xf[~yf, j]):.2f} | {np.median(Xd[:, j]) if len(Xd) else np.nan:.2f} | {a:.2f} |")
L.append(f"| 위치 오차 크기 |e_p| (cm) | {np.median(np.linalg.norm(Xf[yf, :3], axis=1)):.2f} | {np.median(np.linalg.norm(Xf[~yf, :3], axis=1)):.2f} | "
         f"{np.median(np.linalg.norm(Xd[:, :3], axis=1)) if len(Xd) else np.nan:.2f} | {auc(-np.linalg.norm(Xf[:, :3], axis=1), yf):.2f} |")

# 2) supervised boundaries
w_lda = lda(Z(Xf), yf)
w_lr, b_lr = logreg(Z(Xf), yf)
ell = Ellipsoid(Xf[yf])
sc = {
    "LDA (선형 판별)": (lambda Xa, ya, Xb: Z(Xb) @ lda(Z(Xa), ya), lambda X: Z(X) @ w_lda),
    "로지스틱 회귀": (lambda Xa, ya, Xb: Z(Xb) @ logreg(Z(Xa), ya)[0], lambda X: Z(X) @ w_lr),
    "타원 영역 (든 닫힘 중심에서 마할라노비스 거리)": (lambda Xa, ya, Xb: -Ellipsoid(Xa[ya]).dist(Xb), lambda X: -ell.dist(X)),
    "거리 |e_p| 하나만": (lambda Xa, ya, Xb: -np.linalg.norm(Xb[:, :3], axis=1), lambda X: -np.linalg.norm(X[:, :3], axis=1)),
}
L += ["", "## 2. 지도 학습 경계 (든 / 못 든 구분 AUC)", "", "| 방법 | fit 5겹 교차검증 | test (다른 정책 evr, evi) |", "|---|---|---|"]
for n, (cvf, full) in sc.items():
    L.append(f"| {n} | {cv_auc(Xf, yf, cvf):.2f} | {auc(full(Xt), yt) if len(yt) and (~yt).any() else np.nan:.2f} |")
L += ["", "- LDA 방향 (표준화 계수): " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(FEAT, w_lda)),
      "- 로지스틱 회귀 계수 (표준화): " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(FEAT, w_lr))]

# 4) ellipsoid region as a rule
dl, dn, dd_ = ell.dist(Xf[yf]), ell.dist(Xf[~yf]), ell.dist(Xd) if len(Xd) else np.zeros(0)
L += ["", "## 3. 타원 영역 규칙 (비교용): 든 닫힘 분포의 마할라노비스 거리 r 안이면 닫기", "",
      "| r | fit 든 닫힘 포함 | fit 못 든 닫힘 포함 | test 든 닫힘 포함 | test 못 든 닫힘 포함 | 시범 닫힘 포함 |", "|---|---|---|---|---|---|"]
for r in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    tl = np.mean(ell.dist(Xt[yt]) <= r) if yt.any() else np.nan
    tn = np.mean(ell.dist(Xt[~yt]) <= r) if (~yt).any() else np.nan
    L.append(f"| {r} | {np.mean(dl <= r):.2f} | {np.mean(dn <= r):.2f} | {tl:.2f} | {tn:.2f} | {np.mean(dd_ <= r) if len(dd_) else np.nan:.2f} |")
L += ["", "- 타원 중심 (든 닫힘 평균): " + ", ".join(f"{n} {v:.2f}" for n, v in zip(FEAT, ell.m)),
      "- 각 축 표준편차: " + ", ".join(f"{n} {v:.2f}" for n, v in zip(FEAT, Xf[yf].std(0)))]

# 4) LDA threshold (Youden J on gepl), as a rule
s_f, s_t = Z(Xf) @ w_lda, Z(Xt) @ w_lda
s_d = Z(Xd) @ w_lda if len(Xd) else np.zeros(0)
cand = np.sort(s_f)
thr = cand[np.argmax([np.mean(s_f[yf] >= c) - np.mean(s_f[~yf] >= c) for c in cand])]
L += ["", "## 4. LDA 점수 기준값 (gepl에서 Youden J 최대)", "",
      f"- 기준값 {thr:.2f}: gepl 든 닫힘 {np.mean(s_f[yf] >= thr):.2f} / 못 든 닫힘 {np.mean(s_f[~yf] >= thr):.2f} 이 기준을 넘음",
      f"- test (evr, evi): 든 닫힘 {np.mean(s_t[yt] >= thr):.2f} / 못 든 닫힘 {np.mean(s_t[~yt] >= thr):.2f}",
      f"- 시범 닫힘 {np.mean(s_d >= thr) if len(s_d) else np.nan:.2f}"]

# 2D view: lateral error e_xy and depth e_z (|e_p| would count e_z twice), with a 2-feature LDA boundary fitted on gepl
exy = lambda X: np.linalg.norm(X[:, :2], axis=1)  # noqa: E731
F2 = lambda X: np.c_[exy(X), X[:, 2]]  # noqa: E731
m2, s2 = F2(Xf).mean(0), F2(Xf).std(0)
w2 = lda((F2(Xf) - m2) / s2, yf)
sc2 = ((F2(Xf) - m2) / s2) @ w2
c2 = np.sort(sc2)[np.argmax([np.mean(sc2[yf] >= c) - np.mean(sc2[~yf] >= c) for c in np.sort(sc2)])]
L += ["", "## 5. 두 특징(옆 방향 e_xy, 접근 깊이 e_z) LDA",
      f"- 표준화 계수: e_xy {w2[0]:+.2f}, e_z {w2[1]:+.2f}; 5겹 교차검증 AUC "
      f"{cv_auc(F2(Xf), yf, lambda Xa, ya, Xb: ((Xb - Xa.mean(0)) / Xa.std(0)) @ lda((Xa - Xa.mean(0)) / Xa.std(0), ya)):.2f}"]

# figure (LDA)
GREEN, RED, BLACK, GRAY = "#2CA02C", "#D62728", "black", "#9A9A9A"
os.makedirs(A.out, exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [1.15, 0.85, 1.15]})

bins = np.linspace(min(s_f.min(), s_t.min() if len(s_t) else 0), max(s_f.max(), s_t.max() if len(s_t) else 0), 32)
ax[0].hist(s_f[yf], bins=bins, color=GREEN, alpha=0.75, label=f"gepl 닫힘 → 들림 ({yf.sum()})")
ax[0].hist(s_f[~yf], bins=bins, color=BLACK, alpha=0.75, label=f"gepl 닫힘 → 못 듦 ({(~yf).sum()})")
if len(s_d):
    ax[0].hist(s_d, bins=bins, histtype="step", color=GRAY, linewidth=2, label=f"시범 닫힘 ({len(s_d)}, 모두 성공)")
ax[0].axvline(thr, color="#555555", linestyle="--", linewidth=1.5)
ax[0].text(thr, ax[0].get_ylim()[1] * 0.55, f" 기준값 {thr:.2f}", va="top", ha="left", fontsize=10, color="#333333")
ax[0].set_xlabel("LDA 점수 (클수록 들릴 쪽)")
ax[0].set_ylabel("닫힘 수")
ax[0].set_title(f"LDA 점수 분포 (10 cm 안 닫힘, 교차검증 AUC {cv_auc(Xf, yf, sc['LDA (선형 판별)'][0]):.2f})")
ax[0].legend(fontsize=9, loc="upper right", frameon=False)

o = np.argsort(np.abs(w_lda))
ax[1].barh(np.arange(len(o)), w_lda[o], color="#4C72B0", height=0.6)
ax[1].set_yticks(np.arange(len(o)), [FEAT[i] for i in o])
ax[1].axvline(0, color="#555555", linewidth=1)
for k, i in enumerate(o):
    ax[1].text(w_lda[i] + (0.02 if w_lda[i] >= 0 else -0.02), k, f"{w_lda[i]:+.2f}", va="center",
               ha="left" if w_lda[i] >= 0 else "right", fontsize=9, color="#333333")
ax[1].set_xlim(-1.1, 0.5)
ax[1].set_xlabel("표준화 계수 (음수: 값이 클수록 못 듦)")
ax[1].set_title("LDA 방향: 어느 특징이 구분하나")

for X_, y_, sz, al, lab in [(Xt, yt, 14, 0.35, "evr, evi"), (Xf, yf, 38, 1.0, "gepl")]:
    for m, fc in [(y_, GREEN), (~y_, BLACK)]:
        ax[2].scatter(exy(X_[m]), X_[m, 2], s=sz, facecolor=fc, edgecolor=RED, linewidth=1.0 if al == 1 else 0.6, alpha=al, zorder=3 if al == 1 else 2)
xx = np.linspace(0, 10, 200)
# 2-feature LDA boundary: w2 . ((x - m2) / s2) = c2  ->  e_z as a function of |e_p|
zz = m2[1] + s2[1] * (c2 - w2[0] * (xx - m2[0]) / s2[0]) / w2[1]
ax[2].plot(xx, zz, color="#333333", linewidth=2, label="LDA 경계 (e_xy, e_z; gepl로 학습)")
ax[2].plot([0, 4, 4], [3, 3, -10], color="#333333", linestyle="--", linewidth=1.5, label="축별 경계 영역 (e_xy<4, e_z<3)")
ax[2].set_xlim(0, 10)
ax[2].set_ylim(-6, 8)
ax[2].set_xlabel("옆 방향 오차 e_xy = √(e_x²+e_y²) (cm)")
ax[2].set_ylabel("접근 깊이 e_z (cm, +: 덜 들어감)")
ax[2].set_title("구분 경계 (큰 점 gepl, 작은 점 evr·evi)")
ax[2].scatter([], [], s=38, facecolor=GREEN, edgecolor=RED, label="닫힘 → 들림")
ax[2].scatter([], [], s=38, facecolor=BLACK, edgecolor=RED, label="닫힘 → 못 듦")
ax[2].legend(fontsize=9, loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("gepl 닫힘: 트럭이 들리는 닫힘과 못 드는 닫힘을 가르는 선형 판별(LDA)", x=0.01, ha="left", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(A.out, "gepl_close_region.png"), dpi=130)
L += ["", f"그림: {os.path.join(A.out, 'gepl_close_region.png')}"]
open(A.md, "w").write("\n".join(L) + "\n")
print("\n".join(L[6:]))
