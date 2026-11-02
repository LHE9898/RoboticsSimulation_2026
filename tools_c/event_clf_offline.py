# Stage A (offline, CPU): gripper event classifier vs the GMM policy's implicit switch probability.
# Pre-registered plan and criteria: results/event_clf/plan_A.md
#
#   CUDA_VISIBLE_DEVICES= python tools_c/event_clf_offline.py --data datasets/mimic_500_rel.hdf5 \
#       --ckpts <gepl_s101 .pth> <gepl_s111 .pth> --rollouts outputs/rollouts/gepl_s101 outputs/rollouts/gepl_s111
import argparse
import glob
import os

import h5py
import numpy as np
import torch
import torch.nn as nn

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401

# shared helpers (gf, CFG, LIFT, events, colors, Korean font) from the gripper figure script
exec(open("tools_c/plot_gripper_events.py").read().split("def load(run):")[0].split("import numpy as np  # noqa: E402", 1)[1])

KEYS = ["box_orientations", "box_positions", "eef_pos", "eef_quat", "grasp_err_ee", "gripper_pos", "place_err_ee",
        "toy_truck_orientations", "toy_truck_positions"]  # gepl policy obs (32D)
WIN, TOL, TAU, M = 5, 10, 0.5, 3  # label window, timing tolerance, threshold, consecutive steps (fixed in plan_A.md)
FAR = 0.10


# ---------- data ----------
def with_rel_obs(d):
    """Rollout npz -> obs dict with the two EE-frame features (demos already store them)."""
    o = {k: np.asarray(d[k]) for k in KEYS if k in d}
    T = lambda k: torch.tensor(np.asarray(d[k]), dtype=torch.float64)  # noqa: E731
    if "grasp_err_ee" not in o:
        o["grasp_err_ee"] = gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                                    os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()
    if "place_err_ee" not in o:
        o["place_err_ee"] = gf.place_err_from_poses(T("eef_pos"), T("eef_quat"), T("box_positions"), os.path.join(CFG, "place_offset.json"), "ee").numpy()
    return o


def episode(o, a):
    x = np.concatenate([np.asarray(o[k], dtype=np.float64).reshape(len(a), -1) for k in KEYS], 1)
    dx = np.diff(x, axis=0, prepend=x[:1])
    cmd = np.asarray(a)[:, 6] < 0
    c = np.r_[False, cmd[:-1]]  # commanded state before step t (episodes start open)
    sw = np.where(cmd != c)[0]  # switch steps
    i = np.searchsorted(sw, np.arange(len(a)))  # next switch at or after t
    nxt = np.where(i < len(sw), np.r_[sw, 10**9][i], 10**9)
    y = (nxt - np.arange(len(a))) <= WIN
    ep = np.linalg.norm(np.asarray(o["grasp_err_ee"])[:, :3], axis=1)
    xy_eb = np.linalg.norm(np.asarray(o["box_positions"])[:, :2] - np.asarray(o["eef_pos"])[:, :2], axis=1)  # EE-box center, horizontal
    return {"x": x, "dx": dx, "c": c, "y": y, "sw": sw, "ep": ep, "xy_eb": xy_eb, "o": o, "a": np.asarray(a)}


def load_demos(path):
    with h5py.File(path, "r") as f:
        return [episode({k: g["obs"][k][()] for k in KEYS}, g["actions"][()]) for g in (f["data"][n] for n in sorted(f["data"], key=lambda s: int(s[5:])))]


# ---------- classifier ----------
class EventNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, z, c):  # logit of the head that applies to the current state (0: close-event, 1: open-event)
        return self.f(z).gather(1, c.long().unsqueeze(1)).squeeze(1)


def stack(eps):
    z = np.concatenate([np.c_[e["x"], e["dx"]] for e in eps])
    return z, np.concatenate([e["c"] for e in eps]), np.concatenate([e["y"] for e in eps])


def train(eps, seed=0, epochs=8):
    torch.manual_seed(seed)
    z, c, y = stack(eps)
    mu, sd = z.mean(0), z.std(0) + 1e-6
    Z, C, Y = torch.tensor((z - mu) / sd, dtype=torch.float32), torch.tensor(c), torch.tensor(y, dtype=torch.float32)
    pw = torch.tensor([((~y) & (c == h)).sum() / max(((y) & (c == h)).sum(), 1) for h in (0, 1)], dtype=torch.float32)
    net = EventNet(z.shape[1] // 2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(Z))
        tot = 0.0
        for i in range(0, len(Z), 2048):
            b = perm[i : i + 2048]
            logit = net(Z[b], C[b])
            w = torch.where(Y[b] > 0, pw[C[b].long()], torch.ones_like(logit))  # per-head positive weight
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, Y[b], reduction="none") * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        print(f"  classifier epoch {ep + 1}/{epochs}: loss {tot / len(Z):.4f}")
    net.eval()

    @torch.no_grad()
    def predict(e):
        zz = torch.tensor((np.c_[e["x"], e["dx"]] - mu) / sd, dtype=torch.float32)
        return torch.sigmoid(net(zz, torch.tensor(e["c"]))).numpy()

    return predict, pw.numpy()


# ---------- GMM ----------
@torch.no_grad()
def gmm_pclose(policy, o, n):
    algo = policy.policy
    net, H = algo.nets["policy"], algo._rnn_horizon
    out, state = np.zeros(n), None
    for t in range(n):
        if t % H == 0:
            state = net.get_rnn_init_state(batch_size=1, device=torch.device("cpu"))
        ad, state = net.forward_train_step(policy._prepare_observation({k: o[k][t] for k in algo.global_config.all_obs_keys}), rnn_state=state)
        pi = ad.mixture_distribution.probs.reshape(-1)
        mu = ad.component_distribution.base_dist.loc.reshape(len(pi), -1)[:, 6]
        out[t] = float((pi * (mu < 0).to(pi.dtype)).sum())
    return out


def hazard(pc, c):
    """GMM per-step switch probability: P(close) while open, P(open) while closed."""
    return np.where(c, 1 - pc, pc)


# ---------- metrics ----------
def ap(score, y):
    o = np.argsort(-score, kind="stable")
    y = y[o].astype(float)
    if y.sum() == 0:
        return np.nan
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = np.empty(len(score))
    o = np.argsort(score, kind="stable")
    s = np.asarray(score)[o]
    ranks = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = (i + j + 2) / 2
        i = j + 1
    r[o] = ranks
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def runs_trigger(p, tau=TAU):
    """Step at which p > tau has held for M consecutive steps (the rule fires), as a bool array."""
    f = (p > tau).astype(int)
    cs = np.convolve(f, np.ones(M, int), "full")[: len(f)]
    return cs >= M


def timing(e, p, h, tau=(TAU, TAU)):
    """For every true switch: classifier outcome and GMM outcome probabilities (early / on time / miss).
    tau = (close threshold, open threshold)."""
    fires = (runs_trigger(p, tau[0]), runs_trigger(p, tau[1]))
    out = []
    for i, s in enumerate(e["sw"]):
        start = e["sw"][i - 1] + 1 if i else 0
        end = min(s + TOL, (e["sw"][i + 1] - 1) if i + 1 < len(e["sw"]) else len(p) - 1)
        tr = np.where(fires[int(e["c"][s])][start : end + 1])[0]
        k = "miss" if not len(tr) else ("early" if start + tr[0] < s - TOL else "on")
        hh = h[start : end + 1]
        surv = np.cumprod(np.r_[1.0, 1 - hh[:-1]])
        first = hh * surv
        tt = np.arange(start, end + 1)
        out.append({"dir": "close" if not e["c"][s] else "open", "clf": k, "t_trig": start + tr[0] if len(tr) else None, "s": s,
                    "g_early": float(first[tt < s - TOL].sum()), "g_on": float(first[tt >= s - TOL].sum()), "g_miss": float(np.prod(1 - hh))})
    return out


def far_alarms(e, p, h):
    m = (~e["c"]) & (e["ep"] > FAR)
    fire = runs_trigger(p)
    onset = fire & ~np.r_[False, fire[:-1]]
    return int((onset & m).sum()), float(h[m].sum())


# ---------- main ----------
ap_ = argparse.ArgumentParser()
ap_.add_argument("--data", default="datasets/mimic_500_rel.hdf5")
ap_.add_argument("--ckpts", nargs="+", required=True)
ap_.add_argument("--rollouts", nargs="+", required=True, help="same order as --ckpts")
ap_.add_argument("--out", default="results/event_clf")
ap_.add_argument("--threads", type=int, default=6)
ap_.add_argument("--explore", action="store_true", help="post-hoc: thresholds picked on the training demos")
A = ap_.parse_args()
torch.set_num_threads(A.threads)
os.makedirs(A.out, exist_ok=True)

demos = load_demos(A.data)
rng = np.random.default_rng(0)
perm = rng.permutation(len(demos))
val_idx, tr_idx = np.sort(perm[:100]), np.sort(perm[100:])
tr, va = [demos[i] for i in tr_idx], [demos[i] for i in val_idx]
print(f"demos: train {len(tr)}, val {len(va)}; positive rate (train) close {np.mean(np.concatenate([e['y'][~e['c']] for e in tr])):.4f}"
      f", open {np.mean(np.concatenate([e['y'][e['c']] for e in tr])):.4f}")
predict, pw = train(tr)
np.save(os.path.join(A.out, "val_demo_ids.npy"), val_idx)
if A.explore:  # post-hoc, not a pre-registered criterion: threshold per event picked on TRAIN demos, reported on VAL demos
    zero = lambda e: np.zeros(len(e["a"]))  # noqa: E731
    Ptr = [(e, predict(e)) for e in tr]
    best = []
    for hd in (0, 1):
        grid = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
        on = [np.mean([t["clf"] == "on" for e, p in Ptr for t in timing(e, p, zero(e), (g, g)) if t["dir"] == ("close", "open")[hd]]) for g in grid]
        best.append(grid[int(np.argmax(on))])
        print(f"  [explore] {('close', 'open')[hd]}: train on-time by tau " + ", ".join(f"{g}:{o:.2f}" for g, o in zip(grid, on)))
    ex = ["", "## 사후 탐색 (사전 기준 아님): 기준값을 학습용 시범 400개에서 고른 경우", "",
          f"- 선택된 기준값: 닫기 {best[0]}, 열기 {best[1]} ({M} step 연속). 아래는 검증 시범 100개 결과. GMM은 실행 규칙이 같으므로 위 표의 기댓값과 같다.", "",
          "| 사건 | 제때 | 너무 일찍 | 안 바꿈 | 바꾼 순간 거리 (중앙값): 분류기 / 실제 시범 |", "|---|---|---|---|---|"]
    for hd in ("close", "open"):
        dk = "ep" if hd == "close" else "xy_eb"  # close: EE->grasp target, open: EE->box center (xy)
        T2 = [t for e in va for t in [dict(t, ep_trig=e[dk][t["t_trig"]] if t["t_trig"] is not None else np.nan, ep_true=e[dk][t["s"]])
                                       for t in timing(e, predict(e), zero(e), tuple(best))] if t["dir"] == hd]
        ex.append(f"| {hd} | {np.mean([t['clf'] == 'on' for t in T2]):.2f} | {np.mean([t['clf'] == 'early' for t in T2]):.2f}"
                  f" | {np.mean([t['clf'] == 'miss' for t in T2]):.2f}"
                  f" | {('잡기 목표', '박스 중심 xy')[hd == 'open']} {np.nanmedian([t['ep_trig'] for t in T2]) * 100:.1f} cm / {np.nanmedian([t['ep_true'] for t in T2]) * 100:.1f} cm |")
        if hd == "close":
            early = [t for t in T2 if t["clf"] == "early"]
            if early:
                ex.append(f"| close, 너무 일찍 바꾼 경우만 (n={len(early)}) | | | | {np.nanmedian([t['ep_trig'] for t in early]) * 100:.1f} cm / {np.nanmedian([t['ep_true'] for t in early]) * 100:.1f} cm |")
    print("\n".join(ex))

import robomimic.utils.file_utils as FileUtils  # noqa: E402

lines = ["# A단계 결과 (자동 생성: tools_c/event_clf_offline.py)", "",
         f"- 검증 시범 {len(va)}개 (Mimic 500개 중 고정 seed 0으로 추출). 분류기는 나머지 400개로만 학습.",
         f"- 실행 규칙: 분류기 p>{TAU} {M} step 연속, GMM은 매 step 무작위 (현재 평가와 같음). 시점 허용 ±{TOL} step.", ""]
summary = {}
for ck, run in zip(A.ckpts, A.rollouts):
    name = os.path.basename(os.path.normpath(run))
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ck, device=torch.device("cpu"), verbose=False)
    # A1-A3 on validation demos (teacher-forced along demo states)
    P, Hh, Y, Cc, T, FA = [], [], [], [], [], []
    for e in va:
        p = predict(e)
        h = hazard(gmm_pclose(policy, e["o"], len(e["a"])), e["c"])
        P.append(p), Hh.append(h), Y.append(e["y"]), Cc.append(e["c"])
        T += timing(e, p, h)
        FA.append(far_alarms(e, p, h))
    P, Hh, Y, Cc = map(np.concatenate, (P, Hh, Y, Cc))
    r = {}
    for hd, mask in (("close", ~Cc), ("open", Cc)):
        r[f"A1_{hd}"] = (ap(P[mask], Y[mask]), ap(Hh[mask], Y[mask]), auc(P[mask], Y[mask]), auc(Hh[mask], Y[mask]), float(Y[mask].mean()))
        tt = [t for t in T if t["dir"] == hd]
        r[f"A2_{hd}"] = (np.mean([t["clf"] == "on" for t in tt]), np.mean([t["g_on"] for t in tt]),
                         np.mean([t["clf"] == "early" for t in tt]), np.mean([t["g_early"] for t in tt]),
                         np.mean([t["clf"] == "miss" for t in tt]), np.mean([t["g_miss"] for t in tt]), len(tt))
    r["A3"] = (np.mean([f[0] for f in FA]), np.mean([f[1] for f in FA]))
    # A4 on gepl rollout states (closed-loop, off the demo distribution)
    cl_s, cl_y, op_s, op_y = [], [], [], []
    for f in sorted(glob.glob(f"{run}/rollout_*.npz")):
        d = dict(np.load(f))
        e = episode(with_rel_obs(d), d["actions"])
        p = predict(e)
        pc = gmm_pclose(policy, e["o"], len(e["a"]))
        _, segs, closes, opens, z = events(d)
        xy_tb = np.linalg.norm(d["toy_truck_positions"][:, :2] - d["box_positions"][:, :2], axis=1)
        for t, lifted in closes:
            cl_s.append((p[t], pc[t])), cl_y.append(lifted)
        for t in opens:
            if z[t] > LIFT:
                op_s.append((p[t], 1 - pc[t])), op_y.append(xy_tb[t] < 0.10)
    cl_s, op_s, cl_y, op_y = np.array(cl_s), np.array(op_s), np.array(cl_y), np.array(op_y)
    r["A4_close"] = (auc(cl_s[:, 0], cl_y), auc(cl_s[:, 1], cl_y), len(cl_y), int(cl_y.sum()),
                     float(np.mean(cl_s[cl_y, 0] > TAU)), float(np.mean(cl_s[~cl_y, 0] > TAU)))
    r["A4_open"] = (auc(op_s[:, 0], op_y), auc(op_s[:, 1], op_y), len(op_y), int(op_y.sum()),
                    float(np.mean(op_s[op_y, 0] > TAU)), float(np.mean(op_s[~op_y, 0] > TAU)))
    summary[name] = r
    print(name, {k: tuple(np.round(v, 3)) if isinstance(v, tuple) else v for k, v in r.items()})

    lines += [f"## {name}", "",
              "| 지표 | 사건 | 분류기 | GMM | 기준 |", "|---|---|---|---|---|"]
    for hd in ("close", "open"):
        a1 = r[f"A1_{hd}"]
        lines.append(f"| A1 PR-AUC (양성 비율 {a1[4]:.3f}) | {hd} | {a1[0]:.3f} | {a1[1]:.3f} | {'PASS' if a1[0] > a1[1] else 'FAIL'} |")
        lines.append(f"| (참고) ROC-AUC | {hd} | {a1[2]:.3f} | {a1[3]:.3f} | |")
    for hd in ("close", "open"):
        a2 = r[f"A2_{hd}"]
        lines.append(f"| A2 제때 바꿈 (n={a2[6]}) | {hd} | {a2[0]:.2f} | {a2[1]:.2f} (기댓값) | {'PASS' if a2[0] > a2[1] else 'FAIL'} |")
        lines.append(f"| (참고) 너무 일찍 / 안 바꿈 | {hd} | {a2[2]:.2f} / {a2[4]:.2f} | {a2[3]:.2f} / {a2[5]:.2f} | |")
    lines.append(f"| A3 먼 곳(>10 cm) 오작동 닫힘 / 에피소드 | close | {r['A3'][0]:.3f} | {r['A3'][1]:.3f} (기댓값) | {'PASS' if r['A3'][0] < r['A3'][1] else 'FAIL'} |")
    c4, o4 = r["A4_close"], r["A4_open"]
    lines.append(f"| A4 rollout: 든 닫힘 대 안 든 닫힘 ROC-AUC (n={c4[2]}, 든 것 {c4[3]}) | close | {c4[0]:.3f} | {c4[1]:.3f} | {'PASS' if c4[0] > c4[1] else 'FAIL'} |")
    lines.append(f"| (참고) 분류기 p>{TAU} 비율: 든 닫힘 / 안 든 닫힘 | close | {c4[4]:.2f} / {c4[5]:.2f} | | |")
    lines.append(f"| A4 rollout: 박스 안 놓기 대 밖 ROC-AUC (n={o4[2]}, 안 {o4[3]}) | open | {o4[0]:.3f} | {o4[1]:.3f} | {'PASS' if o4[0] > o4[1] else 'FAIL'} |")
    lines.append(f"| (참고) 분류기 p>{TAU} 비율: 박스 안 / 밖 | open | {o4[4]:.2f} / {o4[5]:.2f} | | |")
    lines.append("")

    # figure: two validation demos, classifier p vs GMM switch probability
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.6), sharex=False)
    for ax, e in zip(axs, (va[0], va[1])):
        p, h = predict(e), hazard(gmm_pclose(policy, e["o"], len(e["a"])), e["c"])
        for s, t in zip(*[np.where(np.diff(np.r_[0, e['c'].astype(int), 0]) == v)[0] for v in (1, -1)]):
            ax.axvspan(s, t, color=BAND, lw=0, zorder=0)
        ax.plot(h, color="#555555", lw=1.5, label="GMM: 매 step 바꿀 확률")
        ax.plot(p, color="#e07b00", lw=1.5, label="사건 분류기: 바꿀 확률")
        ax.axhline(TAU, color="#999999", lw=0.8, ls="--")
        for s in e["sw"]:
            ax.axvline(s, color=RED if not e["c"][s] else BLUE, lw=1.2, ls=":")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("확률")
        ax.set_title(f"검증 시범 (빨강 점선 = 실제 닫은 step, 파랑 점선 = 실제 연 step, 분홍 = 닫혀 있던 구간)", loc="left", fontsize=10)
    axs[0].legend(loc="upper left", frameon=False)
    axs[-1].set_xlabel("step")
    fig.suptitle(f"{name} GMM 대 사건 분류기: 검증 시범에서의 그리퍼 전환 확률", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join("results/figs", f"event_clf_val_{name}.png"), dpi=150)
    plt.close(fig)

open(os.path.join(A.out, "results_A.md"), "w").write("\n".join(lines + (ex if A.explore else [])))
print("\n".join(lines))
