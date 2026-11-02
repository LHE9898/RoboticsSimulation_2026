# Gripper open/close timing figures for recorded rollouts (CPU only).
#   red circle = close command (fill green: this close lifted the truck, fill black: it did not), blue circle = open command
#
#   CUDA_VISIBLE_DEVICES= python tools_c/plot_gripper_events.py --runs outputs/rollouts/gepl_s101 outputs/rollouts/gepl_s111 \
#       --ckpts <gepl_s101 model_epoch_2000.pth> <gepl_s111 model_epoch_2000.pth> --out results/figs
import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

exec(open("tools_c/gripper_diag.py").read().split("FLICKER = 10")[0].split("import numpy as np", 1)[1])  # gf, CFG, LIFT, analyze_rollout

# Korean labels: register the system Noto CJK collection explicitly (matplotlib does not pick up .ttc files by family)
_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_cjk):
    matplotlib.font_manager.fontManager.addfont(_cjk)
plt.rcParams.update({"font.family": matplotlib.font_manager.FontProperties(fname=_cjk).get_name() if os.path.exists(_cjk) else "sans-serif",
                     "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888888", "axes.labelcolor": "#333333", "xtick.color": "#555555", "ytick.color": "#555555"})
RED, BLUE, GREEN, BLACK, BAND, GRID = "#d62728", "#1f5fbf", "#2ca02c", "#111111", "#f6d5d5", "#e6e6e6"
ORDER = ["success", "approach", "grasp", "transport", "place"]
NAME = {"success": "성공", "approach": "접근 실패", "grasp": "파지 실패", "transport": "운반 실패", "place": "배치 실패"}
IDLE = 0.01  # same as analyze_rollouts: max |a[:6]| below this counts as a stop (policy outputs are never exactly 0)


def events(d):
    """Close / open switch steps from the sign of the gripper command, and whether each closed run lifted the truck."""
    closed = d["actions"][:, 6] < 0
    z = d["toy_truck_positions"][:, 2] - d["toy_truck_positions"][0, 2]
    sw = np.diff(np.r_[0, closed.astype(int), 0])
    segs = list(zip(np.where(sw == 1)[0], np.where(sw == -1)[0]))
    closes = [(int(s), bool((z[s : min(e + 1, len(z))] > LIFT).any())) for s, e in segs]
    opens = [int(e) for s, e in segs if e < len(closed)]
    return closed, segs, closes, opens, z


def load(run):
    eps = []
    for f in sorted(glob.glob(f"{run}/rollout_*.npz")):
        d = dict(np.load(f))
        closed, segs, closes, opens, z = events(d)
        a = d["actions"]
        xy_tb = np.linalg.norm(d["toy_truck_positions"][:, :2] - d["box_positions"][:, :2], axis=1)
        T = lambda k: torch.tensor(d[k], dtype=torch.float64)  # noqa: E731
        ge = gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                     os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()
        eps.append({"k": int(os.path.basename(f)[8:-4]), "d": d, "n": len(a), "ok": bool(d["success"]), "cls": analyze_rollout(d)["cls"],
                    "segs": segs, "closes": closes, "opens": opens, "z": z, "xy_tb": xy_tb, "ep": np.linalg.norm(ge[:, :3], axis=1),
                    "sw": int(np.abs(np.diff(closed.astype(int))).sum()), "idle": float((np.abs(a[:, :6]).max(1) < IDLE).mean())})
    return eps


def mark_close(ax, x, y, lifted, s=34):
    ax.scatter(x, y, s=s, facecolor=GREEN if lifted else BLACK, edgecolor=RED, linewidth=1.4, zorder=3)


def mark_open(ax, x, y, s=34):
    ax.scatter(x, y, s=s, facecolor="white", edgecolor=BLUE, linewidth=1.4, zorder=3)


LEGEND = [
    Line2D([], [], ls="", marker="o", ms=7, mfc=GREEN, mec=RED, mew=1.4, label="닫음 → 파지 성공 (트럭 들림)"),
    Line2D([], [], ls="", marker="o", ms=7, mfc=BLACK, mec=RED, mew=1.4, label="닫음 → 파지 실패 (안 들림)"),
    Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=BLUE, mew=1.4, label="엶"),
]


def fig_raster(name, eps, out):
    """One row per episode, grouped by outcome: when the gripper was closed / opened."""
    rows = sorted(eps, key=lambda e: (ORDER.index(e["cls"]), e["k"]))
    fig, ax = plt.subplots(figsize=(11, 0.15 * len(rows) + 1.8))
    for i, e in enumerate(rows):
        ax.plot([0, e["n"]], [i, i], color=GRID, lw=1, zorder=1)
        for s, t in e["segs"]:
            ax.plot([s, t], [i, i], color=BAND, lw=4, solid_capstyle="butt", zorder=2)
        for t, lifted in e["closes"]:
            mark_close(ax, t, i, lifted, s=22)
        for t in e["opens"]:
            mark_open(ax, t, i, s=22)
    # outcome group labels instead of per-episode ticks
    ticks, labels, start = [], [], 0
    for c in ORDER:
        n = sum(e["cls"] == c for e in rows)
        if n:
            ticks.append(start + (n - 1) / 2)
            labels.append(f"{NAME[c]} ({n})")
            if start:
                ax.axhline(start - 0.5, color="#999999", lw=0.8)
            start += n
    ax.set_yticks(ticks, labels)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 800)
    ax.set_xlabel("step (20 Hz)")
    ax.set_title(f"{name}: 에피소드별 그리퍼 닫음·엶 시점 (분홍 막대 = 닫혀 있던 구간)", loc="left")
    fig.legend(handles=LEGEND, loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(out, f"{name}_gripper_raster.png"), dpi=150)
    plt.close(fig)


def fig_where(name, eps, out):
    """Where each event happened: close -> distance to grasp target, open -> truck-box xy."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for e in eps:
        for t, lifted in e["closes"]:
            mark_close(a1, t, max(e["ep"][t] * 100, 0.3), lifted)
        for t in e["opens"]:
            if e["z"][t] > LIFT:  # release while the truck is held
                mark_open(a2, t, e["xy_tb"][t] * 100)
    a1.set_yscale("log")
    a1.axhline(3, color="#777777", lw=1, ls="--")
    a1.text(8, 3.3, "3 cm", ha="left", va="bottom", color="#555555")
    a1.set_xlabel("닫은 시점 (step)")
    a1.set_ylabel("EE에서 본 잡기 목표까지 거리 (cm, log)")
    a1.set_title("닫은 순간: 잡기 목표까지 거리", loc="left")
    a2.axhline(10, color="#777777", lw=1, ls="--")
    a2.text(8, 10.5, "성공 판정 10 cm", ha="left", va="bottom", color="#555555")
    a2.set_xlabel("연 시점 (step, 트럭을 든 상태에서)")
    a2.set_ylabel("트럭–박스 xy 거리 (cm)")
    a2.set_title("연 순간: 박스까지 거리", loc="left")
    for a in (a1, a2):
        a.set_xlim(0, 800)
        a.grid(axis="y", color=GRID, lw=0.8)
    fig.legend(handles=LEGEND, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{name}: 그리퍼를 바꾼 순간의 위치", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, f"{name}_gripper_where.png"), dpi=150)
    plt.close(fig)


@torch.no_grad()
def p_close(policy, d):
    """P(close)_t = sum_k pi_k [mu_k,grip < 0], replayed with the same 10-step RNN reset as BC_RNN.get_action."""
    algo = policy.policy
    net, keys, H = algo.nets["policy"], algo.global_config.all_obs_keys, algo._rnn_horizon
    T = lambda k: torch.tensor(d[k], dtype=torch.float64)  # noqa: E731
    o = {k: d[k] for k in d if k in keys}
    o["grasp_err_ee"] = gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                                os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()
    o["place_err_ee"] = gf.place_err_from_poses(T("eef_pos"), T("eef_quat"), T("box_positions"), os.path.join(CFG, "place_offset.json"), "ee").numpy()
    out, state = np.zeros(len(d["actions"])), None
    for t in range(len(out)):
        if t % H == 0:
            state = net.get_rnn_init_state(batch_size=1, device=torch.device("cpu"))
        ad, state = net.forward_train_step(policy._prepare_observation({k: o[k][t] for k in keys}), rnn_state=state)
        pi = ad.mixture_distribution.probs.reshape(-1)
        mu = ad.component_distribution.base_dist.loc.reshape(len(pi), -1)[:, 6]
        out[t] = float((pi * (mu < 0).to(pi.dtype)).sum())
    return out


def fig_pclose(name, eps, ckpt, out):
    """Example episodes: the policy's own P(close) next to the executed switches."""
    import robomimic.utils.file_utils as FileUtils

    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=torch.device("cpu"), verbose=False)
    succ = [e for e in eps if e["ok"]]
    pick = [("성공", min(succ, key=lambda e: e["sw"]))]
    pick.append(("공중에서 여러 번 닫음", max(eps, key=lambda e: sum(not l for _, l in e["closes"]))))
    out_rel = [e for e in eps if not e["ok"] and any(e["z"][t] > LIFT and e["xy_tb"][t] >= 0.10 for t in e["opens"])]
    if out_rel:
        pick.append(("박스 밖에서 놓음", out_rel[0]))
    fig, axs = plt.subplots(len(pick), 1, figsize=(11, 2.6 * len(pick)), sharex=True)
    for ax, (lab, e) in zip(np.atleast_1d(axs), pick):
        pc = p_close(policy, e["d"])
        for s, t in e["segs"]:
            ax.axvspan(s, t, color=BAND, lw=0, zorder=0)
        ax.plot(pc, color="#444444", lw=1.5, zorder=2)
        ax.axhline(0.5, color="#999999", lw=0.8, ls="--")
        for t, lifted in e["closes"]:
            mark_close(ax, t, pc[t], lifted)
        for t in e["opens"]:
            mark_open(ax, t, pc[t])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("정책의 P(닫힘)")
        ax.set_title(f"{lab}: episode {e['k']} ({NAME[e['cls']]}, 바꾼 횟수 {e['sw']})", loc="left")
    np.atleast_1d(axs)[-1].set_xlabel("step (분홍 = 실제로 닫혀 있던 구간, 점선 = 0.5)")
    fig.legend(handles=LEGEND, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{name}: 정책이 매긴 닫힘 확률과 실제로 바꾼 순간", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, f"{name}_gripper_pclose.png"), dpi=150)
    plt.close(fig)


def write_table(name, eps, out):
    """Per-episode summary in the same spirit as the demo summary script (len, switches, idle, ok)."""
    with open(os.path.join(out, f"{name}_gripper_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "len", "sec", "sw", "idle", "ok", "class", "close_lifted", "close_not_lifted", "opens", "first_lifting_close_step"])
        for e in eps:
            lift = [t for t, l in e["closes"] if l]
            w.writerow([e["k"], e["n"], round(e["n"] / 20, 1), e["sw"], round(e["idle"], 3), int(e["ok"]), e["cls"],
                        len(lift), sum(not l for _, l in e["closes"]), len(e["opens"]), lift[0] if lift else ""])


ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--ckpts", nargs="*", default=[], help="same order as --runs; enables the P(close) figure")
ap.add_argument("--out", default="results/figs")
ap.add_argument("--threads", type=int, default=4)
A = ap.parse_args()
torch.set_num_threads(A.threads)
os.makedirs(A.out, exist_ok=True)
for i, run in enumerate(A.runs):
    name = os.path.basename(os.path.normpath(run))
    eps = load(run)
    write_table(name, eps, A.out)
    fig_raster(name, eps, A.out)
    fig_where(name, eps, A.out)
    if i < len(A.ckpts):
        fig_pclose(name, eps, A.ckpts[i], A.out)
    print(f"{name}: {len(eps)} episodes -> {A.out}/{name}_gripper_*.png, _episodes.csv")
