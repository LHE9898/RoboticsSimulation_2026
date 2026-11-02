# CPU check before retraining evr with a wider close label (pre-registered in results/event_clf/plan_C5.md).
# Trains the event classifier only (same EventNet recipe as stage A) on two close labels and replays the recorded evr rollouts:
#   ready  close positives = |e_G,p| <= |e_G,p(s)| + 1 cm (<= 40 steps) + the last 5 steps   (current evr)
#   abs5   close positives = |e_G,p| <= 5 cm, contiguous back from the switch                 + the last 5 steps
#
#   CUDA_VISIBLE_DEVICES= python tools_c/check_close5.py --data datasets/mimic_500_evt.hdf5 \
#       --rollouts outputs/rollouts/evr_s101 outputs/rollouts/evr_s111 --evr_runs <evr_seed101 run dir> <evr_seed111 run dir>
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.argv, _argv = sys.argv[:1], sys.argv
exec(open("tools_c/event_clf_offline.py").read().split("# ---------- GMM ----------")[0])  # KEYS, with_rel_obs, episode, EventNet
sys.argv = _argv

ABS = 0.05
LIFT_RATE = [(0.02, 0.98), (0.03, 1.00), (0.04, 0.87), (0.05, 0.88), (0.07, 0.08), (np.inf, 0.0)]  # evr rollouts, plan_C5.md
GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99]
TOL = 10


def close_abs5(e):
    """Close positives of the abs5 label; open positives are taken from the stored ready label."""
    y = np.zeros(len(e["c"]), bool)
    sw = e["sw"]
    for i, s in enumerate(sw):
        if e["c"][s]:  # open event: handled by the ready label
            continue
        lo = sw[i - 1] + 1 if i else 0
        t = s
        while t - 1 >= lo and e["ep"][t - 1] <= ABS:
            t -= 1
        y[min(t, max(lo, s - WIN)) : s + 1] = True
    return y


def load(path):
    with h5py.File(path, "r") as f:
        names = sorted(f["data"], key=lambda s: int(s[5:]))
        eps = []
        for n in names:
            g = f["data"][n]
            e = episode({k: g["obs"][k][()] for k in KEYS}, g["actions"][()])
            ready = g["grip_event_ready"][:, 0] > 0
            e["y_ready"] = ready
            e["y_abs5"] = np.where(e["c"], ready, close_abs5(e))
            eps.append(e)
    return eps


def fires(p, tau):
    f = (p > tau).astype(int)
    return np.convolve(f, np.ones(M, int), "full")[: len(f)] >= M


def calibrate(eps, prob, key):
    """calibrate_event_tau.py rule on the close head: first fire inside [label window start, s + TOL]."""
    rows = []
    for tau in GRID:
        out = []
        for e, p in zip(eps, prob):
            sw = e["sw"]
            for i, s in enumerate(sw):
                if e["c"][s]:
                    continue
                lo = sw[i - 1] + 1 if i else 0
                hi = min(s + TOL, sw[i + 1] if i + 1 < len(sw) else len(p) - 1)
                w = s
                while w - 1 >= lo and e[key][w - 1]:
                    w -= 1
                tr = np.where(fires(p[lo : hi + 1], tau))[0]
                out.append("miss" if not len(tr) else ("early" if lo + tr[0] < w else "on"))
        rows.append((tau, np.mean([o == "on" for o in out]), np.mean([o == "early" for o in out])))
    return max(rows, key=lambda r: (r[1], -r[2]))


def lift_rate(d):
    return next(r for b, r in LIFT_RATE if d < b)


def replay(rollouts, predict, tau):
    """Close head on recorded evr rollouts (open steps only, before the first close command)."""
    stuck, lifted, nofire = [], [], []
    for e in rollouts:
        p = predict(e)
        closed = e["a"][:, 6] < 0
        t_close = int(np.argmax(closed)) if closed.any() else len(p)
        f = fires(p[:t_close], tau)
        t_fire = int(np.argmax(f)) if f.any() else None
        if not closed.any() and e["ep"].min() < 0.03:  # stuck near the target, never closed
            near = e["ep"] < 0.03
            hit = [t for t in np.where(f)[0] if near[t]]
            stuck.append(bool(hit))
        elif e["lift"]:
            d = e["ep"][t_fire] if t_fire is not None else e["ep"][t_close]  # no earlier fire -> the recorded close stands
            lifted.append(d)
            nofire.append(t_fire is None)
    lifted = np.array(lifted)
    return {"stuck_fired": int(np.sum(stuck)), "stuck_n": len(stuck), "lifted_n": len(lifted),
            "fire_dist_median_cm": float(np.median(lifted) * 100), "beyond5": float(np.mean(lifted > 0.05)),
            "beyond10": float(np.mean(lifted > 0.10)), "exp_lift": float(np.mean([lift_rate(d) for d in lifted])),
            "no_fire_before_close": float(np.mean(nofire))}


ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--rollouts", nargs="+", required=True)
ap.add_argument("--evr_runs", nargs="*", default=[])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--out", default="results/event_clf/results_C5.md")
A = ap.parse_args()
torch.set_num_threads(6)

demos = load(A.data)
rolls = []
for r in A.rollouts:
    for fn in sorted(glob.glob(f"{r}/rollout_*.npz")):
        d = dict(np.load(fn))
        e = episode(with_rel_obs(d), d["actions"])
        tz = d["toy_truck_positions"][:, 2] - d["toy_truck_positions"][0, 2]
        cl = e["a"][:, 6] < 0
        e["lift"] = bool(cl.any() and (tz[int(np.argmax(cl)) : int(np.argmax(cl)) + 100] > 0.02).any())  # first close lifted
        rolls.append(e)
print(f"demos {len(demos)} | rollouts {len(rolls)}")
for key in ("y_ready", "y_abs5"):
    y = np.concatenate([e[key] for e in demos])[~np.concatenate([e["c"] for e in demos])]
    print(f"  {key}: close-head positive rate {y.mean():.4f}")

results = []
for key in ("y_ready", "y_abs5"):
    for seed in A.seeds:
        print(f"== {key} seed {seed}")
        for e in demos:
            e["y"] = e[key]
        predict, _ = train(demos, seed=seed, epochs=A.epochs)
        prob = [predict(e) for e in demos]
        tau, on, early = calibrate(demos, prob, key)
        m = replay(rolls, predict, tau)
        results.append({"label": key[2:], "model": f"CPU seed {seed}", "tau": tau, "train_on_time": on, "train_early": early, **m})
        print(results[-1])

# reference: the trained evr event heads with their calibrated tau
if A.evr_runs:
    sys.path.insert(0, "tools_c")
    import bc_event  # noqa: F401
    import robomimic.utils.file_utils as FileUtils

    for run in A.evr_runs:
        pol, _ = FileUtils.policy_from_checkpoint(ckpt_path=os.path.join(run, "models/model_epoch_2000.pth"), device=torch.device("cpu"), verbose=False)
        algo = pol.policy
        tau = json.load(open(os.path.join(run, "event_tau.json")))["tau"][0]

        @torch.no_grad()
        def predict(e, algo=algo):
            x = torch.tensor(e["x"], dtype=torch.float32)
            return algo.event_prob(x, torch.tensor(e["dx"], dtype=torch.float32), torch.tensor(e["c"], dtype=torch.float32)).numpy()

        m = replay(rolls, predict, tau)
        results.append({"label": "ready", "model": f"evr GPU {os.path.basename(os.path.dirname(run.rstrip('/')))[-7:]}", "tau": tau,
                        "train_on_time": np.nan, "train_early": np.nan, **m})
        print(results[-1])

# judgement (plan_C5.md)
cpu_ready = [r for r in results if r["label"] == "ready" and r["model"].startswith("CPU")]
abs5 = [r for r in results if r["label"] == "abs5"]
crit = [
    ("V. control (ready, CPU) closes in <= 5 of the stuck episodes", all(r["stuck_fired"] <= 5 for r in cpu_ready)),
    ("C1. abs5 closes near the target in >= 10 of the stuck episodes (every seed)", all(r["stuck_fired"] >= 10 for r in abs5)),
    ("C2. abs5 first close beyond 5 cm <= 5% and expected lift >= 0.93", all(r["beyond5"] <= 0.05 and r["exp_lift"] >= 0.93 for r in abs5)),
    ("C3. abs5 first close beyond 10 cm <= 2%", all(r["beyond10"] <= 0.02 for r in abs5)),
]
cols = ["label", "model", "tau", "train_on_time", "train_early", "stuck_fired", "stuck_n", "lifted_n", "fire_dist_median_cm",
        "beyond5", "beyond10", "exp_lift", "no_fire_before_close"]
fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
lines = ["# 잡기 정답 5 cm 검증 결과 (자동 생성: tools_c/check_close5.py, 계획: plan_C5.md)", "",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in results]
lines += ["", "- stuck_fired: 버틴 13건 중 목표 3 cm 안에서 닫기 판단이 발동한 건수",
          "- fire_dist / beyond5 / beyond10 / exp_lift: evr이 트럭을 든 에피소드에서 처음 닫기 판단이 나온 순간의 잡기 목표 거리와, 그 거리로 환산한 들어 올림 기대 비율",
          "- no_fire_before_close: 기록된 닫힘 이전에 한 번도 발동하지 않은 비율 (이 경우 기록된 닫힘 거리를 쓴다; 실제로는 늦게 닫거나 버틸 수 있음)", ""]
lines += [f"- [{'PASS' if ok else 'FAIL'}] {t}" for t, ok in crit]
open(A.out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
