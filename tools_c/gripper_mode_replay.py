# Replay recorded rollouts through the GMM policy on CPU and read out the gripper decision it actually encodes:
#   P(close)_t = sum_k pi_k(h_t) * [mu_k,grip(h_t) < 0]
# The executed action samples one mode k ~ pi each step, so an intermediate P(close) means the open/close choice is a coin flip.
#
#   CUDA_VISIBLE_DEVICES= python tools_c/gripper_mode_replay.py --ckpt <model_epoch_2000.pth> --rollouts outputs/rollouts/gepl_s101
import argparse
import glob
import os

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

exec(open("tools_c/gripper_diag.py").read().split("FLICKER = 10")[0].split("import numpy as np", 1)[1])  # gf, CFG, LIFT, analyze_rollout
FLICKER = 10

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--rollouts", nargs="+", required=True)
ap.add_argument("--threads", type=int, default=4)
A = ap.parse_args()
torch.set_num_threads(A.threads)

policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=A.ckpt, device=torch.device("cpu"), verbose=False)
algo = policy.policy
net = algo.nets["policy"]
keys = algo.global_config.all_obs_keys
H = algo._rnn_horizon


def obs_seq(d):
    T = lambda k: torch.tensor(d[k], dtype=torch.float64)  # noqa: E731
    o = {k: d[k] for k in d if k in keys}
    if "grasp_err_ee" in keys:
        o["grasp_err_ee"] = gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                                    os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()
    if "place_err_ee" in keys:
        o["place_err_ee"] = gf.place_err_from_poses(T("eef_pos"), T("eef_quat"), T("box_positions"),
                                                    os.path.join(CFG, "place_offset.json"), "ee").numpy()
    assert set(keys) <= set(o), set(keys) - set(o)
    return o


@torch.no_grad()
def p_close(d):
    o = obs_seq(d)
    n = len(d["actions"])
    out = np.zeros(n)
    state = None
    for t in range(n):
        ob = policy._prepare_observation({k: o[k][t] for k in keys})
        if t % H == 0:  # same reset schedule as BC_RNN.get_action
            state = net.get_rnn_init_state(batch_size=1, device=torch.device("cpu"))
        ad, state = net.forward_train_step(ob, rnn_state=state)
        pi = ad.mixture_distribution.probs.reshape(-1)
        mu = ad.component_distribution.base_dist.loc.reshape(len(pi), -1)[:, 6]
        out[t] = float((pi * (mu < 0).double() if pi.dtype == torch.float64 else pi * (mu < 0).float()).sum())
    return out


rows = []
for run in A.rollouts:
    for f in sorted(glob.glob(f"{run}/rollout_*.npz")):
        d = dict(np.load(f))
        pc = p_close(d)
        closed = d["actions"][:, 6] < 0
        z = d["toy_truck_positions"][:, 2] - d["toy_truck_positions"][0, 2]
        rows.append({"pc": pc, "closed": closed, "z": z, "cls": analyze_rollout(d)["cls"], "f": f})

PC = np.concatenate([r["pc"] for r in rows])
CL = np.concatenate([r["closed"] for r in rows])
print(f"{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(A.ckpt))))}: {len(rows)} rollouts, {len(PC)} steps, RNN reset every {H} steps")
print(f"- executed close matches P(close)>0.5: {np.mean(CL == (PC > 0.5)) * 100:.1f}% of steps")
print(f"- uncertain steps 0.1<P(close)<0.9: {np.mean((PC > 0.1) & (PC < 0.9)) * 100:.2f}% of steps")
# classify every close event (open->closed switch)
ev = {"flicker": [], "lifting": [], "other": []}
for r in rows:
    c = r["closed"]
    d = np.diff(np.r_[0, c.astype(int), 0])
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        lift = (r["z"][s : min(e + 1, len(r["z"]))] > LIFT).any()
        k = "lifting" if lift else ("flicker" if e - s < FLICKER else "other")
        ev[k].append(r["pc"][s])
for k, v in ev.items():
    v = np.array(v)
    if len(v):
        print(f"- close events [{k}] n={len(v)}: P(close) at the switch step median {np.median(v):.2f} | <0.5 (minority mode sampled) {np.mean(v < 0.5) * 100:.0f}%"
              f" | <0.2 {np.mean(v < 0.2) * 100:.0f}%")

# P(close) around switches and as a function of the distance to the grasp target (open phase before the first lift)
prof_c, prof_r, rel_pc, by_d = [], [], [], []
for r in rows:
    d = dict(np.load(r["f"]))
    ge = gf.grasp_err_from_poses(*(torch.tensor(d[k], dtype=torch.float64) for k in ["eef_pos", "eef_quat", "toy_truck_positions", "toy_truck_orientations"]),
                                 os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()
    ep, c, z, pc = np.linalg.norm(ge[:, :3], axis=1), r["closed"], r["z"], r["pc"]
    sw = np.diff(np.r_[0, c.astype(int), 0])
    for s, e in zip(np.where(sw == 1)[0], np.where(sw == -1)[0]):
        if (z[s : min(e + 1, len(z))] > LIFT).any() and s >= 30 and s + 10 < len(pc):
            prof_c.append(pc[s - 30 : s + 10])
        if e < len(z) and z[e] > LIFT:
            rel_pc.append(pc[e])
            if e >= 30 and e + 10 < len(pc):
                prof_r.append(pc[e - 30 : e + 10])
    L = np.where(z > LIFT)[0]
    pre = slice(0, L[0] if len(L) else len(c))
    by_d.append(np.c_[ep[pre], pc[pre], c[pre]])
P = np.array(prof_c)
print(f"- P(close) around lifting closes (n={len(P)}), median at t-30,-20,-10,-5,-1,0,+1,+3,+9: "
      + ", ".join(f"{np.median(P[:, i]):.2f}" for i in [0, 10, 20, 25, 29, 30, 31, 33, 39]))
print(f"  mean over t-10..t-1: {P[:, 20:30].mean():.3f} -> per-step chance of NOT closing for 10 steps = {np.prod(1 - P[:, 20:30].mean(0)):.2f}")
Rp = np.array(prof_r)
rel_pc = np.array(rel_pc)
print(f"- releases (n={len(rel_pc)}): P(close) at the open step median {np.median(rel_pc):.2f}, >0.5 (minority 'open' sampled) {np.mean(rel_pc > 0.5) * 100:.0f}%;"
      f" median at t-10,-5,-1,0,+1: " + ", ".join(f"{np.median(Rp[:, i]):.2f}" for i in [20, 25, 29, 30, 31]))
D = np.concatenate(by_d)
op = D[:, 2] == 0
print("- open-phase steps before lift: mean P(close) by |e_G,p| bin")
for lo, hi in [(0, .02), (.02, .03), (.03, .05), (.05, .10), (.10, .20), (.20, 9)]:
    m = op & (D[:, 0] >= lo) & (D[:, 0] < hi)
    print(f"  {lo * 100:.0f}-{hi * 100:.0f} cm: n={m.sum()}, mean P(close) {D[m, 1].mean():.3f}, P>0.5 {np.mean(D[m, 1] > .5) * 100:.1f}%")
