# Stage B judgement (pre-registered in results/event_clf/plan_B.md): bc_event runs vs gepl on recorded rollouts. CPU only.
#
#   python tools_c/judge_event.py --baseline outputs/rollouts/gepl_s101 outputs/rollouts/gepl_s111 \
#       --exp outputs/rollouts/evr_s101 outputs/rollouts/evr_s111 [--release]
import argparse
import os
import sys

import numpy as np

sys.argv, _argv = sys.argv[:1], sys.argv  # gripper_diag parses its own CLI at the bottom; load only its functions
exec(open("tools_c/gripper_diag.py").read().split("ap = argparse.ArgumentParser()")[0])
sys.argv = _argv


def metrics(run):
    E = load_rollouts(run)
    first_rel = [e["rel"][0]["xy_TB"] for e in E if e["rel"]]
    return {
        "success": np.mean([e["success"] for e in E]),
        "lifted": np.mean([any(c["lift"] for c in e["closes"]) for e in E]),
        "miss_close": np.mean([sum(not c["lift"] for c in e["closes"]) for e in E]),  # non-lifting closes per episode
        "multi_switch": np.mean([e["n_switch"] > 2 for e in E]),
        "rel_in_box": np.mean(np.array(first_rel) < 0.10) if first_rel else np.nan,
        "release_out": np.mean([e["release_out"] for e in E]),
        "n": len(E),
    }


ap = argparse.ArgumentParser()
ap.add_argument("--baseline", nargs="+", required=True)
ap.add_argument("--exp", nargs="+", required=True)
ap.add_argument("--release", action="store_true", help="secondary comparison (ready_inbox vs ready): release criteria")
A = ap.parse_args()
B = {os.path.basename(os.path.normpath(p)): metrics(p) for p in A.baseline}
E = {os.path.basename(os.path.normpath(p)): metrics(p) for p in A.exp}
cols = ["success", "lifted", "miss_close", "multi_switch", "rel_in_box", "release_out"]
print("| run | n | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 2))
for name, m in {**B, **E}.items():
    print(f"| {name} | {m['n']} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
mean = lambda S, k: float(np.mean([s[k] for s in S.values()]))  # noqa: E731
beats = min(s["success"] for s in E.values()) > max(s["success"] for s in B.values())
gain = mean(E, "success") - mean(B, "success")
if A.release:
    crit = [
        ("R1. first release with truck-box xy < 10 cm: ratio up", mean(E, "rel_in_box") > mean(B, "rel_in_box")),
        ("R2. released-outside-box ratio down", mean(E, "release_out") < mean(B, "release_out")),
        ("R3. success not lower", mean(E, "success") >= mean(B, "success")),
    ]
else:
    crit = [
        (f"B1. success +5%p ({gain:+.3f}) and both exp runs > both baseline runs", gain >= 0.05 and beats),
        ("B2. non-lifting closes per episode down", mean(E, "miss_close") < mean(B, "miss_close")),
        ("B2'. episodes with > 2 gripper switches down", mean(E, "multi_switch") < mean(B, "multi_switch")),
        ("B3. episodes that lift the truck: ratio not lower", mean(E, "lifted") >= mean(B, "lifted")),
    ]
print(f"\nbaseline success {mean(B, 'success'):.3f} | exp success {mean(E, 'success'):.3f}")
for txt, ok in crit:
    print(f"  [{'PASS' if ok else 'FAIL'}] {txt}")
