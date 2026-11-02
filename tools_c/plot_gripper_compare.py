# Gripper raster side by side, one column per model, seeds pooled (CPU only). Same markers as plot_gripper_events.py.
#
#   CUDA_VISIBLE_DEVICES= python tools_c/plot_gripper_compare.py --group baseline outputs/rollouts/base_s101 outputs/rollouts/base_s111 \
#       --group evrz outputs/rollouts/evrz_s101 outputs/rollouts/evrz_s111 --out results/figs/base_vs_evrz_gripper_raster.png
# Also writes the "where" comparison next to it as *_where.png (rows = close e_xy / close e_z / open EE-box xy, one column per model).
import argparse
import sys

sys.argv, _argv = sys.argv[:1], sys.argv
exec(open("tools_c/plot_gripper_events.py").read().split("\nap = argparse.ArgumentParser()\n")[0])  # load, mark_*, LEGEND, styles
sys.argv = _argv


def panel(ax, name, eps):
    rows = sorted(eps, key=lambda e: (ORDER.index(e["cls"]), e["run"], e["k"]))
    for i, e in enumerate(rows):
        for s, t in e["segs"]:
            ax.plot([s, t], [i, i], color=BAND, lw=1.6, solid_capstyle="butt", zorder=2)
        for t, lifted in e["closes"]:  # thinner edge than mark_close so the fill stays readable at this row height
            ax.scatter(t, i, s=20, facecolor=GREEN if lifted else BLACK, edgecolor=RED, linewidth=0.8, zorder=3)
        for t in e["opens"]:
            ax.scatter(t, i, s=20, facecolor="white", edgecolor=BLUE, linewidth=0.8, zorder=3)
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
    nc = sum(len(e["closes"]) for e in eps)
    nl = sum(l for e in eps for _, l in e["closes"])
    no = sum(len(e["opens"]) for e in eps)
    sw = np.array([e["sw"] for e in eps])
    ax.set_title(f"{name} ({len(eps)}회, seed 101+111)\n닫음 {nc}회 (들림 {nl}, 안 들림 {nc - nl}) · 엶 {no}회\n"
                 f"에피소드당 바꾼 횟수 평균 {sw.mean():.1f}, 중앙값 {np.median(sw):.0f}, 최대 {sw.max()}", loc="left")
    return nc, nl, no, sw


def rel(e):
    """Relative coordinates at every step: grasp error in EE axes split into e_xy / e_z, and the EE-box horizontal distance."""
    d = e["d"]
    T = lambda k: torch.tensor(d[k], dtype=torch.float64)  # noqa: E731
    ge = gf.grasp_err_from_poses(T("eef_pos"), T("eef_quat"), T("toy_truck_positions"), T("toy_truck_orientations"),
                                 os.path.join(CFG, "grasp_offset.json"), True, "ee").numpy()
    # release: the truck falls along base z, so the horizontal (base xy) EE-box distance, as in compute_place_offset.py
    return dict(e, exy=np.linalg.norm(ge[:, :2], axis=1) * 100, ez=ge[:, 2] * 100,
                ee_box=np.linalg.norm(d["eef_pos"][:, :2] - d["box_positions"][:, :2], axis=1) * 100)


def demo_release(fn):
    """EE-box horizontal distance (cm) at the human release (first open after the truck is lifted)."""
    import h5py

    D = []
    with h5py.File(fn, "r") as f:
        for k in f["data"]:
            o, g = f["data"][k]["obs"], f["data"][k]["actions"][:, 6]
            lifted = np.where(o["toy_truck_positions"][:, 2] - o["toy_truck_positions"][0, 2] > LIFT)[0]
            op = np.where(g[lifted[0] :] > 0)[0] if len(lifted) else []
            if len(op):
                t = lifted[0] + op[0]
                D.append(np.linalg.norm(o["eef_pos"][t, :2] - o["box_positions"][t, :2]) * 100)
    return np.array(D)


def where(axs, name, eps, demo):
    """Close: lateral error e_xy and depth e_z of the grasp target in EE axes (the close rule's axes). Open: EE-box xy."""
    a1, a2, a3 = axs
    nc = nl = no = n_rule = 0
    for e in eps:
        for t, lifted in e["closes"]:
            for a, y in ((a1, max(e["exy"][t], 0.3)), (a2, e["ez"][t])):
                a.scatter(t, y, s=26, facecolor=GREEN if lifted else BLACK, edgecolor=RED, linewidth=1.0, zorder=3)
            nc, nl, n_rule = nc + 1, nl + lifted, n_rule + (e["exy"][t] < 4 and e["ez"][t] < 3)
        for t in e["opens"]:
            if e["z"][t] > LIFT:  # release while the truck is held
                mark_open(a3, t, e["ee_box"][t], s=26)
                no += 1
    a1.set_yscale("log")
    a1.set_yticks([0.5, 1, 2, 4, 10, 20, 50, 100], ["0.5", "1", "2", "4", "10", "20", "50", "100"])
    a1.axhline(4, color="#777777", lw=1, ls="--")
    a1.text(8, 4.3, "4 cm", ha="left", va="bottom", color="#555555")
    a1.set_ylabel("옆 방향 오차 e_xy (cm, log)")
    a1.set_title(f"{name} · 닫은 순간: 옆 방향 오차 e_xy = √(e_x²+e_y²)\n"
                 f"닫음 {nc}회 (들림 {nl}, 안 들림 {nc - nl}) · 규칙 영역(e_xy<4, e_z<3) 안 {n_rule}회", loc="left")
    a2.set_yscale("symlog", linthresh=5, linscale=1.5)
    a2.set_yticks([-50, -10, -5, -3, 0, 3, 5, 10, 50], ["−50", "−10", "−5", "−3", "0", "3", "5", "10", "50"])
    a2.axhline(3, color="#777777", lw=1, ls="--")
    a2.axhline(0, color="#bbbbbb", lw=0.8)
    a2.text(8, 3.3, "3 cm", ha="left", va="bottom", color="#555555")
    a2.set_ylabel("접근 깊이 오차 e_z (cm, ±5 안은 선형)")
    a2.set_title(f"{name} · 닫은 순간: 접근 깊이 오차 e_z (+ = 목표보다 덜 내려감)", loc="left")
    a3.axhline(np.percentile(demo, 90), color="#777777", lw=1, ls="--")
    a3.text(8, np.percentile(demo, 90) + 0.5, f"사람 데모 놓기 90%: {np.percentile(demo, 90):.1f} cm", ha="left", va="bottom", color="#555555")
    a3.set_ylabel("EE–박스 수평 거리 (cm)")
    no_in = sum(e["z"][t] > LIFT and e["ee_box"][t] < np.percentile(demo, 90) for e in eps for t in e["opens"])
    a3.set_title(f"{name} · 연 순간: EE–박스 수평 거리\n트럭을 든 채 엶 {no}회 (데모 90% 선 안 {no_in}, 밖 {no - no_in})", loc="left")
    for a, lab in ((a1, "닫은 시점 (step)"), (a2, "닫은 시점 (step)"), (a3, "연 시점 (step, 트럭을 든 상태에서)")):
        a.set_xlim(0, 800)
        a.set_xlabel(lab)
        a.grid(axis="y", color=GRID, lw=0.8)

ap = argparse.ArgumentParser()
ap.add_argument("--group", nargs="+", action="append", required=True, help="NAME RUN [RUN ...]; repeat once per column")
ap.add_argument("--out", default="results/figs/base_vs_evrz_gripper_raster.png")
ap.add_argument("--demo", default="datasets/team10_lee_s3.hdf5", help="human demos for the release reference line")
A = ap.parse_args()

G = len(A.group)
fig, axs = plt.subplots(1, G, figsize=(7.5 * G, 15), sharex=True)
fw, aw = plt.subplots(3, G, figsize=(7 * G, 14), sharey="row")
demo = demo_release(A.demo)
for col, (ax, (name, *runs)) in enumerate(zip(axs, A.group)):
    eps = [dict(e, run=r) for r in runs for e in load(r)]
    where(aw[:, col], name, [rel(e) for e in eps], demo)
    nc, nl, no, sw = panel(ax, name, eps)
    print(f"{name}: closes {nc} (lifted {nl}), opens {no}, switches/ep mean {sw.mean():.2f} median {np.median(sw):.0f} max {sw.max()}, "
          f"eps with >2 switches {np.mean(sw > 2):.2f}")
fig.legend(handles=LEGEND, loc="upper right", bbox_to_anchor=(0.99, 0.998), ncol=3, frameon=False)
fig.suptitle("에피소드별 그리퍼 닫음·엶 시점 (한 줄 = 한 에피소드, 분홍 막대 = 닫혀 있던 구간)", x=0.01, y=0.998, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(A.out, dpi=130)
print("->", A.out)
fw.legend(handles=LEGEND, loc="lower center", ncol=3, frameon=False)
fw.suptitle("그리퍼를 바꾼 순간의 상대 위치 (seed 101+111, 각 200회; 잡기 오차는 EE 좌표축 기준)", x=0.01, ha="left")
fw.tight_layout(rect=(0, 0.03, 1, 0.975))
fw.savefig(A.out.replace("_raster.png", "_where.png"), dpi=130)
print("->", A.out.replace("_raster.png", "_where.png"))
