# Policy structure figure: gripper inside the GMM (baseline / gepl) vs separate gripper decision (evrz). CPU only.
#
#   python tools_c/plot_gripper_arch.py --out results/figs/gripper_arch.png
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_cjk):
    matplotlib.font_manager.fontManager.addfont(_cjk)
plt.rcParams.update({"font.family": matplotlib.font_manager.FontProperties(fname=_cjk).get_name() if os.path.exists(_cjk) else "sans-serif",
                     "axes.unicode_minus": False, "font.size": 10})
OBS, NET, ARM, GRIP, RULE, OUT, INK = "#e8eef7", "#dfe9d8", "#f3e3c3", "#f6d5d5", "#f9eccf", "#eeeeee", "#333333"


def box(ax, x, y, w, h, text, fc, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#666666", lw=1.1, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=INK, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, a, b, text=None, rad=0.0, ls="-", off=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#555555", ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", va="bottom", fontsize=8.5, color="#555555")


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/figs/gripper_arch_evrz.png")
A = ap.parse_args()

fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.4), gridspec_kw={"height_ratios": [1, 2.2]})
for ax in (a1, a2):
    ax.set_xlim(0, 13)
    ax.axis("off")

# (a) baseline / gepl: one network, gripper is the 7th GMM dimension
a1.set_ylim(0, 2.0)
a1.set_title("(a) baseline · gepl: 그리퍼가 팔과 함께 GMM의 7번째 차원 (한 번에 샘플링)", loc="left", fontsize=11.5)
box(a1, 0.2, 0.7, 2.4, 1.2, "관측\nbaseline 23D\ngepl 32D", OBS, True)
box(a1, 3.4, 0.7, 2.2, 1.2, "LSTM\n2×400, 길이 10", NET)
box(a1, 6.4, 0.7, 2.4, 1.2, "GMM 5 modes\n7D 분포", NET)
box(a1, 9.6, 0.7, 3.2, 1.2, "행동 7D (샘플링)\n[Δpos 3, Δrot 3, 그리퍼 1]\n그리퍼 = 부호(<0 닫힘)", OUT, True)
arrow(a1, (2.6, 1.3), (3.4, 1.3))
arrow(a1, (5.6, 1.3), (6.4, 1.3))
arrow(a1, (8.8, 1.3), (9.6, 1.3), "샘플링")
a1.text(0.2, 0.25, "손실: 7D 전체의 GMM 음의 로그우도. 같은 상태에서도 그리퍼가 주사위처럼 정해짐 → 반복 여닫기, 허공에서 닫기",
        fontsize=9, color="#555555")

# (b) evr / evrz: same observation, separate arm and gripper networks
a2.set_ylim(0, 4.6)
a2.set_title("(b) evrz: 입력은 같은 32D 관측, 출력 쪽에서 팔과 그리퍼를 분리 (열기 = 분류기, 닫기 = 규칙)", loc="left", fontsize=11.5)
box(a2, 0.2, 1.7, 2.4, 1.4, "관측 32D\n(gepl과 같음)\ngrasp_err_ee\nplace_err_ee 포함", OBS, True)
# arm branch
box(a2, 3.4, 3.4, 2.2, 0.95, "LSTM\n2×400, 길이 10", NET)
box(a2, 6.4, 3.4, 2.4, 0.95, "GMM 5 modes\n6D 분포", NET)
box(a2, 9.6, 3.4, 3.2, 0.95, "팔 6D (샘플링)\n[Δpos 3, Δrot 3]", ARM, True)
arrow(a2, (2.6, 2.75), (3.4, 3.85))
arrow(a2, (5.6, 3.87), (6.4, 3.87))
arrow(a2, (8.8, 3.87), (9.6, 3.87), "샘플링")
# gripper branch: open classifier
box(a2, 3.4, 1.75, 2.2, 1.0, "[x, Δx]\nΔx = 이전 step 대비 변화", OBS)
box(a2, 6.4, 1.6, 2.4, 1.3, "열기 분류기\nMLP 256×2\n\"지금 열어라\" 확률 p", GRIP)
box(a2, 9.6, 1.75, 3.2, 1.0, "p > τ 가 3 step 연속이면 엶\n(결정적, 샘플링 없음)\nτ: 학습 시범에서 보정", GRIP)
arrow(a2, (2.6, 2.3), (3.4, 2.25))
arrow(a2, (5.6, 2.25), (6.4, 2.25))
arrow(a2, (8.8, 2.25), (9.6, 2.25), "p")
# close rule
box(a2, 3.4, 0.25, 5.4, 0.95, "닫기 규칙\ngrasp_err_ee에서 e_xy = √(e_x²+e_y²) < 4 cm 그리고 e_z < 3 cm", RULE)
box(a2, 9.6, 0.25, 3.2, 0.95, "3 step 연속이면 닫음\n트럭을 들었다 놓은 뒤에는 끔", RULE)
arrow(a2, (2.6, 1.9), (3.4, 0.75))
arrow(a2, (8.8, 0.72), (9.6, 0.72))
# outputs
a2.text(12.95, 2.95, "행동 7D = [팔 6D, 그리퍼 ±1]", ha="right", va="center", fontsize=9.5, color=INK, fontweight="bold")
a2.text(0.2, -0.25, "두 네트워크는 층과 옵티마이저를 공유하지 않음 (손실: 팔 6D GMM 음의 로그우도 + 열기 사건 라벨 가중 BCE). "
        "팔은 그리퍼 결정을 입력으로 받지 않고, 관측 속 gripper_pos(실제 개폐)로만 그리퍼 상태를 앎.", fontsize=9, color="#555555")
fig.tight_layout()
fig.savefig(A.out, dpi=150, bbox_inches="tight")
print("->", A.out)
