"""
Generate the MACE-F Decision Flowchart — a standalone publication-quality figure.

Produces: figures/fig0_macef_decision_flowchart.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Script lives in scripts/; figure output is at the repository root.
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DPI = 200
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.unicode_minus": False,
})

# Tier color palette
TIER_C = {"T1": "#0072B2", "T2": "#009E73", "T3": "#F0E442",
           "T4": "#E69F00", "T5": "#D55E00"}
GREEN = "#2ecc71"
RED = "#e74c3c"
GRAY = "#7f8c8d"
WHITE = "#ffffff"
DARK = "#2c3e50"
BLUE_BG = "#eaf2f8"


def draw_box(ax, x, y, w, h, text, color=WHITE, edge=DARK, fontsize=9,
             bold=False, text_color=DARK, alpha=1.0):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.08", facecolor=color,
                         edgecolor=edge, linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=text_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color=DARK):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0,
                            "connectionstyle": "arc3,rad=0"}, zorder=2)


# ── Box-edge helper ──
def _e(cx, cy, w, h):
    """Return (left, right, top, bottom) of a centered box."""
    return (cx - w/2, cx + w/2, cy + h/2, cy - h/2)


def main():
    fig, ax = plt.subplots(figsize=(20, 28))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 28)
    ax.axis("off")
    ax.set_facecolor("#fafafa")

    # ===================================================================
    # TITLE
    # ===================================================================
    ax.text(10, 27.3, "MACE-F 制造业 AI 碳效决策流程图", ha="center", va="center",
            fontsize=20, fontweight="bold", color=DARK)
    ax.text(10, 26.5, "Manufacturing AI Carbon-Efficiency Decision Framework",
            ha="center", va="center", fontsize=12, color=GRAY, fontstyle="italic")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1 — DATA MODALITY  (3-way split)
    # ═══════════════════════════════════════════════════════════════════
    S1_CX, S1_CY, S1_W, S1_H = 10, 25.0, 7.5, 1.0
    draw_box(ax, S1_CX, S1_CY, S1_W, S1_H, "① 数据模态\nWhat type of data?",
             WHITE, DARK, 11, True)
    _, _, _, s1b = _e(S1_CX, S1_CY, S1_W, S1_H)
    ax.text(0.5, S1_CY, "START", fontsize=9, fontweight="bold", color=RED,
            ha="center", va="center")

    # Three data-modality branches
    BR_W, BR_H = 3.2, 0.9
    BR_X = [3.0, 10.0, 17.0]          # Tabular / Time-Series / Image
    BR_LABELS = ["表格/传感器\nTabular", "时间序列\nTime-Series", "图像\nImage"]
    br_cy = s1b - 0.7 - BR_H/2          # branch box centres

    for bx, label in zip(BR_X, BR_LABELS):
        draw_box(ax, bx, br_cy, BR_W, BR_H, label, "#d5f5e3", DARK, 9, True)
        _, _, brt, _ = _e(bx, br_cy, BR_W, BR_H)
        draw_arrow(ax, S1_CX, s1b, bx, brt)

    # Tier-outcome boxes below each branch
    TIER_W, TIER_H = 2.8, 0.85
    tier_cy = br_cy - BR_H/2 - 0.6 - TIER_H/2
    tier_specs = [
        (BR_X[0], "T1–T2  轻量",            TIER_C["T1"]),
        (BR_X[1], "T1 或 T3 (LSTM)",         TIER_C["T3"]),
        (BR_X[2], "T2–T4  视觉",             TIER_C["T4"]),
    ]
    tier_bottoms, tier_centers = [], []
    for tx, tlabel, tc in tier_specs:
        draw_box(ax, tx, tier_cy, TIER_W, TIER_H, tlabel, tc, WHITE, 9, True)
        _, _, _, tb = _e(tx, tier_cy, TIER_W, TIER_H)
        tier_bottoms.append(tb)
        tier_centers.append(tx)
        # arrow from branch bottom → tier top
        _, _, _, brb = _e(tx, br_cy, BR_W, BR_H)
        _, _, ttt, _ = _e(tx, tier_cy, TIER_W, TIER_H)
        draw_arrow(ax, tx, brb, tx, ttt)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2 — ACCURACY REQUIREMENT  (2-way split)
    # ═══════════════════════════════════════════════════════════════════
    S2_CX, S2_W, S2_H = 10, 7.5, 1.0
    s2_cy = tier_cy - TIER_H/2 - 1.5 - S2_H/2
    draw_box(ax, S2_CX, s2_cy, S2_W, S2_H, "② 精度要求\nAccuracy Requirement?",
             WHITE, DARK, 11, True)
    _, _, s2t, s2b = _e(S2_CX, s2_cy, S2_W, S2_H)

    # Connectors: all tier outcomes → Step 2 top
    for tx, tb in zip(tier_centers, tier_bottoms):
        draw_arrow(ax, tx, tb, S2_CX, s2t, GRAY)

    # Two accuracy branches
    A2_W, A2_H = 3.5, 0.9
    a2_cy = s2b - 0.7 - A2_H/2
    # Left: <95% acceptable
    draw_box(ax, 3, a2_cy, A2_W, A2_H, "< 95% 可接受\nAcceptable", "#d5f5e3", DARK, 9, True)
    _, _, a2lt, _ = _e(3, a2_cy, A2_W, A2_H)
    draw_arrow(ax, S2_CX - S2_W*0.3, s2b, 3, a2lt)
    # Right: >99% required
    draw_box(ax, 17, a2_cy, A2_W, A2_H, "> 99% 必需\nRequired", "#fadbd8", DARK, 9, True)
    _, _, a2rt, _ = _e(17, a2_cy, A2_W, A2_H)
    draw_arrow(ax, S2_CX + S2_W*0.3, s2b, 17, a2rt)

    # Accuracy outcome boxes
    AO_W, AO_H = 4.0, 0.85
    ao_cy = a2_cy - A2_H/2 - 0.6 - AO_H/2
    ao_specs = [
        (3,  "T2 最优碳效\nRandom Forest / MobileNet-v2", TIER_C["T2"], TIER_C["T2"]),
        (17, "T3–T4\nResNet-50 / ResNet-152",             TIER_C["T4"], TIER_C["T4"]),
    ]
    ao_bottoms, ao_centers = [], []
    for ax_pos, alabel, bg, tc in ao_specs:
        draw_box(ax, ax_pos, ao_cy, AO_W, AO_H, alabel, bg, WHITE, 8, True, text_color=tc)
        _, _, _, ab = _e(ax_pos, ao_cy, AO_W, AO_H)
        ao_bottoms.append(ab)
        ao_centers.append(ax_pos)
        # arrow from accuracy-branch bottom
        _, _, _, brb = _e(ax_pos, a2_cy, A2_W, A2_H)
        _, _, aot, _ = _e(ax_pos, ao_cy, AO_W, AO_H)
        draw_arrow(ax, ax_pos, brb, ax_pos, aot)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3 — INFERENCE FREQUENCY  (3-way split)
    # ═══════════════════════════════════════════════════════════════════
    S3_CX, S3_W, S3_H = 10, 7.5, 1.0
    s3_cy = ao_cy - AO_H/2 - 1.5 - S3_H/2
    draw_box(ax, S3_CX, s3_cy, S3_W, S3_H, "③ 推理频率\nHow Many Inferences Per Day?",
             WHITE, DARK, 11, True)
    _, _, s3t, s3b = _e(S3_CX, s3_cy, S3_W, S3_H)

    for ax_pos, ab in zip(ao_centers, ao_bottoms):
        draw_arrow(ax, ax_pos, ab, S3_CX, s3t, GRAY)

    # Three frequency branches
    FQ_W, FQ_H = 3.0, 0.9
    fq_cy = s3b - 0.7 - FQ_H/2
    fq_x = [2.5, 10, 17.5]                                           # Low / Medium / High
    fq_labels = ["< 1,000/天\nLow", "1K–10K/天\nMedium", "> 10K/天\nHigh"]
    fq_colors = ["#d5f5e3", "#fef9e7", "#fadbd8"]
    for fx, fl, fc in zip(fq_x, fq_labels, fq_colors):
        draw_box(ax, fx, fq_cy, FQ_W, FQ_H, fl, fc, DARK, 9, True)
        _, _, fqt, _ = _e(fx, fq_cy, FQ_W, FQ_H)
        draw_arrow(ax, fx, s3b, fx, fqt)

    # Frequency outcome boxes
    FO_W, FO_H = 3.0, 0.9
    fo_cy = fq_cy - FQ_H/2 - 0.6 - FO_H/2
    fo_specs = [
        (2.5,  "任意层级\n碳排非瓶颈",           GREEN,    GREEN),
        (10,   "优先 T1–T3\nT5 需谨慎",           "#f39c12", WHITE),
        (17.5, "禁止 T5\nNo T5 Allowed",          RED,      WHITE),
    ]
    fo_bottoms, fo_centers = [], []
    for fx, fl, fc, tc in fo_specs:
        draw_box(ax, fx, fo_cy, FO_W, FO_H, fl, fc, WHITE, 9, True, text_color=tc)
        _, _, _, fb = _e(fx, fo_cy, FO_W, FO_H)
        fo_bottoms.append(fb)
        fo_centers.append(fx)
        _, _, _, fqb = _e(fx, fq_cy, FQ_W, FQ_H)
        _, _, fot, _ = _e(fx, fo_cy, FO_W, FO_H)
        draw_arrow(ax, fx, fqb, fx, fot)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4 — DEPLOYMENT LOCATION  (2-way split)
    # ═══════════════════════════════════════════════════════════════════
    S4_CX, S4_W, S4_H = 10, 7.5, 1.0
    s4_cy = fo_cy - FO_H/2 - 1.5 - S4_H/2
    draw_box(ax, S4_CX, s4_cy, S4_W, S4_H, "④ 部署位置\nWhere Is It Deployed?",
             WHITE, DARK, 11, True)
    _, _, s4t, s4b = _e(S4_CX, s4_cy, S4_W, S4_H)

    for fx, fb in zip(fo_centers, fo_bottoms):
        draw_arrow(ax, fx, fb, S4_CX, s4t, GRAY)

    # Two location branches
    LC_W, LC_H = 4.5, 0.9
    lc_cy = s4b - 0.7 - LC_H/2
    draw_box(ax, 3, lc_cy, LC_W, LC_H, "低碳电网  < 100 gCO₂/kWh\n(北欧、法国)", "#d5f5e3", DARK, 9, True)
    _, _, lclt, _ = _e(3, lc_cy, LC_W, LC_H)
    draw_arrow(ax, S4_CX - S4_W*0.3, s4b, 3, lclt)
    draw_box(ax, 17, lc_cy, LC_W, LC_H, "高碳电网  > 400 gCO₂/kWh\n(中国、印度)", "#fadbd8", DARK, 9, True)
    _, _, lcrt, _ = _e(17, lc_cy, LC_W, LC_H)
    draw_arrow(ax, S4_CX + S4_W*0.3, s4b, 17, lcrt)

    # Location outcome boxes
    LO_W, LO_H = 4.5, 0.85
    lo_cy = lc_cy - LC_H/2 - 0.6 - LO_H/2
    lo_specs = [
        (3,  "碳排非约束\n精度优先",           GREEN, GREEN),
        (17, "必须搬迁 / 云迁移\nMust Migrate", RED,   WHITE),
    ]
    lo_bottoms, lo_centers = [], []
    for lx, ll, lc, tc in lo_specs:
        draw_box(ax, lx, lo_cy, LO_W, LO_H, ll, lc, WHITE, 9, True, text_color=tc)
        _, _, _, lb = _e(lx, lo_cy, LO_W, LO_H)
        lo_bottoms.append(lb)
        lo_centers.append(lx)
        _, _, _, lcb = _e(lx, lc_cy, LC_W, LC_H)
        _, _, lot, _ = _e(lx, lo_cy, LO_W, LO_H)
        draw_arrow(ax, lx, lcb, lx, lot)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5 — OUTPUT
    # ═══════════════════════════════════════════════════════════════════
    S5_CX, S5_W, S5_H = 10, 8.5, 1.1
    s5_cy = lo_cy - LO_H/2 - 1.5 - S5_H/2
    draw_box(ax, S5_CX, s5_cy, S5_W, S5_H,
             "⑤ 输出\n预期年度碳排放 + 碳排等级 + 减排建议",
             WHITE, DARK, 12, True)
    _, _, s5t, _ = _e(S5_CX, s5_cy, S5_W, S5_H)

    for lx, lb in zip(lo_centers, lo_bottoms):
        draw_arrow(ax, lx, lb, S5_CX, s5t, GRAY)

    # ═══════════════════════════════════════════════════════════════════
    # LEFT SIDEBAR — Tier Carbon Reference  (compact)
    # ═══════════════════════════════════════════════════════════════════
    LSIDE_X, LSIDE_W = 1.0, 1.3
    ls_y = S1_CY + S1_H/2 + 0.8    # just above step 1
    ax.text(LSIDE_X, ls_y, "层级碳排\nTIER REF", ha="center", va="bottom",
            fontsize=7, fontweight="bold", color=DARK, zorder=4)

    tiers_ref = [
        ("T1", "0.02g",          TIER_C["T1"]),
        ("T2", "0.01–3.3g",      TIER_C["T2"]),
        ("T3", "0.7–7.9g",       TIER_C["T3"]),
        ("T4", "18–33g",         TIER_C["T4"]),
        ("T5", "67–2635 kg/yr",  TIER_C["T5"]),
    ]
    ls_item_h = 0.55
    ls_gap = 0.25
    ls_ty = ls_y - 1.0
    for i, (tier, val, c) in enumerate(tiers_ref):
        iy = ls_ty - i * (ls_item_h + ls_gap)
        draw_box(ax, LSIDE_X, iy, LSIDE_W, ls_item_h,
                 f"{tier}  {val}", c, WHITE, 6, True, WHITE)

    # ═══════════════════════════════════════════════════════════════════
    # RIGHT SIDEBAR — Carbon Reduction Levers  (compact)
    # ═══════════════════════════════════════════════════════════════════
    RSIDE_X, RSIDE_W = 18.8, 1.3
    rs_y = S1_CY + S1_H/2 + 0.8
    ax.text(RSIDE_X, rs_y, "减排杠杆\nLEVERS", ha="center", va="bottom",
            fontsize=7, fontweight="bold", color=DARK, zorder=4)

    levers = [
        ("P0 模型降级",   "–99.9%",   RED),
        ("P1 选址优化",   "10–78×",   "#e67e22"),
        ("P2 云迁移",     "~50×",     "#f39c12"),
        ("P3 迁移学习",   "–60.6%",   GREEN),
        ("P4 超参数调优", "–16%",     GRAY),
        ("P5 4Ms优化",    "100–1000×", TIER_C["T1"]),
    ]
    for i, (label, val, c) in enumerate(levers):
        iy = rs_y - 1.0 - i * (ls_item_h + ls_gap)
        draw_box(ax, RSIDE_X, iy, RSIDE_W, ls_item_h,
                 f"{label}  {val}", c, WHITE, 6, True, WHITE)

    # ═══════════════════════════════════════════════════════════════════
    # BOTTOM — Legend  (simplified)
    # ═══════════════════════════════════════════════════════════════════
    leg_y = 1.2
    leg_items = [
        (3.0, "推荐 / 安全",    "#d5f5e3", DARK),
        (8.5, "需评估",          "#fef9e7", DARK),
        (14.0, "高风险 / 禁止",  "#fadbd8", DARK),
    ]
    for lx, ltext, lbg, lc in leg_items:
        draw_box(ax, lx, leg_y, 3.5, 0.55, ltext, lbg, lc, 7.5)

    fig.tight_layout(pad=0.5)
    fig.savefig(FIGURES_DIR / "fig0_macef_decision_flowchart.png", dpi=DPI,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  OK  fig0_macef_decision_flowchart.png")


# ═══════════════════════════════════════════════════════════════════════
# COMPACT REPORT VERSION — landscape, 4-column horizontal layout
# ═══════════════════════════════════════════════════════════════════════

def main_compact():
    """Generate a compact landscape flowchart suitable for report inclusion."""
    fig, ax = plt.subplots(figsize=(18, 7.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_facecolor("#fafafa")

    # ── Title ──
    ax.text(9, 7.15, "MACE-F 制造业 AI 碳效决策流程图", ha="center", va="center",
            fontsize=16, fontweight="bold", color=DARK)
    ax.text(9, 6.70, "Manufacturing AI Carbon-Efficiency Decision Framework  |  Compact",
            ha="center", va="center", fontsize=9, color=GRAY, fontstyle="italic")

    # ── Column layout ──
    # 4 decision columns, evenly spaced
    COL_X = [2.8, 7.3, 11.8, 16.0]   # column centres
    COL_W = 3.2                       # decision-box width
    DEC_Y, DEC_H = 6.05, 0.70         # decision-box y-centre & height

    # Decision box labels
    dec_labels = [
        "①  数据模态\n     Data Modality",
        "②  精度要求\n     Accuracy",
        "③  推理频率\n     Inference Freq.",
        "④  区位与输出\n     Location & Output",
    ]
    dec_bottoms = []
    for cx, label in zip(COL_X, dec_labels):
        draw_box(ax, cx, DEC_Y, COL_W, DEC_H, label, WHITE, DARK, 9, True)
        _, _, _, db = _e(cx, DEC_Y, COL_W, DEC_H)
        dec_bottoms.append(db)

    # Horizontal arrows between decision boxes
    for i in range(3):
        _, r, _, _ = _e(COL_X[i], DEC_Y, COL_W, DEC_H)
        l, _, _, _ = _e(COL_X[i+1], DEC_Y, COL_W, DEC_H)
        draw_arrow(ax, r, DEC_Y, l, DEC_Y, GRAY)

    # ── Branch boxes below each decision ──
    # Each branch box shows: CONDITION + TIER RECOMMENDATION (combined)
    BR_W, BR_H = 2.9, 0.52
    GAP_DEC_TO_BR = 0.30          # gap from decision bottom to first branch top
    GAP_BR = 0.10                  # gap between branch boxes
    BR_START_Y = DEC_Y - DEC_H/2 - GAP_DEC_TO_BR - BR_H/2  # centre of first branch

    # Column branch definitions: (condition_line, tier_line, bg_color, text_color)
    col_branches = [
        # Col 1 — Data Modality
        [("表格/传感器",   "T1–T2  轻量模型",    "#d5f5e3", DARK),
         ("时间序列",      "T1 或 T3  LSTM",     "#d5f5e3", DARK),
         ("图像",          "T2–T4  视觉模型",    "#d5f5e3", DARK)],
        # Col 2 — Accuracy
        [("< 95%  可接受",  "T2  RF / MobileNet-v2", "#d5f5e3", TIER_C["T2"]),
         ("> 99%  必需",    "T3–T4  ResNet",         "#fadbd8", TIER_C["T4"])],
        # Col 3 — Inference Frequency
        [("< 1K /天",      "任意层级  OK",           "#d5f5e3", GREEN),
         ("1K – 10K /天",  "优先 T1–T3",             "#fef9e7", DARK),
         ("> 10K /天",     "禁止 T5",                "#fadbd8", RED)],
        # Col 4 — Location
        [("低碳电网",       "精度优先",              "#d5f5e3", GREEN),
         ("高碳电网",       "必须搬迁 / 云迁移",     "#fadbd8", RED)],
    ]

    all_branch_boxes = []   # (cx, cy, w, h) for each branch box
    for ci, branches in enumerate(col_branches):
        for bi, (cond, tier, bg, tc) in enumerate(branches):
            by = BR_START_Y - bi * (BR_H + GAP_BR)
            label = f"{cond}\n{tier}"
            draw_box(ax, COL_X[ci], by, BR_W, BR_H, label, bg, DARK, 6.5, True, text_color=tc)
            all_branch_boxes.append((COL_X[ci], by, BR_W, BR_H))
            # Vertical arrow: decision bottom → branch top
            _, _, brt, _ = _e(COL_X[ci], by, BR_W, BR_H)
            if bi == 0:  # only first branch gets arrow from decision
                draw_arrow(ax, COL_X[ci], dec_bottoms[ci], COL_X[ci], brt)

    # ── Output box at bottom ──
    # Find lowest branch bottom
    lowest_br_bottom = min(
        cy - BR_H/2 for _, cy, _, _ in all_branch_boxes
    )
    OUT_Y = lowest_br_bottom - 0.55 - 0.55   # gap + half output height
    OUT_W, OUT_H = 12, 0.70
    draw_box(ax, 9, OUT_Y, OUT_W, OUT_H,
             "⑤  综合评估输出：年度碳排放  |  碳排风险等级  |  五级减排方案",
             WHITE, DARK, 10, True)

    # Converging arrows: each branch bottom → output top
    _, _, out_top, _ = _e(9, OUT_Y, OUT_W, OUT_H)
    for cx, cy, bw, _ in all_branch_boxes:
        _, _, _, br_bot = _e(cx, cy, bw, BR_H)
        draw_arrow(ax, cx, br_bot, cx, out_top, GRAY)

    # ── Legend ──
    LEG_Y = 0.35
    leg_items = [
        (3.0, "推荐 / 安全",    "#d5f5e3"),
        (8.0, "需评估",          "#fef9e7"),
        (13.0, "高风险 / 禁止",  "#fadbd8"),
    ]
    for lx, ltext, lbg in leg_items:
        draw_box(ax, lx, LEG_Y, 2.6, 0.38, ltext, lbg, DARK, 6.5)

    fig.tight_layout(pad=0.3)
    out_path = FIGURES_DIR / "fig0_macef_decision_flowchart_compact.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  OK  {out_path.name}")


if __name__ == "__main__":
    main()
    main_compact()
