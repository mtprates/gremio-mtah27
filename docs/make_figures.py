"""Figures for docs/TECHNICAL.md.

    .venv/Scripts/python docs/make_figures.py      # -> docs/figures/*.png

Reads only committed, non-match files (validation/readiness_sets.csv and readiness_games.csv from
model/readiness.py; the model's own constants in model/hmm.py) and draws schematics. Nothing here opens the BAR-ATH data.
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "model"))
OUT = os.path.join(ROOT, "docs", "figures")

# Reference palette (dataviz skill, light mode): categorical slots in fixed order, inks, surfaces.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BOX, BOX_EDGE = "#f0efec", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote docs/figures/" + name)


def box(ax, x, y, w, h, text, fill=BOX, edge=BOX_EDGE, color=INK, size=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, weight=weight)


def arrow(ax, a, b, color=MUTED, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.3,
                                 connectionstyle="arc3,rad={}".format(rad), shrinkA=2, shrinkB=2))


# -- 1. pipeline ------------------------------------------------------------------------------------
def pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.grid(False)
    W, H = 1.75, 0.78
    # inputs
    box(ax, 1.0, 3.7, W, H, "Tactical video\n+ homography")
    box(ax, 1.0, 2.4, W, H, "Broadcast video\n+ homography")
    box(ax, 1.0, 1.0, W, H, "Smart Tagging\n(kick-offs only)", size=8.5)
    # stage boxes
    box(ax, 3.2, 3.05, W, H, "YOLO detection\nballs + people")
    box(ax, 3.2, 1.75, W, H, "Alignment, cuts,\nclose-ups, replays", size=8.5)
    box(ax, 5.4, 3.7, W, H, "Players & teams\ntracker + kit mixture", size=8.5)
    box(ax, 5.4, 2.4, W, H, "Ball picks\nViterbi per camera", size=8.5)
    box(ax, 7.6, 2.4, W, 1.0, "3D ball + fusion\narcs, height fix,\n2-camera Kalman", size=8.5)
    box(ax, 9.75, 3.7, W, H, "Events\ntouch, kick, control,\ncarry, dead-ball signals", size=8)
    box(ax, 9.75, 1.9, W, 1.0, "Model C\nteam HMM\nH / A / D", fill=BLUE, edge=BLUE, color="#ffffff",
        weight="bold")
    box(ax, 9.75, 0.45, W, 0.6, "submission CSV", size=8.5)
    arrow(ax, (1.9, 3.6), (2.3, 3.15))
    arrow(ax, (1.9, 2.5), (2.3, 2.95))
    arrow(ax, (1.9, 2.3), (2.3, 1.85))
    arrow(ax, (4.1, 3.15), (4.5, 3.6))
    arrow(ax, (4.1, 2.95), (4.5, 2.5))
    arrow(ax, (4.1, 1.85), (4.5, 2.3))
    arrow(ax, (6.3, 2.4), (6.7, 2.4))
    arrow(ax, (6.3, 3.7), (8.85, 3.7))
    arrow(ax, (8.5, 2.8), (8.95, 3.3))
    arrow(ax, (9.75, 3.3), (9.75, 2.42))
    arrow(ax, (1.9, 1.0), (8.85, 1.55), rad=0.0)
    arrow(ax, (9.75, 1.38), (9.75, 0.77))
    ax.text(5.5, 4.45, "Everything right of the inputs is our own: the provider's tracking only aligns the two "
            "videos; no audio is used", ha="center", fontsize=8.5, color=INK2)
    save(fig, "pipeline.png")


# -- 2. line of sight: why the ball's ground point needs a height ------------------------------------
def line_of_sight():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cam_t, cam_b = np.array([-8.0, 22.0]), np.array([95.0, 14.0])     # synthetic camera positions (x, z)
    t = np.linspace(0, 1, 60)
    x = 20 + 45 * t
    z = 4 * 9.0 * t * (1 - t)                                         # a 9 m high, 45 m long arc
    ax.plot(x, z, color=INK2, linewidth=1.6, label="the ball's true flight")
    ball = np.array([x[24], z[24]])
    for cam, color, name in ((cam_t, BLUE, "tactical camera"), (cam_b, ORANGE, "broadcast camera")):
        d = ball - cam
        g = cam + d * (cam[1] / -d[1])                                # where the sight line meets the grass
        ax.plot([cam[0], g[0]], [cam[1], 0], color=color, linewidth=1.3, linestyle=(0, (4, 3)))
        ax.plot(*g, marker="v", markersize=9, color=color, markeredgecolor=SURFACE, markeredgewidth=2,
                linestyle="none", label="{}: ground point".format(name))
        ax.plot(*cam, marker="s", markersize=9, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(name, cam, xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK2)
    ax.plot(*ball, marker="o", markersize=10, color=AQUA, markeredgecolor=SURFACE, markeredgewidth=2,
            linestyle="none", label="triangulated ball (height recovered)")
    ax.axhline(0, color=AXIS, linewidth=1.2)
    ax.set_xlim(-14, 101)
    ax.set_ylim(-1.5, 26)
    ax.set_xlabel("along the pitch (m), synthetic")
    ax.set_ylabel("height (m)")
    ax.legend(loc="upper center", fontsize=8, ncol=2, bbox_to_anchor=(0.5, -0.22))
    ax.set_title("A ball in the air projects to two different ground points; their disagreement gives its "
                 "height", fontsize=10, loc="left", color=INK)
    save(fig, "line_of_sight.png")


# -- 3. the HMM ------------------------------------------------------------------------------------
def hmm_states():
    import math
    import hmm
    switch = math.log(hmm.POSSESSION_S * 25)
    die = math.log(hmm.LIVE_S * 25)
    revive = math.log(2 * hmm.DEAD_S * 25)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.75, 4)
    ax.axis("off")
    ax.grid(False)
    pos = {"H": (2.0, 2.9), "A": (8.0, 2.9), "D": (5.0, 0.8)}
    for s, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.55, facecolor=BLUE if s != "D" else MUTED, edgecolor=SURFACE,
                                linewidth=2))
        ax.text(x, y, s, ha="center", va="center", fontsize=16, color="#ffffff", weight="bold")
    arrow(ax, (2.6, 3.05), (7.4, 3.05), color=INK2, rad=-0.15)
    arrow(ax, (7.4, 2.75), (2.6, 2.75), color=INK2, rad=-0.15)
    ax.text(5.0, 3.75, "switch H <-> A: log({:g} s x 25 fps) ~ {:.1f} nats (mean possession, IDSSE)".format(
        hmm.POSSESSION_S, switch), ha="center", fontsize=8.5, color=INK2)
    arrow(ax, (2.4, 2.45), (4.5, 1.05), color=INK2, rad=0.1)
    arrow(ax, (4.55, 1.25), (2.5, 2.5), color=INK2, rad=0.1)
    arrow(ax, (7.6, 2.45), (5.5, 1.05), color=INK2, rad=-0.1)
    arrow(ax, (5.45, 1.25), (7.5, 2.5), color=INK2, rad=-0.1)
    ax.text(1.2, 1.35, "H or A -> D\nlog({:g} s x fps) ~ {:.1f}".format(hmm.LIVE_S, die), ha="center", fontsize=8,
            color=INK2)
    ax.text(8.8, 1.35, "D -> H or A\nlog(2 x {:g} s x fps) ~ {:.1f}".format(hmm.DEAD_S, revive), ha="center",
            fontsize=8, color=INK2)
    ax.text(5.0, -0.4, "per-frame evidence (nats):\nplayers' distances (confirm only), touches, kicks,\n"
            "controls, carries, dead-ball and live signals", ha="center", fontsize=8.5, color=INK)
    save(fig, "hmm_states.png")


# -- 4. external validation, per game ----------------------------------------------------------------
def external_scores():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "validation", "readiness_games.csv"), newline="")))
    games = sorted({r["game"] for r in rows})
    show = (("truth 1 s late", "ground truth 1 s late (scale)", MUTED),
            ("R1", "R1: every touch hands over", ORANGE),
            ("before (2026-11-16)", "model C, 2026-11-16", AQUA),
            ("submitted", "submitted model", BLUE))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=False)
    for ax, metric, label in ((axes[0], "spell_f1", "spellF1 (ranking metric)"), (axes[1], "macro_f1", "macroF1")):
        y = np.arange(len(games))
        for k, (model, name, color) in enumerate(show):
            vals = [float(next(r[metric] for r in rows if r["game"] == g and r["model"] == model)) for g in games]
            ax.plot(vals, y + (k - 1.5) * 0.16, marker="o", markersize=6.5, linestyle="none", color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
        ax.set_yticks(y)
        ax.set_yticklabels([g.replace("idsse_", "IDSSE ").replace("metrica_", "Metrica ") for g in games], fontsize=8)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    axes[1].set_yticklabels([])
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.17), ncol=4, fontsize=8)
    fig.suptitle("Clean optical tracking from 10 public matches (not BAR-ATH), one point per match", fontsize=10,
                 x=0.02, ha="left")
    save(fig, "external_scores.png")


# -- 5. finding 5: matched possessions by true length --------------------------------------------------
def by_length():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "validation", "readiness_sets.csv"), newline="")))
    buckets = ["<1s", "1-2s", "2-4s", "4-8s", ">8s"]
    bins = ["< 1 s", "1-2 s", "2-4 s", "4-8 s", "> 8 s"]
    show = (("truth 1 s late", "ground truth 1 s late", MUTED), ("R1", "R1: every touch hands over", ORANGE),
            ("submitted", "submitted model", BLUE))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    x = np.arange(len(bins))
    for ax, ds in zip(axes, ("IDSSE", "Metrica")):
        n = next(r for r in rows if r["dataset"] == ds and r["copy"] == "clean" and r["model"] == "submitted")
        for model, name, color in show:
            r = next(r for r in rows if r["dataset"] == ds and r["copy"] == "clean" and r["model"] == model)
            ax.plot(x, [100 * float(r["matched_" + b]) for b in buckets], color=color, linewidth=2, marker="o",
                    markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(["{}\n(n {})".format(b, n["n_" + k]) for b, k in zip(bins, buckets)], fontsize=8)
        ax.set_xlabel("true possession length")
        ax.set_title("{} (clean tracking)".format(ds), fontsize=9.5, loc="left")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("% of true possessions matched (IoU 0.5)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Short possessions are where the model loses: partly by design (one uncontrolled touch is not a "
                 "change), partly a limitation", fontsize=9.5, x=0.02, ha="left", y=1.04)
    save(fig, "match_by_length.png")


if __name__ == "__main__":
    pipeline()
    line_of_sight()
    hmm_states()
    external_scores()
    by_length()
