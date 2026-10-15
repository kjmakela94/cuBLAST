"""Render the 'what we multiply' tiles for the deck from results/<run>/data/B1_structure.json.
Usage: python make_matrix_figs.py <results_dir> [out_dir=figs]"""
import json, sys, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

RES = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "figs"; os.makedirs(OUT, exist_ok=True)
NAVY, GREEN, AMBER, BLUE, EDGE = "#2B3A4F", "#76B900", "#E6A817", "#3B82F6", "#C9D1DE"
CMAP = LinearSegmentedColormap.from_list("d", ["#DDEFB8", GREEN, "#2F6B1E", NAVY])

def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_color(EDGE); sp_.set_linewidth(1.5)

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02, facecolor="white"); plt.close(fig)

# sparse matrices: non-zero density maps
for m in json.load(open(os.path.join(RES, "data", "B1_structure.json"))):
    H = np.ma.masked_equal(np.array(m["density"], dtype=float), 0)
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.set_facecolor("white"); ax.imshow(H, cmap=CMAP, norm=LogNorm(vmin=1), interpolation="nearest"); frame(ax)
    save(fig, m["matrix"])
    if "zoom" in m:                                   # top-left corner: shows the local stencil / band
        Z = np.ma.masked_equal(np.array(m["zoom"], dtype=float), 0)
        fig, ax = plt.subplots(figsize=(1.2, 1.2))
        ax.set_facecolor("white"); ax.imshow(Z, cmap=CMAP, norm=LogNorm(vmin=1), interpolation="nearest"); frame(ax)
        save(fig, m["matrix"] + "_zoom")

# dense GEMM operand
fig, ax = plt.subplots(figsize=(2.4, 2.4))
rng = np.random.default_rng(3); ax.imshow(rng.random((24, 24)), cmap=LinearSegmentedColormap.from_list("g", ["#3E4E66", NAVY]), interpolation="nearest"); frame(ax)
save(fig, "dense")

# CG model problem: 2-D Poisson 5-point stencil on a grid
fig, ax = plt.subplots(figsize=(2.4, 2.4)); g = 7
for i in range(g):
    ax.plot([0, g - 1], [i, i], color=EDGE, lw=1.2, zorder=1); ax.plot([i, i], [0, g - 1], color=EDGE, lw=1.2, zorder=1)
xx, yy = np.meshgrid(range(g), range(g)); ax.scatter(xx, yy, s=28, color="#B7C1CF", zorder=2)
c = g // 2
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    ax.plot([c, c + dx], [c, c + dy], color=BLUE, lw=3.5, zorder=3); ax.scatter([c + dx], [c + dy], s=150, color=BLUE, zorder=4)
ax.scatter([c], [c], s=260, color=NAVY, zorder=5)
ax.text(c + 0.3, c + 0.32, "4", color="white", fontsize=0.1)  # keep tight bbox stable
ax.set_xlim(-0.6, g - 0.4); ax.set_ylim(-0.6, g - 0.4); ax.set_aspect("equal"); frame(ax)
save(fig, "poisson")
print("tiles in", OUT, sorted(os.listdir(OUT)))
