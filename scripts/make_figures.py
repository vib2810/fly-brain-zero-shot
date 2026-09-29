"""Per-fly results table and the README figures from the runs in runs/vision_loop/.

  python scripts/make_figures.py   -> assets/results_per_fly.csv, assets/results.png, assets/pipeline.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RUNS, ASSETS = Path("runs/vision_loop"), Path("assets")
BG, FG, GREY = "#0d0f14", "#ffffff", "#aab0bd"
GREEN, RED, BLUE, AMBER, PINK = "#50dc8c", "#ff4d5e", "#39a0ff", "#ffbe46", "#ff5cbe"


def collect():
    """One row per fly: window, condition, run, poles reached in 10 s."""
    rows = []

    def add(window, cond, run, hits):
        rows.extend(dict(window=window, condition=cond, run=run, fly=i, poles=int(h)) for i, h in enumerate(hits))

    for path in sorted(RUNS.glob("eval*.npz")):  # `eval` / `record` runs: replicates of one design
        if path.stem == "eval_scramble":
            continue
        run, z = path.stem, np.load(path)
        n = int(z["n"])
        add("intact, 0-10 s", "real wiring", run, z["intact_hit"][:n, :1000].sum(1))
        add("intact, 0-10 s", "turn neurons unplugged", run, z["intact_hit"][n:, :1000].sum(1))
        add("2 legs cut, 5-15 s", "real wiring", run, z["cut_hit"][:n].sum(1))
        add("2 legs cut, 5-15 s", "turn neurons unplugged", run, z["cut_hit"][n:].sum(1))
    for path in sorted(RUNS.glob("shuffle*.npz")):  # scrambled wiring on intact flies, same gain k = 2
        z = np.load(path)
        add("intact, 0-10 s", "wiring scrambled", path.stem, z["hit"][z["k"] == 2.0].sum(1))
    if (RUNS / "eval_scramble.npz").exists():  # clones at the cut: real vs scrambled wiring
        z = np.load(RUNS / "eval_scramble.npz")
        n = int(z["n"])
        add("2 legs cut, 5-15 s", "real wiring", "eval_scramble", z["cut_hit"][:n].sum(1))
        add("2 legs cut, 5-15 s", "wiring scrambled", "eval_scramble", z["cut_hit"][n:].sum(1))
    return pd.DataFrame(rows)


def results_figure(df):
    conds = [("real wiring", GREEN), ("wiring scrambled", RED), ("turn neurons unplugged", GREY)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, facecolor=BG)
    rng = np.random.default_rng(0)
    for ax, window in zip(axes, ["intact, 0-10 s", "2 legs cut, 5-15 s"]):
        ax.set_facecolor(BG)
        for x, (cond, col) in enumerate(conds):
            v = df[(df.window == window) & (df.condition == cond)].poles.to_numpy()
            if not len(v):
                continue
            ax.bar(x, v.mean(), color=col, alpha=0.25, width=0.7, edgecolor=col, linewidth=2)
            ax.scatter(x + rng.uniform(-0.22, 0.22, len(v)), v + rng.uniform(-0.12, 0.12, len(v)), s=12, color=col,
                       alpha=0.8, linewidths=0)
            ax.text(x, v.mean() + 0.25, f"{v.mean():.2f}", ha="center", color=FG, fontsize=13, fontweight="bold")
            ax.text(x, -0.55, f"n={len(v)}", ha="center", color=GREY, fontsize=9)
        ax.set_xticks(range(3), [c.replace(" ", "\n", 1) for c, _ in conds], color=FG, fontsize=10)
        ax.set_title(window.replace("-", "–"), color=FG, fontsize=13, fontweight="bold")
        ax.tick_params(colors=GREY)
        for sp in ax.spines.values():
            sp.set_color("#2a2f3a")
        ax.set_ylim(-0.8, 6.3)
    axes[0].set_ylabel("poles reached in 10 s (per fly)", color=FG, fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "results.png", dpi=160, facecolor=BG)


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(12, 3.4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 11.9)
    ax.set_ylim(-0.35, 3.5)
    ax.axis("off")
    boxes = [
        ("Optics", "bright arena, dark pole\n2 × 721 eye facets", PINK),
        ("Eyes", "FlyVis optic-lobe model\n45,669 neurons", BLUE),
        ("Brain", "Shiu et al. FlyWire model\n138,639 spiking neurons", AMBER),
        ("Turn command", "DNa02 left − right\n× one tuned gain", GREEN),
        ("Body", "NeuroMechFly physics\nhand-built leg rhythm", GREY),
    ]
    w, step = 2.1, 2.35
    for k, (title, sub, col) in enumerate(boxes):
        x = 0.1 + k * step
        ax.add_patch(FancyBboxPatch((x, 1.2), w, 1.5, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor="#161a22", edgecolor=col, linewidth=2))
        ax.text(x + w / 2, 2.3, title, ha="center", va="center", color=col, fontsize=13, fontweight="bold")
        ax.text(x + w / 2, 1.65, sub, ha="center", va="center", color=FG, fontsize=9)
    for k, lab in enumerate(["every 10 ms", "27 cell types →\n31k FlyWire inputs", "spikes", "turn"]):
        x0 = 0.1 + k * step + w + 0.02
        ax.add_patch(FancyArrowPatch((x0, 1.95), (x0 + step - w - 0.04, 1.95), arrowstyle="-|>", mutation_scale=16,
                                     color=GREY))
        ax.text(x0 + (step - w) / 2, 2.85, lab, ha="center", va="bottom", color=GREY, fontsize=8.5)
    ax.add_patch(FancyArrowPatch((0.1 + 4 * step + w / 2, 1.15), (0.1 + w / 2, 1.15), connectionstyle="arc3,rad=-0.12",
                                 arrowstyle="-|>", mutation_scale=18, color=GREY, linestyle="--"))
    ax.text(5.95, -0.25, "closed loop: where the body walks changes what the eyes see next", ha="center", color=GREY,
            fontsize=10)
    fig.savefig(ASSETS / "pipeline.png", dpi=160, facecolor=BG, bbox_inches="tight")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    df = collect()
    df.to_csv(ASSETS / "results_per_fly.csv", index=False)
    print(df.groupby(["window", "condition"]).poles.agg(["mean", "count"]).round(2))
    results_figure(df)
    pipeline_figure()
