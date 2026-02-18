"""
eval.py  —  DQN Pac-Man Training Evaluation
============================================
Reads checkpoint metadata from the checkpoints/ folder and produces a
single publication-quality figure containing 8 performance panels.

Usage
-----
    python eval.py                          # load latest checkpoint
    python eval.py --checkpoint path.h5     # load specific checkpoint
    python eval.py --all                    # merge all checkpoints in order
    python eval.py --smooth 100             # change moving-average window
    python eval.py --out results.pdf        # change output filename

Output
------
  results.png  —  300 DPI raster  (good for Word / PowerPoint)
  results.pdf  —  vector          (good for LaTeX)
  results_stats.txt — summary statistics table

Panels
------
  1. Episode Reward       — shaped reward with moving average + std band
  2. Game Score           — raw Pac-Man score per episode
  3. Win Rate             — rolling % of episodes cleared
  4. Pellets Eaten        — pellets per episode
  5. Episode Length       — agent decision steps per episode
  6. Training Loss        — Huber loss over training steps (smoothed)
  7. Epsilon Decay        — exploration rate over episodes
  8. Outcome Distribution — pie chart (win / death / timeout counts)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  Styling
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "reward":   "#2196F3",   # blue
    "score":    "#4CAF50",   # green
    "win_rate": "#FF9800",   # orange
    "pellets":  "#9C27B0",   # purple
    "steps":    "#00BCD4",   # teal
    "loss":     "#F44336",   # red
    "epsilon":  "#607D8B",   # grey-blue
    "avg":      "#212121",   # near-black for moving average lines
    "fill":     0.18,        # alpha for std shading
}

# Line widths — change these to adjust all plots at once
LW = {
    "raw":    1.2,   # per-episode scatter/raw series
    "avg":    2.5,   # moving average line
    "ref":    1.5,   # reference / threshold lines (e.g. board cleared)
    "win":    2.5,   # win-rate fill line
    "eps":    2.0,   # epsilon curve
    "loss_r": 1.0,   # raw loss (thin background)
    "loss_s": 2.5,   # smoothed loss
    "grid":   0.6,   # grid lines
}

FONT = {
    "family": "DejaVu Sans",
    "size":   9,
}

FIG_W, FIG_H = 14, 10   # inches
DPI          = 300


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────
def find_checkpoints(ckpt_dir: str = "checkpoints") -> list[str]:
    """Return all .meta.json files sorted by step number."""
    p = Path(ckpt_dir)
    if not p.exists():
        return []
    files = sorted(
        p.glob("*.meta.json"),
        key=lambda f: int("".join(filter(str.isdigit, f.stem)) or "0")
    )
    return [str(f) for f in files]


def load_meta(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def merge_metas(paths: list[str]) -> dict:
    """
    Merge multiple sequential metadata files into one.
    Later checkpoints already contain cumulative history, so we just
    take the last file (which has the most data).
    """
    if not paths:
        raise FileNotFoundError("No metadata files found in checkpoints/")
    # The last checkpoint contains the full history
    return load_meta(paths[-1])


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def moving_avg(values: list, window: int) -> np.ndarray:
    """Simple causal moving average."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def moving_std(values: list, window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    out = np.array([
        arr[max(0, i - window + 1): i + 1].std()
        for i in range(len(arr))
    ])
    return out[window - 1:]


def x_for_avg(n_total: int, window: int) -> np.ndarray:
    """Episode indices aligned with the valid portion of a moving average."""
    return np.arange(window - 1, n_total)


def styled_ax(ax, title: str, xlabel: str, ylabel: str):
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=LW["grid"], alpha=0.5, linestyle="--")


def plot_series(ax, raw_y, window, color, raw_alpha=0.25,
                raw_label="Per episode", avg_label=None):
    """Plot raw values + moving average + std shading."""
    episodes = np.arange(len(raw_y))
    ax.plot(episodes, raw_y, color=color, alpha=raw_alpha,
            linewidth=LW["raw"], label=raw_label)

    if len(raw_y) >= window:
        avg = moving_avg(raw_y, window)
        std = moving_std(raw_y, window)
        x   = x_for_avg(len(raw_y), window)
        lbl = avg_label or f"{window}-ep avg"
        ax.plot(x, avg, color=PALETTE["avg"], linewidth=LW["avg"], label=lbl)
        ax.fill_between(x, avg - std, avg + std,
                        color=color, alpha=PALETTE["fill"], label="±1 SD")


# ─────────────────────────────────────────────────────────────────────────────
#  Summary statistics (printed + saved to .txt)
# ─────────────────────────────────────────────────────────────────────────────
def compute_summary(meta: dict, window: int) -> str:
    rewards  = meta.get("episode_rewards",  [])
    scores   = meta.get("episode_scores",   [])
    pellets  = meta.get("episode_pellets",  [])
    steps    = meta.get("episode_steps",    [])
    outcomes = meta.get("episode_outcomes", [])
    losses   = meta.get("losses",           [])

    n = len(rewards)
    if n == 0:
        return "No episode data found."

    total_steps = meta.get("step_count", 0)
    epsilon     = meta.get("epsilon",    float("nan"))

    wins     = outcomes.count("win")
    deaths   = outcomes.count("death")
    timeouts = outcomes.count("timeout")

    last_w = min(window, n)
    lines = [
        "=" * 56,
        "  DQN PAC-MAN  —  TRAINING SUMMARY",
        "=" * 56,
        f"  Total episodes      : {n:,}",
        f"  Total steps         : {total_steps:,}",
        f"  Final epsilon (ε)   : {epsilon:.4f}",
        "",
        "  Reward (shaped)",
        f"    All-time mean     : {np.mean(rewards):+.1f}",
        f"    All-time std      : {np.std(rewards):.1f}",
        f"    Last {last_w}-ep mean  : {np.mean(rewards[-last_w:]):+.1f}",
        f"    Best episode      : {max(rewards):+.1f}",
        "",
        "  Game score",
        f"    All-time mean     : {np.mean(scores):.1f}" if scores else "    (not recorded)",
        f"    Last {last_w}-ep mean  : {np.mean(scores[-last_w:]):.1f}" if scores else "",
        f"    Best episode      : {max(scores):.0f}" if scores else "",
        "",
        "  Pellets eaten / episode",
        f"    All-time mean     : {np.mean(pellets):.1f}" if pellets else "    (not recorded)",
        f"    Last {last_w}-ep mean  : {np.mean(pellets[-last_w:]):.1f}" if pellets else "",
        f"    Max               : {max(pellets)}" if pellets else "",
        "",
        "  Episode outcomes",
        f"    Wins              : {wins}  ({100*wins/n:.1f}%)",
        f"    Deaths            : {deaths}  ({100*deaths/n:.1f}%)",
        f"    Timeouts          : {timeouts}  ({100*timeouts/n:.1f}%)",
        "",
        "  Training loss",
        f"    Mean              : {np.mean(losses):.4f}" if losses else "    (not recorded)",
        f"    Final 1k-step avg : {np.mean(losses[-100:]):.4f}" if len(losses) >= 100 else "",
        "=" * 56,
    ]
    return "\n".join(line for line in lines if line is not None)


# ─────────────────────────────────────────────────────────────────────────────
#  Main plot
# ─────────────────────────────────────────────────────────────────────────────
def make_figure(meta: dict, window: int, out_stem: str):
    rewards  = meta.get("episode_rewards",  [])
    scores   = meta.get("episode_scores",   [])
    pellets  = meta.get("episode_pellets",  [])
    steps    = meta.get("episode_steps",    [])
    outcomes = meta.get("episode_outcomes", [])
    epsilons = meta.get("episode_epsilons", [])
    losses   = meta.get("losses",           [])
    loss_x   = meta.get("loss_steps",       [])
    n = len(rewards)

    # ── Figure & grid ────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": FONT["family"],
        "font.size":   FONT["size"],
        "figure.facecolor": "white",
        "axes.facecolor":   "#FAFAFA",
    })

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.52, wspace=0.38,
                            left=0.07, right=0.97,
                            top=0.93,  bottom=0.07)

    axes = {
        "reward":   fig.add_subplot(gs[0, :2]),   # wide, top-left
        "win_rate": fig.add_subplot(gs[0,  2]),
        "score":    fig.add_subplot(gs[1,  0]),
        "pellets":  fig.add_subplot(gs[1,  1]),
        "steps":    fig.add_subplot(gs[1,  2]),
        "loss":     fig.add_subplot(gs[2,  0]),
        "epsilon":  fig.add_subplot(gs[2,  1]),
        "outcomes": fig.add_subplot(gs[2,  2]),
    }

    # ── 1. Episode reward ────────────────────────────────────────────────────
    ax = axes["reward"]
    if rewards:
        plot_series(ax, rewards, window, PALETTE["reward"])
        ax.axhline(0, color="#999", linewidth=LW["ref"], linestyle=":")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
    styled_ax(ax, "Episode Reward (shaped)", "Episode", "Total reward")

    # ── 2. Win rate ──────────────────────────────────────────────────────────
    ax = axes["win_rate"]
    if outcomes:
        wins_binary = [1.0 if o == "win" else 0.0 for o in outcomes]
        if len(wins_binary) >= window:
            wr  = moving_avg(wins_binary, window) * 100
            x   = x_for_avg(len(wins_binary), window)
            ax.plot(x, wr, color=PALETTE["win_rate"], linewidth=LW["win"])
            ax.fill_between(x, 0, wr, color=PALETTE["win_rate"], alpha=0.15)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    styled_ax(ax, f"Win Rate ({window}-ep rolling)", "Episode", "Win %")

    # ── 3. Game score ────────────────────────────────────────────────────────
    ax = axes["score"]
    if scores:
        plot_series(ax, scores, window, PALETTE["score"])
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
    styled_ax(ax, "Game Score", "Episode", "Score")

    # ── 4. Pellets eaten ─────────────────────────────────────────────────────
    ax = axes["pellets"]
    if pellets:
        total_pellets = 240  # standard Pac-Man maze
        ax.axhline(total_pellets, color="#aaa", linewidth=LW["ref"],
                   linestyle="--", label="Board cleared (240)")
        plot_series(ax, pellets, window, PALETTE["pellets"])
        ax.set_ylim(0, total_pellets + 10)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
    styled_ax(ax, "Pellets Eaten / Episode", "Episode", "Pellets")

    # ── 5. Episode length ────────────────────────────────────────────────────
    ax = axes["steps"]
    if steps:
        plot_series(ax, steps, window, PALETTE["steps"])
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    styled_ax(ax, "Episode Length", "Episode", "Decision steps")

    # ── 6. Training loss ─────────────────────────────────────────────────────
    ax = axes["loss"]
    if losses:
        if loss_x and len(loss_x) == len(losses):
            lx = np.array(loss_x)
        else:
            lx = np.arange(len(losses))

        ax.plot(lx, losses, color=PALETTE["loss"],
                alpha=0.2, linewidth=LW["loss_r"], label="Raw loss")
        sm_w = max(1, len(losses) // 100)
        if len(losses) >= sm_w:
            smooth = moving_avg(losses, sm_w)
            sx     = lx[sm_w - 1:]
            ax.plot(sx, smooth, color=PALETTE["avg"],
                    linewidth=LW["loss_s"], label=f"{sm_w}-step avg")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        ax.set_yscale("log")
    styled_ax(ax, "Training Loss (Huber)", "Training step", "Loss (log)")

    # ── 7. Epsilon decay ─────────────────────────────────────────────────────
    ax = axes["epsilon"]
    if epsilons:
        ax.plot(range(len(epsilons)), epsilons,
                color=PALETTE["epsilon"], linewidth=LW["eps"])
        ax.axhline(epsilons[-1], color="#aaa", linewidth=LW["ref"],
                   linestyle="--", label=f"Final ε = {epsilons[-1]:.3f}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    styled_ax(ax, "Exploration Rate (ε)", "Episode", "Epsilon")

    # ── 8. Outcome distribution ──────────────────────────────────────────────
    ax = axes["outcomes"]
    if outcomes:
        counts = {
            "Win":     outcomes.count("win"),
            "Death":   outcomes.count("death"),
            "Timeout": outcomes.count("timeout"),
        }
        counts = {k: v for k, v in counts.items() if v > 0}
        colors = ["#4CAF50", "#F44336", "#FF9800"][:len(counts)]
        wedges, texts, autotexts = ax.pie(
            counts.values(),
            labels=counts.keys(),
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 8},
            wedgeprops={"linewidth": LW["ref"], "edgecolor": "white"},
        )
        for t in autotexts:
            t.set_fontsize(7)
        ax.set_title("Outcome Distribution", fontsize=10,
                     fontweight="bold", pad=6)
    else:
        ax.text(0.5, 0.5, "No outcome data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#aaa")
        ax.set_title("Outcome Distribution", fontsize=10,
                     fontweight="bold", pad=6)
        ax.axis("off")

    # ── Super-title ──────────────────────────────────────────────────────────
    total_steps = meta.get("step_count", 0)
    eps_final   = meta.get("epsilon",    float("nan"))
    fig.suptitle(
        f"Pac-Man DQN Training  —  {n} episodes  |  "
        f"{total_steps:,} steps  |  ε = {eps_final:.4f}",
        fontsize=12, fontweight="bold", y=0.975
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    png_path = out_stem + ".png"
    pdf_path = out_stem + ".pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Saved → {png_path}  (300 DPI)")
    print(f"[eval] Saved → {pdf_path}  (vector)")
    return png_path, pdf_path


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Plot DQN Pac-Man training results")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a specific .h5 or .meta.json file")
    p.add_argument("--all", action="store_true",
                   help="Merge all checkpoints (uses latest which has full history)")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints",
                   help="Directory containing checkpoint files (default: checkpoints/)")
    p.add_argument("--smooth", type=int, default=50,
                   help="Moving-average window in episodes (default: 50)")
    p.add_argument("--out", type=str, default="results",
                   help="Output filename stem, no extension (default: results)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Locate metadata ──────────────────────────────────────────────────────
    if args.checkpoint:
        # Accept either the .h5 path or the .meta.json path
        meta_path = args.checkpoint
        if not meta_path.endswith(".meta.json"):
            meta_path += ".meta.json"
        if not os.path.exists(meta_path):
            print(f"[eval] ERROR: metadata not found at {meta_path}")
            sys.exit(1)
        meta = load_meta(meta_path)
        print(f"[eval] Loaded {meta_path}")

    else:
        ckpts = find_checkpoints(args.ckpt_dir)
        if not ckpts:
            print(f"[eval] ERROR: no .meta.json files in {args.ckpt_dir}/")
            print("       Train first, or pass --checkpoint <path>")
            sys.exit(1)
        meta = merge_metas(ckpts)
        print(f"[eval] Using latest checkpoint: {ckpts[-1]}")

    n = len(meta.get("episode_rewards", []))
    if n == 0:
        print("[eval] No episode data in metadata — train for at least one "
              "episode before evaluating.")
        sys.exit(0)

    print(f"[eval] {n} episodes found. Generating plots...")

    # ── Summary stats ────────────────────────────────────────────────────────
    summary = compute_summary(meta, window=args.smooth)
    print("\n" + summary)
    stats_path = args.out + "_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n[eval] Stats saved → {stats_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    make_figure(meta, window=args.smooth, out_stem=args.out)


if __name__ == "__main__":
    main()
