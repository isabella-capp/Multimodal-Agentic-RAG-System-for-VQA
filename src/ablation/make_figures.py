from __future__ import annotations

import argparse
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
COL_W = 3.25          # inches: one column of a two-column letter layout


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def save(fig, path: str) -> None:
    """PDF for the paper, PNG beside it for looking at."""
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=300)


def ladder(run_dir: str) -> list[tuple[str, float, float]]:
    """(name, coverage %, BEM) for the arms that retrieve, in ladder order."""
    out = []
    for arm, label in [("B", "B"), ("Bplus", "B+name"),
                       ("Btext", "B+text"), ("Bgated", "B+gated")]:
        res = json.load(open(f"{run_dir}/results_{arm}.json"))
        cov = json.load(open(f"{run_dir}/pool_{arm}.json"))["percent"]["union"]
        out.append((label, cov, 100 * res["accuracy_overall"]))
    return out


def fig_coverage_vs_accuracy(run_dir: str, path: str) -> None:
    """Why coverage is not the objective: the last step goes left and up."""
    pts = ladder(run_dir)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    xs, ys = [p[1] for p in pts], [p[2] for p in pts]

    ax.plot(xs[:3], ys[:3], "-", color=BLUE, lw=1.2, zorder=1)
    ax.plot(xs[2:], ys[2:], "-", color=ORANGE, lw=1.4, zorder=1)
    ax.plot(xs, ys, "o", color=BLUE, ms=5, mec="white", mew=1.0, zorder=2)
    ax.plot(xs[-1], ys[-1], "o", color=ORANGE, ms=5.5, mec="white", mew=1.0, zorder=3)

    for (name, x, y), (dx, dy, ha) in zip(pts, [(6, -2, "left"),
                                                (6, -3, "left"),
                                                (0, -11, "center"),
                                                (-7, 1, "right")]):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va="center",
                    fontsize=7.5, color=INK)

    ax.annotate("less coverage,\nhigher accuracy", xy=(xs[-1], (ys[-1] + ys[-2]) / 2),
                xytext=(-10, 0), textcoords="offset points",
                fontsize=7, color=ORANGE, ha="right", va="center")

    ax.set_xlim(min(xs) - 2.5, max(xs) + 2.5)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.set_xlabel("gold article in candidate pool (%)")
    ax.set_ylabel("BEM")
    ax.grid(True, lw=0.4, color="#e6e6e4", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, path)
    plt.close(fig)
    print(f"  {path}")
    for name, cov, bem in pts:
        print(f"      {name:9} cov {cov:5.1f}  BEM {bem:5.1f}")


def fig_gate_scores(probe: str, path: str, tau: float = -1.0) -> None:
    """What the gate sees: the score separates pools that hold the answer."""
    rows = [json.loads(l) for l in open(probe)]
    hit = [r["top_score"] for r in rows if r["image_hit"]]
    miss = [r["top_score"] for r in rows if not r["image_hit"]]

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    bins = [-11 + 0.8 * i for i in range(28)]
    for data, colour, label in ((hit, BLUE, "gold article present"),
                                (miss, ORANGE, "gold article absent")):
        w = [100 / len(data)] * len(data)
        ax.hist(data, bins=bins, weights=w, histtype="step", lw=1.4,
                color=colour, label=f"{label} ($n$={len(data)})")

    ax.axvline(tau, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=3)
    ax.annotate(f"$\\tau={tau:g}$", xy=(tau, 0), xycoords=("data", "axes fraction"),
                xytext=(3, 4), textcoords="offset points",
                fontsize=7, color=INK, ha="left", va="bottom")

    ax.set_xlabel("cross-encoder score, best pooled passage")
    ax.set_ylabel("% of group")
    ax.legend(frameon=False, loc="upper right", handlelength=1.3,
              borderpad=0.2, labelspacing=0.3)
    ax.grid(True, axis="y", lw=0.4, color="#e6e6e4", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, path)
    plt.close(fig)
    below = lambda v: 100 * sum(1 for s in v if s < tau) / len(v)  # noqa: E731
    print(f"  {path}")
    print(f"      median  present {statistics.median(hit):+.2f}   "
          f"absent {statistics.median(miss):+.2f}")
    print(f"      below tau: {below(miss):.1f}% of absent, {below(hit):.1f}% of present")


def main() -> None:
    p = argparse.ArgumentParser(description="Build the paper figures from the recorded results.")
    p.add_argument("--ladder",
                   default="outputs/baselines/qwen3vl8b/20260910-210504-b-ladder-best")
    p.add_argument("--probe", default="outputs/retrieval/gate.jsonl")
    p.add_argument("--out", default="outputs/paper/figures")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    style()
    fig_coverage_vs_accuracy(args.ladder, f"{args.out}/coverage_vs_accuracy.pdf")
    fig_gate_scores(args.probe, f"{args.out}/gate_scores.pdf")

if __name__ == "__main__":
    main()
