"""Combined 2×2 figure for β sweep experiments → results/beta_sweep.png."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_beta_sweep_figure(
    betas: list[float],
    entropy: list[float],
    anon: list[float],
    fee_overhead_pct: list[float],
    adversary_acc: list[float],
    *,
    out_path: Path,
) -> Path:
    """Plot β sweep metrics. Data are sorted by β ascending on a common x-axis."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    x = np.asarray(betas, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    H = np.asarray(entropy, dtype=float)[order]
    A = np.asarray(anon, dtype=float)[order]
    F = np.asarray(fee_overhead_pct, dtype=float)[order]
    Adv = np.asarray(adversary_acc, dtype=float)[order]

    xp = np.arange(len(xs), dtype=float)
    x_labels = [str(b) for b in xs]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")

    # Top-left: H vs β
    ax = axes[0, 0]
    ax.plot(xp, H, marker="o", color="#7BDFF2", linewidth=2)
    ax.axhline(1.58, color="#ffca85", linestyle="--", linewidth=1.2, label="Paper estimate at β=2")
    ax.set_ylabel("H (bits)")
    ax.set_title("Shannon entropy")
    ax.legend(frameon=False, loc="best", fontsize=9)

    # Top-right: adversary accuracy
    ax = axes[0, 1]
    ax.plot(xp, Adv, marker="o", color="#8CF078", linewidth=2)
    ax.axhline(1.0, color="#ff6b6b", linestyle="--", linewidth=1.2, label="Baseline (LN today)")
    ax.set_ylabel("Adversary accuracy")
    ax.set_title("Deterministic adversary")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="best", fontsize=9)

    # Bottom-left: 2^H bars (same x positions)
    ax = axes[1, 0]
    ax.bar(xp, A, color="#CBA6F7", alpha=0.9, align="center")
    ax.axhline(10, color="#ffca85", linestyle="--", linewidth=1.2, label="Paper estimate at β=2")
    ax.set_ylabel("2^H (est. anon set)")
    ax.set_title("Anonymity set size")
    ax.legend(frameon=False, loc="best", fontsize=9)

    # Bottom-right: fee overhead with fill
    ax = axes[1, 1]
    ax.fill_between(xp, 0, F, color="#FF935C", alpha=0.35)
    ax.plot(xp, F, marker="o", color="#FF935C", linewidth=2)
    ax.axhline(12, color="#ffca85", linestyle="--", linewidth=1.2, label="Paper estimate at β=2")
    ax.set_ylabel("Fee overhead %")
    ax.set_title("Fee vs deterministic route")
    ax.legend(frameon=False, loc="best", fontsize=9)

    for ax in axes[1, :]:
        ax.set_xlabel("β")

    for ax in axes.flat:
        ax.set_xticks(xp)
        ax.set_xticklabels(x_labels)
        ax.set_xlim(xp.min() - 0.5, xp.max() + 0.5)

    fig.suptitle("PCAR β sweep", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=175, bbox_inches="tight")
    plt.close(fig)
    return out_path
