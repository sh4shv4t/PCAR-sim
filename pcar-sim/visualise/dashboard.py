"""Multi-panel dashboard assembly for PCAR-sim (3×2 grid → results/dashboard.png)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import PCARConfig
from core.metrics import SimulationResults

from visualise.entropy_plot import plot_entropy_suite, rolling_mean
from visualise.network_plot import plot_network_demo
from visualise.tradeoff_plot import plot_tradeoff_suite


def build_dashboard(cfg: PCARConfig, results: SimulationResults, results_dir: Path) -> Path:
    plt.style.use("dark_background")
    results_dir.mkdir(parents=True, exist_ok=True)

    paths_net = results_dir / "network_demo.png"
    plot_network_demo(results.graph, results.demo, paths_net)

    entropy_paths, entropy_curves = plot_entropy_suite(cfg, results, results_dir)
    trade_paths, beta_grid, succ_rates = plot_tradeoff_suite(cfg, results, results_dir)

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.33, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    img_net = plt.imread(paths_net)
    ax0.imshow(img_net)
    ax0.axis("off")
    ax0.set_title("Topology heatmap & exemplar routes")

    ax1 = fig.add_subplot(gs[0, 1])
    for b, ent in entropy_curves.items():
        ax1.plot(np.arange(1, ent.size + 1), ent, linewidth=1.1, label=f"β={b}")
    ax1.set_title("Entropy vs payments (β sweep)")
    ax1.set_xlabel("Payment index")
    ax1.set_ylabel("H (bits)")
    ax1.legend(frameon=False, loc="lower right")

    ax2 = fig.add_subplot(gs[1, 0])
    adv_b = np.array([1.0 if x.adversary_correct else 0.0 for x in results.baseline_logs], dtype=float)
    adv_p = np.array([1.0 if x.adversary_correct else 0.0 for x in results.pcar_logs], dtype=float)
    idx = np.arange(1, adv_b.size + 1)
    ax2.plot(idx, rolling_mean(adv_b, 40), label="Baseline (deterministic)", color="#FF935C")
    ax2.plot(idx, rolling_mean(adv_p, 40), label="PCAR (probabilistic)", color="#8CF078")
    ax2.set_title("Adversary prediction accuracy (rolling)")
    ax2.set_xlabel("Payment index")
    ax2.set_ylabel("Accuracy")
    ax2.legend(frameon=False)

    ax3 = fig.add_subplot(gs[1, 1])
    fees = np.array([p.fee_overhead_pct_vs_baseline for p in results.pcar_logs], dtype=float)
    ano = np.array([p.anonymity_size for p in results.pcar_logs], dtype=float)
    betas_eff = np.array(
        [p.beta_effective if p.beta_effective is not None else cfg.beta for p in results.pcar_logs],
        dtype=float,
    )
    sc = ax3.scatter(fees, ano, c=betas_eff, cmap="plasma", s=14, alpha=0.85)
    cb = fig.colorbar(sc, ax=ax3, fraction=0.046)
    cb.set_label("β_eff")
    ax3.set_title("Fee overhead vs anonymity")
    ax3.set_xlabel("Fee overhead %")
    ax3.set_ylabel("2^H")

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(beta_grid, succ_rates, marker="o", color="#7BDFF2")
    ax4.set_title("Mean success vs β")
    ax4.set_xlabel("β")
    ax4.set_ylabel("Success rate")

    ax5 = fig.add_subplot(gs[2, 1])
    ranks = [p.path_rank_1based for p in results.pcar_logs if p.path_rank_1based > 0]
    ax5.hist(ranks, bins=max(8, int(np.max(ranks)) if ranks else 8), color="#CBA6F7", alpha=0.9)
    ax5.set_title("PCAR rank histogram")
    ax5.set_xlabel("Rank (1 = cheapest)")
    ax5.set_ylabel("Count")

    dashboard_path = results_dir / "dashboard.png"
    fig.savefig(dashboard_path, dpi=175)
    plt.close(fig)

    _ = entropy_paths, trade_paths  # exported artifacts live beside dashboard
    return dashboard_path
