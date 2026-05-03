"""Privacy–fee tradeoffs and β sensitivity plots for PCAR-sim."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import PCARConfig, merge_cli_into_config
from core.metrics import SimulationResults
from core.simulate import run_simulation


def plot_tradeoff_suite(
    cfg_template: PCARConfig,
    main_results: SimulationResults,
    out_dir: Path,
) -> tuple[dict[str, Path], np.ndarray, np.ndarray]:
    plt.style.use("dark_background")
    out_dir.mkdir(parents=True, exist_ok=True)

    fees = np.array([p.fee_overhead_pct_vs_baseline for p in main_results.pcar_logs], dtype=float)
    ano = np.array([p.anonymity_size for p in main_results.pcar_logs], dtype=float)
    betas_eff = np.array(
        [p.beta_effective if p.beta_effective is not None else cfg_template.beta for p in main_results.pcar_logs],
        dtype=float,
    )

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sc = ax1.scatter(fees, ano, c=betas_eff, cmap="plasma", s=22, alpha=0.85)
    cb = fig1.colorbar(sc, ax=ax1)
    cb.set_label("β_eff")
    ax1.set_xlabel("Fee overhead % vs deterministic cheapest route")
    ax1.set_ylabel("Estimated anonymity size 2^H")
    ax1.set_title("PCAR privacy–liquidity tradeoff cloud")
    fig1.tight_layout()
    p_trade = out_dir / "tradeoff_scatter.png"
    fig1.savefig(p_trade, dpi=160)
    plt.close(fig1)

    grid = np.linspace(cfg_template.beta_min, cfg_template.beta_max, 10)
    succ_rates: list[float] = []
    for b in grid:
        c = merge_cli_into_config(cfg_template, beta=float(b))
        sub = run_simulation(
            c,
            dynamic_beta=False,
            multipath=False,
            progress=False,
            save_results_path=None,
            print_summary=False,
            warn_guard=False,
        )
        succ_rates.append(float(np.mean([1.0 if x.success else 0.0 for x in sub.pcar_logs])))

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(grid, succ_rates, marker="o", color="#7BDFF2")
    ax2.set_xlabel("β")
    ax2.set_ylabel("Mean payment success rate")
    ax2.set_title("Liquidity-limited success sensitivity to softmax temperature")
    fig2.tight_layout()
    p_beta_succ = out_dir / "tradeoff_beta_success.png"
    fig2.savefig(p_beta_succ, dpi=160)
    plt.close(fig2)

    ranks = [p.path_rank_1based for p in main_results.pcar_logs if p.path_rank_1based > 0]
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(ranks, bins=max(10, int(np.max(ranks)) if ranks else 10), color="#CBA6F7", alpha=0.9)
    ax3.set_xlabel("Chosen candidate rank (1 = cheapest fee)")
    ax3.set_ylabel("Count")
    ax3.set_title("PCAR sampling distribution over Yen-ranked paths")
    fig3.tight_layout()
    p_hist = out_dir / "tradeoff_rank_hist.png"
    fig3.savefig(p_hist, dpi=160)
    plt.close(fig3)

    paths = {"scatter": p_trade, "beta_success": p_beta_succ, "rank_hist": p_hist}
    return paths, grid, np.array(succ_rates, dtype=float)
