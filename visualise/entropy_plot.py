"""Entropy trajectories, anonymity bars, and adversary dynamics for PCAR-sim."""

from __future__ import annotations

# [3] Feigenbaum, Johnson & Syverson (ACM TISSEC 2012) — not Danezis & Syverson
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

from config import PCARConfig, merge_cli_into_config
from core.metrics import SimulationResults
from core.simulate import run_simulation


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x
    out = np.empty_like(x, dtype=float)
    for i in range(x.size):
        lo = max(0, i - window + 1)
        out[i] = float(np.mean(x[lo : i + 1]))
    return out


def plot_entropy_suite(
    cfg_template: PCARConfig,
    main_results: SimulationResults,
    out_dir: Path,
    *,
    betas: list[float] | None = None,
) -> tuple[dict[str, Path], dict[float, np.ndarray]]:
    plt.style.use("dark_background")
    betas = betas or [0.5, 1.0, 2.0, 5.0]
    fig_a, ax_a = plt.subplots(figsize=(10, 5))

    entropy_curves: dict[float, np.ndarray] = {}
    anon_per_beta: list[float] = []
    anon_std: list[float] = []

    for b in betas:
        c = merge_cli_into_config(cfg_template, beta=b)
        sub = run_simulation(
            c,
            dynamic_beta=False,
            multipath=False,
            progress=False,
            save_results_path=None,
            print_summary=False,
            warn_guard=False,
        )
        ent = np.array([p.entropy_bits for p in sub.pcar_logs], dtype=float)
        entropy_curves[b] = ent
        ax_a.plot(np.arange(1, ent.size + 1), ent, linewidth=1.4, label=f"β={b}")

        ano = np.array([p.anonymity_size for p in sub.pcar_logs], dtype=float)
        anon_per_beta.append(float(np.mean(ano)))
        anon_std.append(float(sem(ano, ddof=1)) if ano.size > 1 else 0.0)

    ax_a.set_title("Shannon entropy H(β, k) across payments")
    ax_a.set_xlabel("Payment index")
    ax_a.set_ylabel("Entropy (bits)")
    ax_a.legend(frameon=False)
    fig_a.tight_layout()
    p_lines = out_dir / "entropy_lines.png"
    fig_a.savefig(p_lines, dpi=160)
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(7, 4))
    xpos = np.arange(len(betas))
    ax_b.bar(xpos, anon_per_beta, yerr=anon_std, capsize=6, color="#6CCFF6", alpha=0.85)
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels([str(b) for b in betas])
    ax_b.set_xlabel("β")
    ax_b.set_ylabel("Estimated anonymity size 𝔼[2^H] ± std")
    ax_b.set_title("Anonymity set scaling with inverse temperature")
    fig_b.tight_layout()
    p_bar = out_dir / "entropy_anon_bar.png"
    fig_b.savefig(p_bar, dpi=160)
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(10, 5))
    adv_b = np.array([1.0 if x.adversary_correct else 0.0 for x in main_results.baseline_logs], dtype=float)
    adv_p = np.array([1.0 if x.adversary_correct else 0.0 for x in main_results.pcar_logs], dtype=float)
    idx = np.arange(1, adv_b.size + 1)
    ax_c.plot(idx, rolling_mean(adv_b, 40), label="Baseline routing — rolling accuracy", color="#FF935C")
    ax_c.plot(idx, rolling_mean(adv_p, 40), label="PCAR — rolling accuracy", color="#8CF078")
    ax_c.set_xlabel("Payment index")
    ax_c.set_ylabel("Rolling adversary accuracy")
    ax_c.set_title("Deterministic LN adversary vs observed PCAR traces")
    ax_c.legend(frameon=False)
    fig_c.tight_layout()
    p_adv = out_dir / "entropy_adversary.png"
    fig_c.savefig(p_adv, dpi=160)
    plt.close(fig_c)

    paths = {"lines": p_lines, "anon_bar": p_bar, "adversary": p_adv}
    return paths, entropy_curves
