#!/usr/bin/env python3
"""CLI entry point for PCAR-sim."""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        pass

import matplotlib

matplotlib.use("Agg")

import argparse
from pathlib import Path

from config import DEFAULT_CONFIG, merge_cli_into_config
from core.simulate import run_simulation
from visualise.beta_sweep import build_beta_sweep_figure
from visualise.dashboard import build_dashboard

BETA_SWEEP_VALUES = (5.0, 2.0, 1.0, 0.5, 0.1)


def build_parser() -> argparse.ArgumentParser:
    d = DEFAULT_CONFIG
    p = argparse.ArgumentParser(
        prog="PCAR-sim",
        description=(
            "PCAR-sim: Probabilistic Cost-Aware Routing (PCAR) Monte Carlo simulator "
            "for synthetic Lightning-style liquidity networks."
        ),
    )
    p.add_argument("--beta", type=float, default=d.beta, help="Softmax inverse temperature β.")
    p.add_argument("--k", type=int, default=d.k, help="Number of Yen candidate paths.")
    p.add_argument("--n-payments", type=int, default=d.n_payments, help="Payments per simulation episode.")
    p.add_argument("--n-nodes", type=int, default=d.n_nodes, help="Barabási–Albert graph order.")
    p.add_argument("--seed", type=int, default=d.seed, help="RNG seed for reproducibility.")
    p.add_argument("--alpha", type=float, default=d.alpha, help="Fee weight in composite cost.")
    p.add_argument("--gamma", type=float, default=d.gamma, help="Delay weight.")
    p.add_argument("--delta", type=float, default=d.delta, help="Pickhardt–Richter risk weight.")
    p.add_argument("--lambda", dest="lambda_", type=float, default=d.lambda_, help="Privacy penalty weight λ.")
    p.add_argument("--w1", type=float, default=d.w1, help="π_pop weight.")
    p.add_argument("--w2", type=float, default=d.w2, help="π_cent weight.")
    p.add_argument("--w3", type=float, default=d.w3, help="π_uniq weight.")
    p.add_argument("--mu", type=float, default=d.mu, help="Popularity saturation µ.")
    p.add_argument("--nu", type=float, default=d.nu, help="Uniqueness decay ν.")
    p.add_argument("--T-paths", dest="T_paths", type=int, default=d.T_paths, help="Sender history length T.")
    p.add_argument("--delta-max", dest="delta_max", type=int, default=d.delta_max, help="Timelock padding upper bound.")
    p.add_argument("--B-thresh", dest="B_thresh", type=float, default=d.B_thresh, help="Guard centrality threshold.")
    p.add_argument("--U-min", dest="U_min", type=float, default=d.U_min, help="Minimum normalised channel age for guards.")
    p.add_argument(
        "--beta-noise-delta",
        dest="beta_noise_delta",
        type=float,
        default=d.beta_noise_delta,
        help="Uniform noise half-width on β before sampling.",
    )
    p.add_argument("--multipath", action="store_true", help="Enable probabilistic multi-path (PMP-PCAR) splits.")
    p.add_argument("--dynamic-beta", action="store_true", help="Rolling-success scheduler on β.")
    p.add_argument("--save-results", action="store_true", help="Persist paired CSV logs under results/metrics.csv.")
    p.add_argument(
        "--beta-sweep",
        action="store_true",
        help="Run β ∈ {5,2,1,0.5,0.1} sequentially and print a combined table plus results/beta_sweep.png.",
    )
    return p


def _print_beta_sweep_table(
    *,
    k: int,
    n_payments: int,
    seed: int,
    baseline_summary: dict,
    sweep_rows: list[tuple[float, dict]],
) -> None:
    w = 88
    print("\n" + "=" * w)
    print(f"β Sweep Results (k={k}, n={n_payments} payments, seed={seed})")
    print("=" * w)
    bs = baseline_summary
    b_succ = bs.get("success_rate", float("nan"))
    print(
        f"{'Baseline (deterministic LN)':<26} "
        f"{0.0:>10.4f} "
        f"{1.0:>10.4f} "
        f"{0.0:>16.2f}% "
        f"{1.0:>14.4f} "
        f"{b_succ:>14.4f}"
    )
    print("-" * w)
    print(
        f"{'β':<26} "
        f"{'H (bits)':>10} "
        f"{'Anon Set':>10} "
        f"{'Fee Overhead %':>16} "
        f"{'Adversary Acc':>14} "
        f"{'Success Rate':>14}"
    )
    print("-" * w)
    for beta, s in sweep_rows:
        print(
            f"{beta:<26.1f} "
            f"{s.get('entropy_mean', float('nan')):>10.4f} "
            f"{s.get('anon_mean', float('nan')):>10.4f} "
            f"{s.get('fee_overhead_pct_mean', float('nan')):>15.2f}% "
            f"{s.get('adversary_accuracy', float('nan')):>14.4f} "
            f"{s.get('success_rate', float('nan')):>14.4f}"
        )
    print("=" * w + "\n")


def run_beta_sweep(args: argparse.Namespace) -> None:
    k = args.k
    n_pay = args.n_payments
    seed = args.seed
    sweep_rows: list[tuple[float, dict]] = []
    baseline_summary: dict | None = None

    for beta in BETA_SWEEP_VALUES:
        cfg = merge_cli_into_config(
            DEFAULT_CONFIG,
            beta=beta,
            k=args.k,
            n_payments=args.n_payments,
            n_nodes=args.n_nodes,
            seed=args.seed,
            alpha=args.alpha,
            gamma=args.gamma,
            delta=args.delta,
            lambda_=args.lambda_,
            w1=args.w1,
            w2=args.w2,
            w3=args.w3,
            mu=args.mu,
            nu=args.nu,
            T_paths=args.T_paths,
            delta_max=args.delta_max,
            B_thresh=args.B_thresh,
            U_min=args.U_min,
            beta_noise_delta=args.beta_noise_delta,
        )
        results = run_simulation(
            cfg,
            dynamic_beta=args.dynamic_beta,
            multipath=args.multipath,
            save_results_path=None,
            progress=False,
            print_summary=False,
            warn_guard=False,
        )
        if baseline_summary is None:
            baseline_summary = results.summary_block("Baseline", results.baseline_logs)
        psum = results.summary_block("PCAR", results.pcar_logs)
        sweep_rows.append((beta, psum))

    assert baseline_summary is not None
    _print_beta_sweep_table(k=k, n_payments=n_pay, seed=seed, baseline_summary=baseline_summary, sweep_rows=sweep_rows)

    betas = [b for b, _ in sweep_rows]
    ent = [s["entropy_mean"] for _, s in sweep_rows]
    anon = [s["anon_mean"] for _, s in sweep_rows]
    fee = [s["fee_overhead_pct_mean"] for _, s in sweep_rows]
    adv = [s["adversary_accuracy"] for _, s in sweep_rows]

    out = Path("results/beta_sweep.png")
    saved = build_beta_sweep_figure(betas, ent, anon, fee, adv, out_path=out)
    print(f"PCAR-sim: beta sweep figure saved to {saved.resolve()}")


def main() -> None:
    args = build_parser().parse_args()

    if args.beta_sweep:
        run_beta_sweep(args)
        return

    cfg = merge_cli_into_config(
        DEFAULT_CONFIG,
        beta=args.beta,
        k=args.k,
        n_payments=args.n_payments,
        n_nodes=args.n_nodes,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        delta=args.delta,
        lambda_=args.lambda_,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        mu=args.mu,
        nu=args.nu,
        T_paths=args.T_paths,
        delta_max=args.delta_max,
        B_thresh=args.B_thresh,
        U_min=args.U_min,
        beta_noise_delta=args.beta_noise_delta,
    )

    save_path = Path("results/metrics.csv") if args.save_results else None
    results = run_simulation(
        cfg,
        dynamic_beta=args.dynamic_beta,
        multipath=args.multipath,
        save_results_path=save_path,
        progress=True,
    )
    dash = build_dashboard(cfg, results, Path("results"))
    print(f"PCAR-sim: dashboard figure saved to {dash.resolve()}")


if __name__ == "__main__":
    main()
