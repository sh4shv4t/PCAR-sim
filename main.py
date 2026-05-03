#!/usr/bin/env python3
"""CLI entry point for PCAR-sim."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
from pathlib import Path

from config import DEFAULT_CONFIG, merge_cli_into_config
from core.simulate import run_simulation
from visualise.dashboard import build_dashboard


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
    return p


def main() -> None:
    args = build_parser().parse_args()
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
