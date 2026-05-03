"""Monte Carlo payment engine for PCAR-sim: baseline vs PCAR with dynamic β and optional multipath."""

from __future__ import annotations

import csv
import logging
import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from tqdm import tqdm

from config import PCARConfig
from core.graph import generate_graph, refresh_centrality_cache
from core.metrics import PaymentLog, SimulationResults, shannon_entropy_bits
from core.privacy import PrivacyEngine
from core.routing import (
    EdgeStep,
    edge_data,
    enumerate_pcar_candidates,
    forward_amount_msat,
    path_total_fee_msat,
    pcar_softmax_from_candidates,
    route_baseline,
    nodes_tuple_from_edges,
)

logger = logging.getLogger(__name__)


def liquidity_payment_success(G: nx.MultiDiGraph, edges: list[EdgeStep], amount_sat: float, rng: random.Random) -> bool:
    amt_msat = amount_sat * 1000.0
    for u, v, key in edges:
        d = edge_data(G, u, v, key)
        amt_sat_hop = amt_msat / 1000.0
        cap = max(float(d["capacity"]), 1.0)
        p_succ = max(0.0, min(1.0, 1.0 - amt_sat_hop / cap))
        if rng.random() > p_succ:
            return False
        amt_msat = forward_amount_msat(amt_msat, d)
    return True


def fee_rank_1based(fees: list[float]) -> list[int]:
    order = sorted(range(len(fees)), key=lambda i: fees[i])
    ranks = [0] * len(fees)
    for pos, idx in enumerate(order):
        ranks[idx] = pos + 1
    return ranks


def random_partition_amount(total_sat: float, n_parts: int, rng: random.Random) -> list[float]:
    if n_parts <= 1:
        return [total_sat]
    cuts = sorted(rng.random() for _ in range(n_parts - 1))
    pts = [0.0] + cuts + [1.0]
    return [(pts[i + 1] - pts[i]) * total_sat for i in range(n_parts)]


def sample_sr_pair(G: nx.MultiDiGraph, rng: random.Random, max_tries: int = 400) -> tuple[Any, Any]:
    nodes = list(G.nodes())
    for _ in range(max_tries):
        s = rng.choice(nodes)
        r = rng.choice(nodes)
        if s != r and nx.has_path(G, s, r):
            return s, r
    raise RuntimeError("Failed to sample a connected sender/receiver pair.")


@dataclass
class DemoSnapshot:
    sender: Any
    receiver: Any
    amount_sat: float
    candidate_paths: list[list[EdgeStep]]
    pcar_index: int
    baseline_edges: list[EdgeStep]


def run_simulation(
    cfg: PCARConfig,
    *,
    dynamic_beta: bool = False,
    multipath: bool = False,
    save_results_path: Path | None = None,
    progress: bool = True,
    print_summary: bool = True,
    warn_guard: bool = True,
) -> SimulationResults:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)

    G = generate_graph(cfg.n_nodes, cfg.seed)
    B_cache = refresh_centrality_cache(G)

    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    privacy = PrivacyEngine(cfg)
    results = SimulationResults()
    rolling_success: deque[float] = deque(maxlen=cfg.rolling_window)

    demo: DemoSnapshot | None = None

    bar = tqdm(range(cfg.n_payments), disable=not progress, desc="PCAR-sim payments", dynamic_ncols=True)

    mean_entropy_stream: deque[float] = deque(maxlen=50)

    for pay_i in bar:
        s, r = sample_sr_pair(G, rng)
        amount = float(rng.uniform(50_000, min(900_000, 3 * cfg.multipath_threshold_sat)))

        use_mp = multipath and amount >= cfg.multipath_threshold_sat
        shards = rng.randint(2, 5) if use_mp else 1
        amounts = random_partition_amount(amount, shards, rng) if use_mp else [amount]

        # --- Baseline (aggregate shards) ---
        b_fees_msat: list[float] = []
        b_success_all = True
        b_retries = 0
        b_paths: list[list[EdgeStep]] = []
        base_last = None
        for shard_amt in amounts:
            succ_shard = False
            while not succ_shard and b_retries < 25:
                base_last = route_baseline(G, s, r, shard_amt, cfg)
                if base_last is None:
                    b_retries += 1
                    continue
                if liquidity_payment_success(G, base_last.path_edges, shard_amt, rng):
                    succ_shard = True
                    b_paths.append(base_last.path_edges)
                    b_fees_msat.append(base_last.fee_msat)
                    break
                b_retries += 1
            if not succ_shard:
                b_success_all = False
                if base_last is not None:
                    b_paths.append(base_last.path_edges)
                    b_fees_msat.append(base_last.fee_msat)
                else:
                    b_paths.append([])
                    b_fees_msat.append(0.0)

        baseline_fee_total_msat = float(sum(b_fees_msat))

        # --- PCAR ---
        p_success_all = True
        p_retries = 0
        p_fees_msat: list[float] = []
        entropies: list[float] = []
        anon_sizes: list[float] = []
        ranks_chosen: list[int] = []
        hops_list: list[int] = []
        adversary_correct_any = True

        cands: list[list[EdgeStep]] = []
        guard_relaxed = False
        beta_eff_used: float | None = None

        for shard_amt in amounts:
            succ_shard = False
            while not succ_shard and p_retries < 25:
                cands, guard_relaxed = enumerate_pcar_candidates(
                    G,
                    s,
                    r,
                    shard_amt,
                    cfg,
                    B_cache,
                    log_guard=warn_guard,
                )
                if not cands:
                    p_retries += 1
                    continue
                Pi_list = privacy.compute_pi_components(cands, s, r, B_cache)
                pcar = pcar_softmax_from_candidates(
                    G,
                    cands,
                    shard_amt,
                    cfg,
                    Pi_list,
                    rng,
                    np_rng,
                    guard_relaxed=guard_relaxed,
                )
                if pcar is None:
                    p_retries += 1
                    continue

                fees = [path_total_fee_msat(G, p, shard_amt) for p in pcar.candidate_paths]
                ranks = fee_rank_1based(fees)
                chosen = pcar.candidate_paths[pcar.sampled_index]
                adv_base = route_baseline(G, s, r, shard_amt, cfg)
                adv_ok = adv_base is not None and nodes_tuple_from_edges(adv_base.path_edges) == nodes_tuple_from_edges(chosen)
                adversary_correct_any = adversary_correct_any and adv_ok

                if liquidity_payment_success(G, chosen, shard_amt, rng):
                    succ_shard = True
                    H = shannon_entropy_bits(pcar.probs)
                    entropies.append(H)
                    anon_sizes.append(float(2**H))
                    ranks_chosen.append(ranks[pcar.sampled_index])
                    hops_list.append(len(chosen))
                    p_fees_msat.append(path_total_fee_msat(G, chosen, shard_amt))
                    beta_eff_used = pcar.beta_effective

                    privacy.record_choice(chosen)

                    if demo is None or rng.random() < 0.05:
                        demo = DemoSnapshot(
                            sender=s,
                            receiver=r,
                            amount_sat=shard_amt,
                            candidate_paths=list(pcar.candidate_paths),
                            pcar_index=pcar.sampled_index,
                            baseline_edges=adv_base.path_edges if adv_base else [],
                        )
                    break

                p_retries += 1

            if not succ_shard:
                p_success_all = False
                if cands:
                    Pi_list = privacy.compute_pi_components(cands, s, r, B_cache)
                    pcar = pcar_softmax_from_candidates(
                        G,
                        cands,
                        shard_amt,
                        cfg,
                        Pi_list,
                        rng,
                        np_rng,
                        guard_relaxed=guard_relaxed,
                    )
                    if pcar:
                        fees = [path_total_fee_msat(G, p, shard_amt) for p in pcar.candidate_paths]
                        ranks = fee_rank_1based(fees)
                        chosen = pcar.candidate_paths[pcar.sampled_index]
                        H = shannon_entropy_bits(pcar.probs)
                        entropies.append(H)
                        anon_sizes.append(float(2**H))
                        ranks_chosen.append(ranks[pcar.sampled_index])
                        hops_list.append(len(chosen))
                        p_fees_msat.append(path_total_fee_msat(G, chosen, shard_amt))
                        beta_eff_used = pcar.beta_effective

        rolling_success.append(1.0 if p_success_all else 0.0)

        if dynamic_beta and rolling_success:
            p_roll = float(sum(rolling_success) / len(rolling_success))
            if p_roll < cfg.tau_lo:
                cfg.beta = min(cfg.beta_max, cfg.beta + cfg.delta_beta_schedule)
                logger.info(
                    "Dynamic β increase: rolling_success=%.3f β=%.3f",
                    p_roll,
                    cfg.beta,
                )
            elif p_roll > cfg.tau_hi:
                cfg.beta = max(cfg.beta_min, cfg.beta - cfg.delta_beta_schedule)
                logger.info(
                    "Dynamic β decrease: rolling_success=%.3f β=%.3f",
                    p_roll,
                    cfg.beta,
                )

        pcar_fee_total_msat = float(sum(p_fees_msat)) if p_fees_msat else float("nan")
        overhead_pct = (
            100.0 * (pcar_fee_total_msat - baseline_fee_total_msat) / baseline_fee_total_msat
            if baseline_fee_total_msat > 0 and not math.isnan(pcar_fee_total_msat)
            else 0.0
        )

        H_mean = float(np.mean(entropies)) if entropies else 0.0
        anon_mean = float(np.mean(anon_sizes)) if anon_sizes else 1.0
        rank_mean = float(np.mean(ranks_chosen)) if ranks_chosen else float("nan")
        hops_mean = float(np.mean(hops_list)) if hops_list else float("nan")

        mean_entropy_stream.append(H_mean)
        rolling_ent = float(np.mean(mean_entropy_stream)) if mean_entropy_stream else H_mean

        p_roll_disp = float(sum(rolling_success) / len(rolling_success)) if rolling_success else float("nan")

        b_log = PaymentLog(
            payment_idx=pay_i,
            sender=s,
            receiver=r,
            amount_sat=amount,
            entropy_bits=0.0,
            anonymity_size=1.0,
            fee_sat=baseline_fee_total_msat / 1000.0,
            fee_overhead_pct_vs_baseline=0.0,
            path_rank_1based=1,
            hops=int(round(sum(len(p) for p in b_paths) / max(len(b_paths), 1))),
            success=b_success_all,
            retries=b_retries,
            adversary_correct=True,
            beta_effective=None,
            rolling_success_rate=p_roll_disp,
            multipath_shards=shards,
        )

        p_log = PaymentLog(
            payment_idx=pay_i,
            sender=s,
            receiver=r,
            amount_sat=amount,
            entropy_bits=H_mean,
            anonymity_size=anon_mean,
            fee_sat=(pcar_fee_total_msat / 1000.0) if not math.isnan(pcar_fee_total_msat) else float("nan"),
            fee_overhead_pct_vs_baseline=overhead_pct,
            path_rank_1based=int(round(rank_mean)) if not math.isnan(rank_mean) else 0,
            hops=int(round(hops_mean)) if not math.isnan(hops_mean) else 0,
            success=p_success_all,
            retries=p_retries,
            adversary_correct=bool(adversary_correct_any),
            beta_effective=beta_eff_used,
            rolling_success_rate=p_roll_disp,
            multipath_shards=shards,
        )

        results.append_pair(b_log, p_log)

        bar.set_postfix(
            beta=f"{cfg.beta:.2f}",
            p_roll=f"{p_roll_disp:.2f}",
            H=f"{rolling_ent:.3f}",
            refresh=False,
        )

    results.demo = demo
    results.graph = G

    if print_summary:
        _print_summary_table(results)

    if save_results_path is not None:
        save_results_path.parent.mkdir(parents=True, exist_ok=True)
        _write_logs_csv(save_results_path, results)

    return results


def _write_logs_csv(path: Path, results: SimulationResults) -> None:
    cols = [
        "payment_idx",
        "mode",
        "entropy_bits",
        "anonymity_size",
        "fee_sat",
        "fee_overhead_pct_vs_baseline",
        "path_rank_1based",
        "hops",
        "success",
        "retries",
        "adversary_correct",
        "beta_effective",
        "rolling_success_rate",
        "multipath_shards",
        "amount_sat",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for b, p in zip(results.baseline_logs, results.pcar_logs):
            w.writerow(
                [
                    b.payment_idx,
                    "baseline",
                    b.entropy_bits,
                    b.anonymity_size,
                    b.fee_sat,
                    b.fee_overhead_pct_vs_baseline,
                    b.path_rank_1based,
                    b.hops,
                    int(b.success),
                    b.retries,
                    int(b.adversary_correct),
                    b.beta_effective if b.beta_effective is not None else "",
                    b.rolling_success_rate if b.rolling_success_rate is not None else "",
                    b.multipath_shards,
                    b.amount_sat,
                ]
            )
            w.writerow(
                [
                    p.payment_idx,
                    "pcar",
                    p.entropy_bits,
                    p.anonymity_size,
                    p.fee_sat,
                    p.fee_overhead_pct_vs_baseline,
                    p.path_rank_1based,
                    p.hops,
                    int(p.success),
                    p.retries,
                    int(p.adversary_correct),
                    p.beta_effective if p.beta_effective is not None else "",
                    p.rolling_success_rate if p.rolling_success_rate is not None else "",
                    p.multipath_shards,
                    p.amount_sat,
                ]
            )


def _print_summary_table(results: SimulationResults) -> None:
    bsum = results.summary_block("Baseline", results.baseline_logs)
    psum = results.summary_block("PCAR", results.pcar_logs)

    keys = [
        ("success_rate", "Success rate"),
        ("entropy_mean", "Entropy mean (bits)"),
        ("anon_mean", "Anon set mean"),
        ("fee_overhead_pct_mean", "Fee overhead % mean"),
        ("rank_mean", "Path rank mean"),
        ("hops_mean", "Hops mean"),
        ("adversary_accuracy", "Adversary accuracy"),
    ]

    print("\n" + "=" * 92)
    print(f"{'Metric':<34} {'Baseline':>26} {'PCAR':>26}")
    print("-" * 92)
    for k, label in keys:
        bv = bsum.get(k, float("nan"))
        pv = psum.get(k, float("nan"))
        print(f"{label:<34} {bv:>26.4f} {pv:>26.4f}")
    print("=" * 92 + "\n")
