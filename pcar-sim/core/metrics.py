"""Metrics recording for PCAR-sim (entropy, fees, adversary accuracy, summaries)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def shannon_entropy_bits(probs: list[float]) -> float:
    h = 0.0
    for p in probs:
        if p <= 0.0:
            continue
        h -= float(p) * math.log(float(p), 2)
    return h


def summarize(arr: np.ndarray) -> tuple[float, float, float]:
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.median(arr)), float(np.quantile(arr, 0.9))


@dataclass
class PaymentLog:
    payment_idx: int
    sender: Any
    receiver: Any
    amount_sat: float
    entropy_bits: float
    anonymity_size: float
    fee_sat: float
    fee_overhead_pct_vs_baseline: float
    path_rank_1based: int
    hops: int
    success: bool
    retries: int
    adversary_correct: bool
    beta_effective: float | None = None
    rolling_success_rate: float | None = None
    multipath_shards: int = 1


@dataclass
class SimulationResults:
    baseline_logs: list[PaymentLog] = field(default_factory=list)
    pcar_logs: list[PaymentLog] = field(default_factory=list)
    demo: Any | None = None
    graph: Any | None = None

    def append_pair(self, b: PaymentLog, p: PaymentLog) -> None:
        self.baseline_logs.append(b)
        self.pcar_logs.append(p)

    def summary_block(self, label: str, logs: list[PaymentLog]) -> dict[str, Any]:
        if not logs:
            return {}
        ent = np.array([x.entropy_bits for x in logs], dtype=float)
        ano = np.array([x.anonymity_size for x in logs], dtype=float)
        fee_over = np.array([x.fee_overhead_pct_vs_baseline for x in logs], dtype=float)
        rank = np.array([float(x.path_rank_1based) for x in logs], dtype=float)
        hops = np.array([float(x.hops) for x in logs], dtype=float)
        succ = np.array([1.0 if x.success else 0.0 for x in logs], dtype=float)
        adv = np.array([1.0 if x.adversary_correct else 0.0 for x in logs], dtype=float)

        em, e_med, e90 = summarize(ent)
        am, a_med, a90 = summarize(ano)
        fm, f_med, f90 = summarize(fee_over)
        rm, r_med, r90 = summarize(rank)
        hm, h_med, h90 = summarize(hops)
        overall_succ = float(np.mean(succ))
        overall_adv = float(np.mean(adv))

        return {
            "label": label,
            "entropy_mean": em,
            "entropy_median": e_med,
            "entropy_p90": e90,
            "anon_mean": am,
            "anon_median": a_med,
            "anon_p90": a90,
            "fee_overhead_pct_mean": fm,
            "fee_overhead_pct_median": f_med,
            "fee_overhead_pct_p90": f90,
            "rank_mean": rm,
            "rank_median": r_med,
            "rank_p90": r90,
            "hops_mean": hm,
            "hops_median": h_med,
            "hops_p90": h90,
            "success_rate": overall_succ,
            "adversary_accuracy": overall_adv,
            "payments": len(logs),
        }
