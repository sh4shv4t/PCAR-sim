"""Routing for PCAR-sim: baseline deterministic paths vs probabilistic PCAR (Yen's k-shortest + softmax)."""

from __future__ import annotations

# [3] Feigenbaum, Johnson & Syverson (ACM TISSEC 2012) — not Danezis & Syverson
import heapq
import logging
import math
import random
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from config import PCARConfig

logger = logging.getLogger(__name__)

EdgeStep = tuple[Any, Any, Any]  # (u, v, key)


def _inf() -> float:
    return float("inf")


def hop_fee_msat(amount_msat: float, data: dict) -> float:
    return float(data["base_fee"]) + (float(data["fee_rate"]) / 1e6) * amount_msat


def forward_amount_msat(amount_msat: float, data: dict) -> float:
    return amount_msat + hop_fee_msat(amount_msat, data)


def edge_data(G: nx.MultiDiGraph, u, v, key) -> dict:
    return G.edges[u, v, key]


def surrogate_edge_cost(
    G: nx.MultiDiGraph,
    u,
    v,
    key,
    amount_msat_in: float,
    cfg: PCARConfig,
) -> float:
    d = edge_data(G, u, v, key)
    fee = hop_fee_msat(amount_msat_in, d)
    delay = float(d["cltv_delta"])
    amt_sat = amount_msat_in / 1000.0
    risk_surrogate = amt_sat / max(float(d["capacity"]), 1.0)
    return (
        cfg.alpha * (fee / cfg.fee_ref_msat)
        + cfg.gamma * (delay / cfg.delay_ref_blocks)
        + cfg.delta * risk_surrogate
    )


def path_total_fee_msat(G: nx.MultiDiGraph, edges: list[EdgeStep], amount_sat_start: float) -> float:
    amt_msat = amount_sat_start * 1000.0
    total = 0.0
    for u, v, key in edges:
        d = edge_data(G, u, v, key)
        f = hop_fee_msat(amt_msat, d)
        total += f
        amt_msat = forward_amount_msat(amt_msat, d)
    return total


def path_delay_sum(G: nx.MultiDiGraph, edges: list[EdgeStep], padding_blocks: list[float] | None = None) -> float:
    s = 0.0
    for i, (_u, v, key) in enumerate(edges):
        d = edge_data(G, _u, v, key)
        pad = padding_blocks[i] if padding_blocks is not None else 0.0
        s += float(d["cltv_delta"]) + pad
    return s


def path_pickhardt_risk(G: nx.MultiDiGraph, edges: list[EdgeStep], amount_sat_start: float) -> float:
    """Risk(P) = 1 - Π(1 - amount/capacity) along hops (amount propagation includes fees)."""
    amt_msat = amount_sat_start * 1000.0
    prod = 1.0
    for u, v, key in edges:
        d = edge_data(G, u, v, key)
        amt_sat = amt_msat / 1000.0
        cap = max(float(d["capacity"]), 1.0)
        frac = min(max(amt_sat / cap, 0.0), 1.0)
        prod *= 1.0 - frac
        amt_msat = forward_amount_msat(amt_msat, d)
    return 1.0 - prod


def dijkstra_shortest_path(
    G: nx.MultiDiGraph,
    source: Any,
    target: Any,
    amount_sat_start: float,
    cfg: PCARConfig,
    *,
    initial_amount_msat: float | None = None,
    removed_edges: set[EdgeStep] | None = None,
    removed_nodes: set[Any] | None = None,
) -> list[EdgeStep] | None:
    """Surrogate-cost shortest path with HTLC amount propagation."""
    removed_edges = removed_edges or set()
    removed_nodes = removed_nodes or set()
    dist: dict[Any, float] = {n: _inf() for n in G.nodes()}
    pred: dict[Any, EdgeStep | None] = {n: None for n in G.nodes()}
    amt_msat: dict[Any, float] = {n: 0.0 for n in G.nodes()}
    dist[source] = 0.0
    amt_msat[source] = (
        float(initial_amount_msat) if initial_amount_msat is not None else amount_sat_start * 1000.0
    )
    pq: list[tuple[float, Any]] = [(0.0, source)]
    visited: set[Any] = set()

    while pq:
        d_u, u = heapq.heappop(pq)
        if u in visited:
            continue
        if d_u > dist[u]:
            continue
        visited.add(u)
        if u == target:
            break
        if u in removed_nodes and u != source:
            continue
        amt_u = amt_msat[u]
        for v in G.successors(u):
            if v in removed_nodes and v != target:
                continue
            for key in G[u][v]:
                e = (u, v, key)
                if e in removed_edges:
                    continue
                inc = surrogate_edge_cost(G, u, v, key, amt_u, cfg)
                dv = d_u + inc
                if dv < dist[v]:
                    dist[v] = dv
                    pred[v] = e
                    amt_msat[v] = forward_amount_msat(amt_u, edge_data(G, u, v, key))
                    heapq.heappush(pq, (dv, v))

    if pred[target] is None:
        return None
    edges_rev: list[EdgeStep] = []
    cur = target
    while cur != source:
        e = pred[cur]
        if e is None:
            return None
        edges_rev.append(e)
        cur = e[0]
    return list(reversed(edges_rev))


def _nodes_from_edges(edges: list[EdgeStep]) -> list[Any]:
    if not edges:
        return []
    out = [edges[0][0]]
    for _u, v, _k in edges:
        out.append(v)
    return out


def yen_k_shortest_paths(
    G: nx.MultiDiGraph,
    source: Any,
    target: Any,
    k: int,
    amount_sat_start: float,
    cfg: PCARConfig,
) -> list[list[EdgeStep]]:
    """
    Yen's algorithm for k-shortest loopless paths (multi-digraph).
    Uses surrogate additive costs with amount-aware Dijkstra.
    """
    first = dijkstra_shortest_path(G, source, target, amount_sat_start, cfg)
    if first is None:
        return []
    A: list[list[EdgeStep]] = [first]
    def path_cost(edges: list[EdgeStep]) -> float:
        amt_msat = amount_sat_start * 1000.0
        total = 0.0
        for u, v, key in edges:
            total += surrogate_edge_cost(G, u, v, key, amt_msat, cfg)
            amt_msat = forward_amount_msat(amt_msat, edge_data(G, u, v, key))
        return total

    def path_nodes_key(edges: list[EdgeStep]) -> tuple[Any, ...]:
        return tuple(_nodes_from_edges(edges))

    existing_keys = {path_nodes_key(p) for p in A}

    for _rank in range(1, k):
        prev = A[-1]
        spur_prev_nodes = _nodes_from_edges(prev)
        B_candidates: list[tuple[float, list[EdgeStep]]] = []
        for spur_idx in range(len(prev)):
            root_edges = prev[:spur_idx]
            root_nodes = _nodes_from_edges(root_edges)
            spur_node = prev[spur_idx][0]
            amt_at_spur = amount_sat_start * 1000.0
            for u, v, key in root_edges:
                amt_at_spur = forward_amount_msat(amt_at_spur, edge_data(G, u, v, key))

            removed_edges: set[EdgeStep] = set()
            removed_nodes: set[Any] = set()
            for p in A:
                p_nodes = _nodes_from_edges(p)
                if len(p_nodes) <= spur_idx:
                    continue
                if p_nodes[: spur_idx + 1] == spur_prev_nodes[: spur_idx + 1]:
                    removed_edges.add(p[spur_idx])
            for n in root_nodes[:-1]:
                removed_nodes.add(n)

            spur_path = dijkstra_shortest_path(
                G,
                spur_node,
                target,
                amount_sat_start,
                cfg,
                initial_amount_msat=amt_at_spur,
                removed_edges=removed_edges,
                removed_nodes=removed_nodes,
            )
            if spur_path is None:
                continue
            full_edges = list(root_edges)
            if not spur_path or spur_path[0][0] != spur_node:
                continue
            combined = full_edges + spur_path
            nodes_combined = _nodes_from_edges(combined)
            if len(nodes_combined) != len(set(nodes_combined)):
                continue
            c = path_cost(combined)
            heapq.heappush(B_candidates, (c, combined))

        while B_candidates:
            c, cand = heapq.heappop(B_candidates)
            key_c = path_nodes_key(cand)
            if key_c not in existing_keys:
                A.append(cand)
                existing_keys.add(key_c)
                break
        else:
            break
    return A


def _normalise_vec(vals: list[float], eps: float = 1e-12) -> list[float]:
    hi = max(vals) if vals else 1.0
    lo = min(vals) if vals else 0.0
    span = hi - lo if hi > lo else 1.0
    return [(v - lo) / span for v in vals]


def apply_guard_filter(
    G: nx.MultiDiGraph,
    paths: list[list[EdgeStep]],
    B_cache: dict[Any, float],
    cfg: PCARConfig,
    max_age: float,
    *,
    log_relaxed: bool = True,
) -> tuple[list[list[EdgeStep]], bool]:
    """
    Keep paths whose first hop edge leads to intermediate node v with B(v) < B_thresh
    and normalised channel_age / max_age > U_min (guard against hub-like first relays).
    """
    if not paths:
        return [], False
    kept: list[list[EdgeStep]] = []
    for p in paths:
        if len(p) < 1:
            continue
        _u, v, key = p[0]
        d = edge_data(G, _u, v, key)
        age_n = float(d["channel_age"]) / max(max_age, 1.0)
        if B_cache.get(v, 0.0) < cfg.B_thresh and age_n > cfg.U_min:
            kept.append(p)
    if kept:
        return kept, False
    if log_relaxed:
        logger.warning(
            "Guard-node constraint yielded no paths; relaxing for this payment (%s candidates). "
            "B_thresh=%s U_min=%s",
            len(paths),
            cfg.B_thresh,
            cfg.U_min,
        )
    return paths, True


def softmax_sample(costs: list[float], beta_eff: float, rng: random.Random) -> tuple[int, list[float]]:
    if not costs:
        raise ValueError("empty costs")
    m = max(costs)
    exps = [math.exp(-beta_eff * (c - m)) for c in costs]
    z = sum(exps)
    probs = [e / z for e in exps]
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i, probs
    return len(costs) - 1, probs


@dataclass
class BaselineRouteResult:
    path_edges: list[EdgeStep]
    fee_msat: float
    delay_blocks: float
    risk: float


@dataclass
class PCARRouteResult:
    candidate_paths: list[list[EdgeStep]]
    sampled_index: int
    probs: list[float]
    costs: list[float]
    padded_delays: list[float]
    beta_effective: float
    guard_relaxed: bool


def route_baseline(
    G: nx.MultiDiGraph,
    s: Any,
    r: Any,
    amount_sat: float,
    cfg: PCARConfig,
) -> BaselineRouteResult | None:
    edges = dijkstra_shortest_path(G, s, r, amount_sat, cfg)
    if edges is None:
        return None
    fee = path_total_fee_msat(G, edges, amount_sat)
    delay = path_delay_sum(G, edges, None)
    risk = path_pickhardt_risk(G, edges, amount_sat)
    return BaselineRouteResult(path_edges=edges, fee_msat=fee, delay_blocks=delay, risk=risk)


def enumerate_pcar_candidates(
    G: nx.MultiDiGraph,
    s: Any,
    r: Any,
    amount_sat: float,
    cfg: PCARConfig,
    B_cache: dict[Any, float],
    *,
    log_guard: bool = True,
) -> tuple[list[list[EdgeStep]], bool]:
    k_paths = yen_k_shortest_paths(G, s, r, cfg.k, amount_sat, cfg)
    if not k_paths:
        return [], False
    max_age = float(G.graph.get("max_channel_age", 1.0))
    return apply_guard_filter(G, k_paths, B_cache, cfg, max_age, log_relaxed=log_guard)


def pcar_softmax_from_candidates(
    G: nx.MultiDiGraph,
    k_paths: list[list[EdgeStep]],
    amount_sat: float,
    cfg: PCARConfig,
    privacy_penalties: list[float],
    rng: random.Random,
    np_rng: np.random.Generator,
    *,
    guard_relaxed: bool,
) -> PCARRouteResult | None:
    if not k_paths:
        return None
    fees = [path_total_fee_msat(G, p, amount_sat) for p in k_paths]
    risks = [path_pickhardt_risk(G, p, amount_sat) for p in k_paths]

    delays: list[float] = []
    for p in k_paths:
        pads = [float(np_rng.uniform(0.0, float(cfg.delta_max))) for _ in p]
        delays.append(path_delay_sum(G, p, padding_blocks=pads))

    fee_n = _normalise_vec(fees)
    delay_n = _normalise_vec(delays)
    risk_n = _normalise_vec(risks)

    Pi = privacy_penalties[: len(k_paths)]
    if len(Pi) != len(k_paths):
        raise ValueError("privacy_penalties length mismatch")

    costs = [
        cfg.alpha * fee_n[i]
        + cfg.gamma * delay_n[i]
        + cfg.delta * risk_n[i]
        + cfg.lambda_ * Pi[i]
        for i in range(len(k_paths))
    ]

    beta_eff = cfg.beta + rng.uniform(-cfg.beta_noise_delta, cfg.beta_noise_delta)
    beta_eff = min(cfg.beta_max, max(cfg.beta_min, beta_eff))

    idx, probs = softmax_sample(costs, beta_eff, rng)

    return PCARRouteResult(
        candidate_paths=k_paths,
        sampled_index=idx,
        probs=probs,
        costs=costs,
        padded_delays=[delays[idx]],
        beta_effective=beta_eff,
        guard_relaxed=guard_relaxed,
    )


def route_pcar(
    G: nx.MultiDiGraph,
    s: Any,
    r: Any,
    amount_sat: float,
    cfg: PCARConfig,
    B_cache: dict[Any, float],
    privacy_penalties: list[float],
    rng: random.Random,
    np_rng: np.random.Generator,
    *,
    log_guard: bool = True,
) -> PCARRouteResult | None:
    k_paths, guard_relaxed = enumerate_pcar_candidates(G, s, r, amount_sat, cfg, B_cache, log_guard=log_guard)
    return pcar_softmax_from_candidates(
        G,
        k_paths,
        amount_sat,
        cfg,
        privacy_penalties,
        rng,
        np_rng,
        guard_relaxed=guard_relaxed,
    )


def nodes_tuple_from_edges(edges: list[EdgeStep]) -> tuple[Any, ...]:
    return tuple(_nodes_from_edges(edges))
