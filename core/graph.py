"""Graph generation for PCAR-sim: synthetic Lightning-style topology via Barabási–Albert + directed multi-edges."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

# Module-level cache: node -> normalised betweenness B(v) ∈ [0, 1]
_betweenness_cache: dict[Any, float] = {}


def get_centrality_cache() -> dict[Any, float]:
    return _betweenness_cache.copy()


def refresh_centrality_cache(G: nx.MultiDiGraph) -> dict[Any, float]:
    """
    Recompute normalised betweenness centrality (Brandes) on an undirected simple
    projection of G (parallel directed edges collapsed). Values are min-max
    scaled into [0, 1] for routing guard thresholds.
    """
    global _betweenness_cache
    simple = nx.Graph()
    simple.add_nodes_from(G.nodes())
    for u, v, _k in G.edges(keys=True):
        simple.add_edge(u, v)
    bc = nx.betweenness_centrality(simple, normalized=True)
    if not bc:
        _betweenness_cache = {}
        return _betweenness_cache
    vals = np.array(list(bc.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo if hi > lo else 1.0
    _betweenness_cache = {n: float((bc[n] - lo) / span) for n in bc}
    return _betweenness_cache


def _sample_edge_attrs(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "base_fee": int(rng.integers(100, 1001)),
        "fee_rate": int(rng.integers(1, 501)),
        "cltv_delta": int(rng.integers(6, 145)),
        "capacity": int(rng.integers(100_000, 10_000_001)),
        "channel_age": int(rng.integers(1, 500_001)),
    }


def generate_graph(n_nodes: int, seed: int) -> nx.MultiDiGraph:
    """
    Build a directed multigraph (~80–120 nodes typical) from Barabási–Albert:
    undirected preferential attachment is duplicated into opposing directed edges,
    each with independent fee/capacity parameters.
    """
    if n_nodes < 8:
        raise ValueError("n_nodes must be at least 8 for robust BA attachment.")
    rng = np.random.default_rng(seed)
    # Preferential attachment parameter m (LN-like hubs)
    m = max(2, min(6, max(2, n_nodes // 18)))
    ba = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n_nodes))
    for u, v in ba.edges():
        Gu = _sample_edge_attrs(rng)
        G.add_edge(u, v, **Gu)
        G.add_edge(v, u, **_sample_edge_attrs(rng))
    refresh_centrality_cache(G)
    max_age = max(d.get("channel_age", 1) for _u, _v, _k, d in G.edges(keys=True, data=True))
    G.graph["max_channel_age"] = float(max_age)
    return G
