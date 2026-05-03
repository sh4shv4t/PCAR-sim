"""
Privacy penalty Π(P) for PCAR-sim: Π(P) = w1·π_pop + w2·π_cent + w3·π_uniq.

Note on correlation: π_pop and π_cent tend to be positively correlated in LN-like
graphs — popular routes often traverse high-betweenness hubs, so repetitive path
choices co-occur with higher intermediate centrality. Treat Π as a heuristic probe,
not an orthogonal decomposition.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from config import PCARConfig

from core.routing import EdgeStep, _nodes_from_edges


def final_two_edges_signature(edges: list[EdgeStep]) -> tuple[EdgeStep, ...]:
    if len(edges) >= 2:
        return (edges[-2], edges[-1])
    if len(edges) == 1:
        return (edges[-1],)
    return tuple()


def _min_max(vals: list[float]) -> list[float]:
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    span = hi - lo if hi > lo else 1.0
    return [(v - lo) / span for v in vals]


class PrivacyEngine:
    """Sender-local popularity tracking + composite Π for candidate paths."""

    def __init__(self, cfg: PCARConfig):
        self.cfg = cfg
        self._buf: deque[list[EdgeStep]] = deque(maxlen=int(cfg.T_paths))

    def record_choice(self, edges: list[EdgeStep]) -> None:
        self._buf.append(list(edges))

    def count_popularity_collision(self, edges: list[EdgeStep]) -> int:
        sig = final_two_edges_signature(edges)
        if not sig:
            return 0
        n = 0
        for prev in self._buf:
            if final_two_edges_signature(prev) == sig:
                n += 1
        return n

    def compute_pi_components(
        self,
        candidate_paths: list[list[EdgeStep]],
        sender: Any,
        receiver: Any,
        B_cache: dict[Any, float],
    ) -> list[float]:
        k_set = len(candidate_paths)
        pops_raw: list[float] = []
        cents_raw: list[float] = []
        uniq_raw: list[float] = []
        for path in candidate_paths:
            nodes = _nodes_from_edges(path)
            intermediates = nodes[1:-1]
            pc = float(sum(B_cache.get(v, 0.0) for v in intermediates) / max(len(intermediates), 1))

            n_hist = self.count_popularity_collision(path)
            pi_pop = 1.0 - math.exp(-self.cfg.mu * float(n_hist))

            pi_uniq = math.exp(-self.cfg.nu * float(k_set))

            pops_raw.append(pi_pop)
            cents_raw.append(pc)
            uniq_raw.append(pi_uniq)

        pops_n = _min_max(pops_raw)
        cents_n = _min_max(cents_raw)
        uniq_n = _min_max(uniq_raw)

        return [
            self.cfg.w1 * pops_n[i] + self.cfg.w2 * cents_n[i] + self.cfg.w3 * uniq_n[i]
            for i in range(len(candidate_paths))
        ]
