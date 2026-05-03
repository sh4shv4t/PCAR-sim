"""Spring-layout topology plot for PCAR-sim (centrality heatmap + exemplar routes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from core.graph import get_centrality_cache


def plot_network_demo(graph: Any, demo: Any | None, out_path: Path) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 9))

    ug = nx.Graph()
    ug.add_nodes_from(graph.nodes())
    for u, v, _k in graph.edges(keys=True):
        ug.add_edge(u, v)

    pos = nx.spring_layout(ug, seed=42, iterations=55)
    B = get_centrality_cache()
    nodes = list(graph.nodes())
    degs = dict(graph.degree())
    max_deg = max(degs.values()) if degs else 1
    cols = np.array([B.get(n, 0.0) for n in nodes], dtype=float)
    sizes = [120 + 260 * (degs[n] / max_deg) for n in nodes]

    nc = nx.draw_networkx_nodes(
        ug,
        pos,
        nodelist=nodes,
        node_color=cols,
        cmap="inferno",
        node_size=sizes,
        ax=ax,
    )
    nx.draw_networkx_edges(ug, pos, alpha=0.07, width=0.6, edge_color="white", ax=ax)

    def seg_draw(edge_list: list[tuple[Any, Any, Any]], color: str, lw: float, alpha: float) -> None:
        for u, v, _k in edge_list:
            if u not in pos or v not in pos:
                continue
            xu, yu = pos[u]
            xv, yv = pos[v]
            ax.plot([xu, xv], [yu, yv], color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")

    if demo is not None and getattr(demo, "candidate_paths", None):
        cmap = plt.get_cmap("tab10")
        for idx, path_edges in enumerate(demo.candidate_paths):
            c = cmap(idx % 10)
            seg_draw(path_edges, color=c, lw=1.8, alpha=0.45)

        seg_draw(getattr(demo, "baseline_edges", []) or [], color="#FF5C5C", lw=2.6, alpha=0.95)

        chosen = demo.candidate_paths[demo.pcar_index]
        seg_draw(chosen, color="#39FF14", lw=3.4, alpha=1.0)

    cb = fig.colorbar(nc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Normalised betweenness B(v)")

    ax.set_title("Synthetic LN graph — candidates vs baseline (red) vs PCAR choice (green)")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
