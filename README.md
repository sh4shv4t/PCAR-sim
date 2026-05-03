# PCAR-sim — Probabilistic Cost-Aware Routing for Lightning-style Networks

Research-grade Monte Carlo simulator for **Probabilistic Cost-Aware Routing (PCAR)** on synthetic Lightning Network (LN) topologies. It couples:

- **Deterministic baseline routing** (Dijkstra on an LN-style surrogate cost with fee-aware amount propagation),
- **PCAR** (Yen’s _k_ shortest loopless paths + Boltzmann sampling + privacy penalties),
- **Dynamic softmax temperature**, optional **PMP-PCAR multipath splitting**, and a **metrics/visualisation** stack tuned for experiments on Windows.

```
┌─────────────┐    ┌───────────────────────┐    ┌─────────────────────┐
│ config.py   │───▶│ core/simulate.py      │───▶│ visualise/*.py      │
│ hyperparams │    │ payments + scheduler  │    │ plots + dashboard   │
└─────────────┘    └──────────┬────────────┘    └─────────────────────┘
                              │
                   ┌──────────┴──────────┐
                   │ core/graph.py       │  BA → MultiDiGraph + B(v)
                   │ core/routing.py     │  Dijkstra, Yen, softmax
                   │ core/privacy.py     │  Π(P) popularity / centrality / uniq
                   │ core/metrics.py     │  entropy, overhead, adversary
                   └─────────────────────┘
```

## Installation 

```powershell
cd PCAR-sim
python -m pip install -r requirements.txt
python main.py
```

Outputs are written under `results/`, including `dashboard.png` and standalone figures (`entropy_lines.png`, `tradeoff_scatter.png`, …).

## Usage examples

```powershell
# Defaults: ~100 nodes, 500 payments, full dashboard
python main.py

python main.py --beta 2.5 --k 12 --n-payments 800 --n-nodes 110 --seed 7

python main.py --dynamic-beta --multipath --save-results

python main.py --lambda 0.25 --alpha 0.35 --gamma 0.25 --delta 0.15 `
               --B-thresh 0.65 --U-min 0.85 --delta-max 12
```

CLI flags override `config.py` without editing source. Boolean switches: `--multipath`, `--dynamic-beta`, `--save-results` (writes paired baseline/PCAR rows to `results/metrics.csv`).

## Key parameters

| Symbol / flag | Role |
|---------------|------|
| `k`, `--k` | Yen candidate count |
| `β`, `--beta` | Softmax inverse temperature (scheduled if `--dynamic-beta`) |
| `α, γ, δ`, `--alpha`, `--gamma`, `--delta` | Fee, delay (with optional hop padding), Pickhardt–Richter-style risk weights |
| `λ`, `--lambda` | Weight on composite privacy penalty Π(P) |
| `w1,w2,w3`, `--w1`… | Mix weights on π_pop, π_cent, π_uniq |
| `μ, ν`, `--mu`, `--nu` | Popularity saturation & uniqueness decay |
| `T`, `--T-paths` | Sender path-history buffer length |
| `Δ_max`, `--delta-max` | Uniform timelock padding per hop (PCAR cost) |
| `B_thresh`, `--B-thresh` | Guard: first-hop relay centrality ceiling |
| `U_min`, `--U-min` | Guard: minimum **normalised** channel age on first hop (`age / max_age`) |
| `δ_beta`, `--beta-noise-delta` | Uniform jitter on β before sampling |
| Rolling τ_lo/τ_hi | Dynamic β scheduler thresholds (`config.py`) |

## Sample output

After `python main.py`, inspect `results/dashboard.png` — a **3×2 dark-theme mosaic** with:

1. Spring-layout topology coloured by betweenness, exemplar candidate routes, baseline (red) vs PCAR choice (green).  
2. Shannon entropy trajectories for β ∈ {0.5, 1.0, 2.0, 5.0}.  
3. Rolling adversary prediction accuracy (deterministic LN vs observed PCAR path).  
4. Fee-overhead vs anonymity scatter (colour = β_eff).  
5. Mean success vs β (auxiliary sweeps reuse the same payment budget).  
6. Histogram of chosen path ranks.

Individual PNGs are saved beside the dashboard for inclusion in papers or slides.

## Model notes

- **Topology**: Undirected Barabási–Albert backbone expanded into opposing directed multi-edges with independent fees, capacities, and ages; betweenness is computed on an undirected simple projection then min–max normalised to **[0, 1]** (`refresh_centrality_cache`).
- **Pickhardt–Richter-style risk**: \(R(P) = 1 - \prod_i (1 - a_i/c_i)\) with forward fee-aware amounts.
- **Privacy Π(P)**: Popularity via recent path tails, mean intermediate centrality, and uniqueness term; components are min–max normalised across candidates before mixing (see `core/privacy.py` docstring on π_pop vs π_cent correlation).

## References

- Kappos et al., *A Practical Threat Analysis of Lightning Network Privacy* (routing observability & probing context).
- Kumble et al., *Probabilistic Routing Protocol for High-throughput Payment Channel Networks* (randomised path selection vocabulary).
- Malavolta et al., *Concurrency and Privacy with Payment-Channel Networks* (anonymous multi-hop locks / privacy goals).
- Pickhardt & Richter, *Liquidity in Lightning Network* (liquidity / uncertainty modelling motivating hop failure probabilities).
