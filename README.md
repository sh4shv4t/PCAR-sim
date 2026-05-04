![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Simulation](https://img.shields.io/badge/Type-Research%20Simulation-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square)
![NetworkX](https://img.shields.io/badge/Graph-NetworkX-informational?style=flat-square)
![Privacy](https://img.shields.io/badge/Focus-Payment%20Privacy-purple?style=flat-square)

# PCAR-sim

### Probabilistic Cost-Aware Routing Simulator for the Lightning Network

A research-grade simulation demonstrating how probabilistic path sampling improves payment privacy over deterministic routing in Bitcoin's Lightning Network.

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Interpretation](#results--interpretation)
- [References](#references)

## Overview

**PCAR-sim** is a Monte Carlo sandbox for Probabilistic Cost-Aware Routing (PCAR): it builds synthetic Lightning-style graphs and compares a deterministic cheapest-path baseline to stochastic routing over a menu of feasible routes. Lightning wallets that always minimise advertised cost replay the same optimisation on public gossip data, collapsing the anonymity set to roughly one or two paths regardless of Sphinx-style onion forwarding. PCAR replaces that peaked choice with Boltzmann (softmax) sampling over \(k\) Yen shortest candidates so probability mass spreads across comparable paths at some fee and delay cost. The simulator reports entropy of the softmax distribution, anonymity-set surrogates, fee overhead, success rates, and how often a cheap-path predictor matches the sampled route.

## The Problem

Onion routing hides hop-by-hop payloads, but deterministic least-cost routing on a largely public channel graph yields a sharply peaked posterior over paths once the adversary aligns with the wallet’s objective. Predictable paths strip routing anonymity without breaking ciphertext—the adversary’s route guess is pinned before timing or traffic analysis contributes much. PCAR-sim contrasts that baseline with stochastic routing using the same topology and adversary surrogate so entropy and anonymity-set metrics move when sampling does.

## The Solution

Candidates are \(k\) shortest loopless paths under a surrogate cost blending fee, delay, liquidity-style risk, and a composite privacy penalty \(\Pi(P)\) over popularity, hub centrality, and path uniqueness (weights \(w_1,w_2,w_3\)). Paths are drawn with softmax sampling, \(\Pr[P_i] \propto \exp(-\beta\, C(P_i))\), where \(\beta\) trades concentration on cheap routes against dispersion. Optional \(\beta\) jitter, guard relays, rolling dynamic \(\beta\), and multipath splitting extend experiments.

## Architecture

Repository layout follows the modular core plus `visualise/` assets below.

```
.
├── README.md             # This file (repository root)
├── main.py               # CLI entry point
├── LICENSE
├── config.py             # All hyperparameters
├── requirements.txt
├── core/
│   ├── graph.py          # Barabási–Albert LN graph generation + centrality cache
│   ├── routing.py        # Baseline (Dijkstra) + PCAR (Yen's k-SP + softmax)
│   ├── privacy.py        # Privacy penalty Π(P): popularity, centrality, uniqueness
│   ├── simulate.py       # Simulation loop, dynamic-β scheduler, PMP-PCAR
│   └── metrics.py        # Shannon entropy, anonymity set size, fee overhead, adversary accuracy
└── visualise/
    ├── network_plot.py   # Graph + candidate path visualisation
    ├── entropy_plot.py   # Entropy curves, anonymity set bars, adversary accuracy
    ├── tradeoff_plot.py  # Fee-vs-privacy scatter, success rate, path rank histogram
    └── dashboard.py      # Combined 3×2 multi-panel dashboard → results/dashboard.png
```

## Key Concepts

| Concept | What it means in PCAR-sim |
|--------|---------------------------|
| Shannon Entropy *H* | Bits of uncertainty an adversary has about the chosen path |
| *β* (inverse temperature) | Higher *β* = more deterministic; lower *β* = more random |
| Anonymity Set Size | Estimated as *2^H* — effective number of paths the adversary must consider |
| Risk(P) | Bayesian per-hop liquidity failure probability |
| Guard Node | First hop constrained to low-centrality, long-lived node |

## Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | `10` | Number of Yen candidate paths per payment. |
| `beta` | `2.0` | Base softmax inverse temperature before jitter / scheduling. |
| `alpha` | `0.3` | Weight on normalised fee term in composite cost. |
| `gamma` | `0.3` | Weight on normalised delay term (includes optional padding). |
| `delta` | `0.2` | Weight on normalised Pickhardt–Richter-style risk. |
| `lambda_` | `0.2` | Weight on privacy penalty \(\Pi(P)\). |
| `w1` | `0.4` | Mix weight on popularity component \(\pi_{\mathrm{pop}}\). |
| `w2` | `0.4` | Mix weight on centrality component \(\pi_{\mathrm{cent}}\). |
| `w3` | `0.2` | Mix weight on uniqueness component \(\pi_{\mathrm{uniq}}\). |
| `mu` | `0.1` | Popularity saturation scale in \(1 - e^{-\mu n}\). |
| `nu` | `0.2` | Uniqueness decay: multiplier \(\nu\) in \(\exp(-\nu \cdot k)\) with \(k\) = number of candidate paths. |
| `T_paths` | `100` | Sender-local circular buffer length for popularity tracking. |
| `delta_max` | `10` | Upper bound (blocks) for uniform timelock padding per hop. |
| `B_thresh` | `0.7` | Maximum allowed normalised betweenness on first-hop relay when guards apply. |
| `U_min` | `0.9` | Minimum normalised channel age (`age / max_age`) on first-hop edge when guards apply. |
| `beta_noise_delta` | `0.1` | Half-width of uniform \(\beta\) jitter before sampling. |
| `tau_lo` | `0.6` | Rolling success threshold triggering \(\beta\) increase. |
| `tau_hi` | `0.85` | Rolling success threshold triggering \(\beta\) decrease. |
| `delta_beta_schedule` | `0.2` | Step size for dynamic \(\beta\) adjustments. |
| `beta_min` | `0.5` | Clamp floor for \(\beta\). |
| `beta_max` | `5.0` | Clamp ceiling for \(\beta\). |
| `rolling_window` | `100` | Window length for rolling success-rate estimation. |
| `n_payments` | `500` | Monte Carlo payment count per simulation episode. |
| `n_nodes` | `100` | Synthetic graph order (Barabási–Albert backbone). |
| `seed` | `42` | Random seed for reproducibility. |
| `multipath_threshold_sat` | `250000` | Amount threshold (satoshis) triggering optional PMP-PCAR splits. |
| `fee_ref_msat` | `5000000` | Scale reference for fee surrogate in routing discovery. |
| `delay_ref_blocks` | `500` | Scale reference for delay surrogate in routing discovery. |

## Installation

```bash
git clone https://github.com/sh4shv4t/PCAR-sim.git
cd PCAR-sim
pip install -r requirements.txt
python main.py
```

## Usage

```bash
# Run with all defaults (recommended starting point)
python main.py
```

```bash
# Run with custom beta and payment count
python main.py --beta 2.0 --n-payments 500
```

```bash
# Enable dynamic-β scheduling and multipath splitting
python main.py --dynamic-beta --multipath
```

```bash
# Full custom run saving results to CSV
python main.py --beta 1.5 --k 10 --n-payments 1000 --n-nodes 100 --seed 42 --save-results
```

## Results & Interpretation

The table below shows analytical upper-bound estimates from the paper. Run `python main.py --beta-sweep` to generate an empirical version on the synthetic graph.

| \(\beta\) | \(H\) (bits) | Est. anonymity set | Fee overhead |
|-----------|--------------|-------------------|--------------|
| \(\infty\) (LN today) | 0.00 | ≈ 1 | ~ 20 % |
| 5.0 | 0.71 | ≈ 4 | ~ 4 % |
| 2.0 | 1.58 | ≈ 10 | ~ 12 % |
| 1.0 | 1.97 | ≈ 20 | ~ 21 % |
| 0.5 | 2.18 | ≈ 35 | ~ 38 % |

## Limitations

- Synthetic Barabási–Albert graphs differ from labelled mainnet topology, capacities, and liquidity evolution.
- Entropy and \(2^{H}\) are exploratory indicators, not formal anonymity guarantees under joint or adaptive attackers.
- End-to-end threats such as passive global observers, ISP correlation, Bitcoin-layer clustering, or wallet fingerprinting are not modeled.

## References

1. G. Kappos et al., empirical privacy analysis of the Lightning Network, *Financial Cryptography (FC)*, 2021.
2. S. Kumble et al., routing-anonymity collapse under deterministic policies, *ARES*, 2021.
3. J. Feigenbaum, A. Johnson, and P. Syverson, "Probabilistic analysis of onion routing in a black-box model," *ACM Transactions on Information and System Security (TISSEC)*, vol. 15, no. 3, Nov. 2012.
4. G. Malavolta et al., Anonymous Multi-Hop Locks, *NDSS*, 2019.
5. R. Pickhardt & S. Richter, optimal reliable payment flows / liquidity diagnostics, *arXiv*, 2021.
6. E. Rohrer \& F. Tschorsch, HTLC timelock inference and privacy implications, *AFT*, 2020.

## License

MIT License.
