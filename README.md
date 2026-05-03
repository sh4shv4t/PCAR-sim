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
- [The Solution: PCAR](#the-solution-pcar)
- [Architecture](#architecture)
- [Key Concepts](#key-concepts)
- [Simulation Parameters](#simulation-parameters)
- [Installation](#installation)
- [Usage](#usage)

## Overview

**PCAR-sim** is a reproducible Monte Carlo sandbox for *Probabilistic Cost-Aware Routing (PCAR)*: Boltzmann-weighted path sampling over candidate Lightning-style routes, with metrics for entropy, fee overhead, and a straw-man deterministic adversary.

Wallets that always minimise cost on a largely public channel graph make routes easy to reproduce; PCAR-sim contrasts that baseline with stochastic routing that spreads probability across acceptable paths so a guess-the-cheapest attacker is wrong more often—at some fee and delay cost.

## The Problem

Routing predictability is the core privacy issue: least-cost routing on a gossiped graph lets an adversary who approximates the same objective narrow the path to a singleton without breaking Sphinx. Onion routing hides hop identifiers from casual observers, but a sharply peaked path distribution collapses the anonymity set before ciphertext tells you much. PCAR-sim pairs a deterministic baseline with PCAR and flags when the cheapest path equals the sampled one—the stand-in for “routing privacy lost to algorithmic replay.”

## The Solution: PCAR

PCAR scores **\(k\)** Yen shortest paths using normalised fee, delay, Pickhardt–Richter-style liquidity risk, and a composite privacy penalty \(\Pi(P)\) (popularity, intermediate centrality, and path uniqueness), then samples with \(\Pr[P_i] \propto \exp(-\beta\, C(P_i))\). **\(\beta\)** is the privacy–cost knob: higher \(\beta\) behaves like today’s wallets; lower \(\beta\) flattens the distribution and buys entropy at higher fees or latency. Optional guard relays, \(\beta\) jitter, dynamic \(\beta\) from rolling success, and multipath splits extend the experiment; AMHL-style composition is referenced but not cryptographically simulated. Paper-style analytic magnitudes (e.g. ~\(2^H\) anonymous paths near \(\beta \approx 2\) at ~12 % fee overhead) are upper-bound sketches on stylised assumptions—compare against this simulator, not as mainnet guarantees.

## Architecture

This repository’s layout matches what GitHub renders on the project home page — **`README.md` sits at the repository root** alongside `main.py` and `config.py`.

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

Every knob lives in `config.py` (`PCARConfig`). Defaults below match the shipped simulator.

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

Use **`PCAR-sim`** as the GitHub repository name so the clone directory matches the project branding; all commands below assume your shell’s current directory is **the repository root** (where `README.md` and `main.py` live).

## Usage

From the **repository root**:

```bash
# Run with all defaults (recommended starting point)
python main.py

# Run with custom beta and payment count
python main.py --beta 2.0 --n-payments 500

# Enable dynamic-β scheduling and multipath splitting
python main.py --dynamic-beta --multipath

# Full custom run saving results to CSV
python main.py --beta 1.5 --k 10 --n-payments 1000 --n-nodes 100 --seed 42 --save-results
```

Boolean flags: `--multipath`, `--dynamic-beta`, `--save-results`. Additional hyperparameters (`--alpha`, `--gamma`, `--lambda`, …) are documented via `python main.py --help`. Paths such as `results/` are relative to the **repository root**.

## Outputs

- **Console summary** — After the Monte Carlo loop, PCAR-sim prints a formatted **Baseline vs PCAR** table (success rate, entropy, anonymity-set surrogate, fee overhead, average hop count, adversary accuracy).
- **`results/dashboard.png`** — Dark-themed **3×2** mosaic assembled by `visualise/dashboard.py`.
- **`results/beta_sweep.png`** — Combined **2×2** figure from `python main.py --beta-sweep` (entropy, adversary accuracy, *2^H*, fee overhead vs *β*).
- **Standalone figures** — `network_demo.png`, `entropy_lines.png`, `entropy_anon_bar.png`, `entropy_adversary.png`, `tradeoff_scatter.png`, `tradeoff_beta_success.png`, `tradeoff_rank_hist.png`.
- **`results/metrics.csv`** (optional) — Paired per-payment logs when `--save-results` is passed.

## Results & Interpretation

The dashboard contrasts hub topology and exemplar routes (green = sampled PCAR, red = cheapest path), rolling adversary accuracy, fee–anonymity scatter, success vs *β*, and Yen rank histograms. The analytic table below is a paper-style upper bound; **`python main.py --beta-sweep --n-payments 500 --seed 42`** prints a console summary and **`results/beta_sweep.png`** for the same *β* grid on this simulator.

Analytical reference table (upper-bound approximations; uniform distinguishability idealisation):

| \(\beta\) | \(H\) (bits) | Est. anonymity set | Fee overhead |
|-----------|--------------|-------------------|--------------|
| \(\infty\) (LN today) | 0.00 | ≈ 1 | ~ 20 % |
| 5.0 | 0.71 | ≈ 4 | ~ 4 % |
| 2.0 | 1.58 | ≈ 10 | ~ 12 % |
| 1.0 | 1.97 | ≈ 20 | ~ 21 % |
| 0.5 | 2.18 | ≈ 35 | ~ 38 % |

## Limitations & Honest Caveats

- **Synthetic topology** — PCAR-sim uses Barabási–Albert–style random graphs, not a labelled snapshot of mainnet; hub concentration, channel sizes, and liquidity drift may differ materially from production.
- **Privacy metrics are coarse** — Shannon entropy and \(2^{H}\) are exploratory indicators, not rigorous anonymity guarantees under joint statistical attacks or adaptive probing.
- **Threat model gaps** — Global passive adversaries, ISP-level correlation, Bitcoin-layer clustering, and wallet fingerprinting beyond routing are **not** modeled end-to-end here.
- **Dynamic \(\beta\) scheduling** — Adapting \(\beta\) from rolling success leaks temporal structure unless smoothed or coupled with additional noise; production deployments would need deliberate mitigation.
- **Fees vs analytic reference** — Initial runs using a wide synthetic fee range (1–500 ppm) produced fee overhead of ~80%, significantly above the paper’s analytical estimate of ~12% at \(\beta=2\). After calibrating edge parameters to real LN mainnet values (base fee 1000 msat, fee rate 1–50 ppm), fee overhead dropped to approximately **62.7%** on a representative run (`python main.py --beta 2.0 --n-payments 500 --seed 42`), closer in spirit to separating “cheap” from “sampled” paths but still above ~12% because the graph and payment model remain synthetic—validation on a real LN snapshot remains future work.

## References

- **Kappos et al.** — Empirical privacy analysis of the Lightning Network (*Financial Cryptography, FC 2021*).
- **Kumble et al.** — Formal discussion of routing-anonymity collapse under deterministic policies (*ARES 2021*).
- **Malavolta et al.** — Anonymous Multi-Hop Locks (*NDSS 2019*).
- **Pickhardt & Richter** — Optimal reliable payment flows / liquidity diagnostics (*arXiv 2021*).
- **Rohrer & Tschorsch** — HTLC timelock inference and privacy implications (*AFT 2020*).
- **Danezis & Syverson** — UC-framework analysis of onion routing (*ACM TISSEC 2012*).

## License

Released under the **MIT License** — see the [`LICENSE`](LICENSE) file.
