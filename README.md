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
- [Outputs](#outputs)
- [Results & Interpretation](#results--interpretation)
- [Limitations & Honest Caveats](#limitations--honest-caveats)
- [References](#references)
- [License](#license)

## Overview

**PCAR-sim** is an executable research sandbox for *Probabilistic Cost-Aware Routing (PCAR)*: a family of routing policies that deliberately inject entropy into Lightning-style path selection instead of always returning the single cheapest route. The Lightning Network is widely described as offering sender/receiver anonymity through Sphinx-style onion encryption at each hop. That narrative captures an important piece of the truth—payment identifiers are hidden hop-by-hop—but it is incomplete as a privacy story for *routing*. Fees, timelocks (`cltv_delta`), and advertised capacities are gossiped across the network under BOLT 7-style semantics; the channel graph is therefore largely public to anyone who runs a node or scrapes data.

Because path selection is typically framed as a deterministic minimisation problem over that observable graph, any adversary who knows (or guesses with high confidence) the sender, receiver, and payment amount can replay the same routing objective as the wallet software and recover the exact path with overwhelming probability. In effect, the anonymity set implied by “many possible routes” collapses to one or two concrete paths whenever routing entropy is near zero. Empirical and analytical studies on the live network have argued that this predictability is not merely theoretical; it interacts with probing, balance inference, and correlation of HTLC timing across channels.

**PCAR-sim** instantiates a stylised countermeasure: replace deterministic arg-min routing with Boltzmann-weighted sampling over a finite menu of candidate paths, augmented by an explicit privacy penalty and optional operational hardening (timelock padding, guard relays, multipath splits). The simulator reports Shannon entropy of the path distribution, a coarse anonymity-set surrogate \(2^{H}\), fee overhead versus the deterministic baseline, and how often a naive deterministic adversary would have guessed the realised path—metrics intended for comparative experiments rather than production guarantees.

The codebase targets reproducibility: numerically-intensive batches remain manageable via CLI sizing knobs (`--n-payments`, `--n-nodes`, `--seed`) while auxiliary \(\beta\) sweeps regenerate dashboards suitable for talks or supplementary plots—all implemented against synthetic graphs precisely because controlled entropy gradients matter when benchmarking probabilistic routing.

## The Problem

Routing predictability is the root privacy failure mode that PCAR targets. When wallets implement standard least-cost routing—whether through classical shortest-path algorithms or ranking APIs that effectively collapse to a similar arg-min—the sender’s behaviour becomes algorithmically reproducible. An attacker does not need to break Sphinx to learn the route; they only need to approximate the sender’s cost model (fees, delay preferences, liquidity heuristics) and execute the same optimisation on the same public graph snapshot.

Onion encryption still matters: it hides *who* forwarded *what* hop-by-hop from casual observers. But when the *distribution* over feasible paths is sharply peaked—entropy approaching zero—the ciphertext envelope does not enlarge the anonymity set in practice. The adversary’s posterior over paths is already pinned down before observing ciphertext patterns beyond timing side channels. Once the path is known or tightly constrained, dependent threats become easier: linking multi-hop payments via HTLC correlation, refining balance estimates through deliberate probing, or combining Lightning inference with broader blockchain clustering.

PCAR-sim models this threat abstractly by embedding a deterministic baseline router alongside PCAR. After each simulated payment, it asks whether a deterministic least-cost prediction equals the path actually sampled—a simple stand-in for “routing anonymity collapsed to a singleton.” The goal is not to reproduce every real-world wallet quirk, but to show how structural randomisation moves mass away from the predictable optimum.

## The Solution: PCAR

PCAR replaces deterministic arg-min selection with **softmax / Boltzmann sampling** over **\(k\)** candidate paths produced by Yen’s \(k\)-shortest path algorithm on a surrogate LN-style cost (fee-aware, delay-aware, liquidity-sensitive). Each candidate receives a composite score blending normalised fee, delay (with optional random timelock padding per hop), Pickhardt–Richter-style liquidity risk, and a privacy penalty **\(\Pi(P)\)**. The penalty aggregates three normalised sub-components: **path popularity** (penalising reuse of recently chosen tails), **intermediate-node centrality** (discouraging hub-heavy intermediaries via cached betweenness), and **path uniqueness** (down-weighting sender–receiver pairs with few alternatives). Sampling probabilities follow \(\Pr[P_i] \propto \exp(-\beta\, C(P_i))\).

The inverse temperature **\(\beta\)** governs the privacy–cost tradeoff: large \(\beta\) concentrates probability on low-cost paths (closer to today’s wallets); smaller \(\beta\) flattens the distribution and raises entropy at the expense of fees or delays. PCAR-sim also implements optional **guard-node** filtering on the first hop (low centrality and mature channels), **\(\beta\) jitter** to hinder fingerprinting of exact softmax parameters, **dynamic \(\beta\) scheduling** based on rolling payment success, **probabilistic multi-path (PMP-PCAR)** splits for large amounts, and hooks compatible in spirit with **Anonymous Multi-Hop Locks (AMHLs)** as a privacy-enhancing composition layer—though AMHL cryptography itself is not simulated here.

Analytical sketches in the PCAR literature sometimes cite order-of-magnitude gains—for example, roughly **tenfold** anonymity-set growth near **\(\beta \approx 2\)** with on the order of **12 %** fee overhead—relative to deterministic LN routing. **PCAR-sim** surfaces comparable diagnostics, but these numbers are **upper-bound-style approximations** under stylised symmetric assumptions; they are **not** certificates of privacy on the live network and require empirical validation against real wallet policies and graph snapshots.

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

- **Shannon entropy \(H\)** — PCAR-sim measures \(H = -\sum_i p_i \log_2 p_i\) over the softmax distribution across the \(k\) candidates for each payment (aggregated or averaged when multipath splits occur). Higher entropy means the realised path was drawn from a broader distribution—loosely, more routing uncertainty for an external observer who must guess among candidates.

- **Boltzmann sampling & \(\beta\)** — Probabilities scale as \(\exp(-\beta C)\). Intuitively, \(\beta\) is an “inverse temperature”: large \(\beta\) makes cheap paths dominate (low entropy, wallet-like behaviour); moderate \(\beta\) spreads probability mass across acceptable alternatives (privacy upside, fee/delay downside).

- **Anonymity set estimate \(2^{H}\)** — Under a simplifying interpretation where each candidate path is roughly distinguishable, \(2^{H}\) acts as a crude effective candidate count implied by the distribution. It is **not** a formal anonymity-set metric under joint attacks; treat it as an exploratory diagnostic aligned with standard entropy-based reporting in routing papers.

- **Pickhardt–Richter liquidity risk** — Risk along a path is modeled as \(R(P) = 1 - \prod_i (1 - a_i/c_i)\) with amounts propagated hop-by-hop including fees. This mirrors liquidity contention intuition used in Pickhardt–Richter payment-flow analyses rather than a full Bayesian channel-capacity posterior.

- **Guard-node policy** — Candidate paths can be filtered so the first intermediate relay has low normalised betweenness and a sufficiently “aged” channel (relative to the maximum channel age in the synthetic graph). When no path qualifies, PCAR-sim relaxes the constraint and emits a diagnostic—mirroring graceful degradation in experimental stacks.

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
- **Standalone figures** — `network_demo.png`, `entropy_lines.png`, `entropy_anon_bar.png`, `entropy_adversary.png`, `tradeoff_scatter.png`, `tradeoff_beta_success.png`, `tradeoff_rank_hist.png`.
- **`results/metrics.csv`** (optional) — Paired per-payment logs when `--save-results` is passed.

## Results & Interpretation

Use the dashboard as a comparative lens:

1. **Topology / routes** — Shows hub structure (betweenness heatmap) and an exemplar payment’s candidate paths; the **green** trace is the sampled PCAR path, **red** the deterministic baseline.
2. **Entropy curves** — Overlay \(H\) versus payment index for multiple \(\beta\) values; separation between curves indicates sensitivity of path entropy to temperature.
3. **Adversary accuracy** — Rolling mean of “did deterministic routing predict the realised path?” Lower is better for privacy against this straw-man attacker.
4. **Fee vs anonymity scatter** — Each point is a payment; colour encodes effective \(\beta\). Up-and-to-the-right clouds indicate paying more fees for larger anonymity-set surrogates.
5. **Success vs \(\beta\)** — Shows liquidity-limited reliability when sampling becomes too flat (\(\beta\) too small).
6. **Rank histogram** — Displays how often PCAR chose the cheapest candidate (rank 1) versus deeper Yen alternatives.

**Good privacy signals (holding cost budgets loose):** entropy rises, \(2^{H}\) grows, adversary accuracy falls. **Tuning \(\beta\)** trades reliability and fees against those signals; start near **\(\beta = 2.0\)** unless graphs are unusually hostile to exploration.

Analytical reference table (upper-bound approximations; uniform distinguishability idealisation):

| \(\beta\) | \(H\) (bits) | Est. anonymity set | Fee overhead |
|-----------|--------------|-------------------|--------------|
| \(\infty\) (LN today) | 0.00 | ≈ 1 | ~ 20 % |
| 5.0 | 0.71 | ≈ 4 | ~ 4 % |
| 2.0 | 1.58 | ≈ 10 | ~ 12 % |
| 1.0 | 1.97 | ≈ 20 | ~ 21 % |
| 0.5 | 2.18 | ≈ 35 | ~ 38 % |

These values are analytical upper-bound approximations assuming uniform path distinguishability. Actual results will vary by network topology. **\(\beta = 2.0\)** is the suggested default operating point.

## Limitations & Honest Caveats

- **Synthetic topology** — PCAR-sim uses Barabási–Albert–style random graphs, not a labelled snapshot of mainnet; hub concentration, channel sizes, and liquidity drift may differ materially from production.
- **Privacy metrics are coarse** — Shannon entropy and \(2^{H}\) are exploratory indicators, not rigorous anonymity guarantees under joint statistical attacks or adaptive probing.
- **Threat model gaps** — Global passive adversaries, ISP-level correlation, Bitcoin-layer clustering, and wallet fingerprinting beyond routing are **not** modeled end-to-end here.
- **Dynamic \(\beta\) scheduling** — Adapting \(\beta\) from rolling success leaks temporal structure unless smoothed or coupled with additional noise; production deployments would need deliberate mitigation.

## References

- **Kappos et al.** — Empirical privacy analysis of the Lightning Network (*Financial Cryptography, FC 2021*).
- **Kumble et al.** — Formal discussion of routing-anonymity collapse under deterministic policies (*ARES 2021*).
- **Malavolta et al.** — Anonymous Multi-Hop Locks (*NDSS 2019*).
- **Pickhardt & Richter** — Optimal reliable payment flows / liquidity diagnostics (*arXiv 2021*).
- **Rohrer & Tschorsch** — HTLC timelock inference and privacy implications (*AFT 2020*).
- **Danezis & Syverson** — UC-framework analysis of onion routing (*ACM TISSEC 2012*).

## License

Released under the **MIT License** — see the [`LICENSE`](LICENSE) file.
