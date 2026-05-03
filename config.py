"""Hyperparameters for PCAR-sim (`PCARConfig`), importable and CLI-overridable."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class PCARConfig:
    # Routing & softmax
    k: int = 10
    beta: float = 2.0
    alpha: float = 0.3
    gamma: float = 0.3
    delta: float = 0.2
    lambda_: float = 0.2  # privacy weight λ in composite cost
    # Privacy penalty weights (w1+w2+w3 typically = 1)
    w1: float = 0.4
    w2: float = 0.4
    w3: float = 0.2
    mu: float = 0.1
    nu: float = 0.2
    T_paths: int = 100  # circular buffer length for π_pop (paper notation T)
    delta_max: int = 10  # timelock padding U(0, delta_max) per hop (blocks)
    B_thresh: float = 0.7  # guard: betweenness below threshold
    U_min: float = 0.9  # guard: normalised channel age must exceed this (see routing docstring)
    beta_noise_delta: float = 0.1  # δ_beta: β_eff = β + U(-δ_beta, +δ_beta)

    # Dynamic β scheduler
    tau_lo: float = 0.6
    tau_hi: float = 0.85
    delta_beta_schedule: float = 0.2
    beta_min: float = 0.5
    beta_max: float = 5.0
    rolling_window: int = 100

    # Simulation defaults
    n_payments: int = 500
    n_nodes: int = 100
    seed: int = 42
    multipath_threshold_sat: int = 250_000  # split above this if --multipath

    # Fee parameters calibrated to real Lightning Network mainnet values.
    # base_fee: ~1000 msat is the most common value on mainnet.
    # fee_rate: real channels typically 1–50 ppm, not 1–500 ppm.
    # This calibration reduces synthetic fee variance and brings
    # fee overhead estimates closer to the paper's ~12% at β=2.
    # Metrics / refs for additive routing surrogate (scale raw fees/delays)
    fee_ref_msat: float = 5_000_000.0
    delay_ref_blocks: float = 500.0


DEFAULT_CONFIG = PCARConfig()


def merge_cli_into_config(cfg: PCARConfig, **overrides) -> PCARConfig:
    """Return a new `PCARConfig` with keys from ``overrides`` applied (PCAR-sim CLI)."""
    data = {f.name: getattr(cfg, f.name) for f in fields(PCARConfig)}
    for key, val in overrides.items():
        if key == "lambda":
            key = "lambda_"
        if val is None:
            continue
        if key in data:
            data[key] = val
    return PCARConfig(**data)
