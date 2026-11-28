"""Pillars 1 and 2: Kalman trend filter and HMM regime detection.

Kalman
------
Local linear trend on the daily close (state = [level, velocity in $/day]).
Only FILTERED estimates are used (never smoothed), so there is no look-ahead.
Noise parameters are chosen by maximum likelihood over a small grid, with the
overall scale concentrated out analytically (so the grid is only two ratios).

HMM
---
3 hidden regimes on daily features (gold return, 10-day realised volatility and,
if available, DXY return and US10Y change). State indices are arbitrary after
fitting, so states are named by the RANK of their mean realised volatility:
  lowest -> Konsolidasi, middle -> Trending (active/expansion), highest -> Shock.
Simulation showed that ranking by volatility is far more stable than ranking by
|mu|/sigma (means are much harder to estimate than volatilities from ~2 years of
daily data), so "Trending" should be read as "moderately volatile / active".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hmm import GaussianHMM

REGIME_CALM = "Konsolidasi"
REGIME_TREND = "Trending"
REGIME_SHOCK = "Shock"


# --------------------------------------------------------------------------
# Kalman filter: local linear trend
# --------------------------------------------------------------------------
@dataclass
class KalmanResult:
    level: float          # filtered level ($)
    velocity: float       # filtered velocity ($/day)
    velocity_se: float    # standard error of the velocity estimate ($/day)
    q_level: float        # chosen level-noise ratio (relative to observation noise)
    q_slope: float        # chosen slope-noise ratio

    @property
    def z(self) -> float:
        return self.velocity / self.velocity_se if self.velocity_se > 0 else 0.0

    @property
    def label(self) -> str:
        z = self.z          # |z| < 2 is indistinguishable from noise for this filter
        return "naik" if z > 2.0 else "turun" if z < -2.0 else "datar"


def _kf_run(y: np.ndarray, q_level: float, q_slope: float, burn: int = 20):
    """Run the filter with observation variance 1 (scale concentrated out).

    Returns (loglik_concentrated, level, velocity, P22, scale_s2).
    """
    n = len(y)
    l, s = float(y[0]), 0.0
    a, b, d = 1e3, 0.0, 1e3                      # P = [[a, b], [b, d]]
    sum_ln_f = 0.0
    sum_v2f = 0.0
    cnt = 0
    for t in range(1, n):
        # predict
        lp, sp = l + s, s
        a_p = a + 2 * b + d + q_level
        b_p = b + d
        d_p = d + q_slope
        # update
        f = a_p + 1.0
        v = y[t] - lp
        k1, k2 = a_p / f, b_p / f
        l, s = lp + k1 * v, sp + k2 * v
        a = a_p - a_p * a_p / f
        b = b_p - a_p * b_p / f
        d = d_p - b_p * b_p / f
        if t >= burn:
            sum_ln_f += np.log(f)
            sum_v2f += v * v / f
            cnt += 1
    s2 = max(sum_v2f / max(cnt, 1), 1e-12)
    loglik = -0.5 * sum_ln_f - 0.5 * cnt * np.log(s2)
    return loglik, l, s, d, s2


def kalman_trend(close: pd.Series) -> KalmanResult:
    """Fit the local linear trend by grid-search MLE and return the filtered end state."""
    y = np.asarray(close, dtype=float)
    if len(y) < 60:
        raise ValueError("Need at least 60 observations for the Kalman filter.")
    best = None
    for q_level in (10, 30, 100, 300, 1000, 3000, 10000):
        for q_slope in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
            ll, l, s, d, s2 = _kf_run(y, q_level, q_slope)
            if best is None or ll > best[0]:
                best = (ll, l, s, d, s2, q_level, q_slope)
    _, l, s, d, s2, q_level, q_slope = best
    return KalmanResult(
        level=l, velocity=s, velocity_se=float(np.sqrt(max(d * s2, 0.0))),
        q_level=q_level, q_slope=q_slope,
    )


def kalman_filtered_path(close: pd.Series, q_level: float, q_slope: float) -> tuple[np.ndarray, np.ndarray]:
    """Filtered level and velocity for every day (for plotting / explanation only)."""
    y = np.asarray(close, dtype=float)
    n = len(y)
    lev, vel = np.empty(n), np.empty(n)
    l, s = float(y[0]), 0.0
    a, b, d = 1e3, 0.0, 1e3
    lev[0], vel[0] = l, s
    for t in range(1, n):
        lp, sp = l + s, s
        a_p, b_p, d_p = a + 2 * b + d + q_level, b + d, d + q_slope
        f = a_p + 1.0
        v = y[t] - lp
        k1, k2 = a_p / f, b_p / f
        l, s = lp + k1 * v, sp + k2 * v
        a, b, d = a_p - a_p * a_p / f, b_p - a_p * b_p / f, d_p - b_p * b_p / f
        lev[t], vel[t] = l, s
    return lev, vel


# --------------------------------------------------------------------------
# HMM regimes
# --------------------------------------------------------------------------
@dataclass
class RegimeResult:
    names: list[str]              # name of each hidden state
    probs_today: np.ndarray       # filtered state probabilities at the last observation
    probs_next: np.ndarray        # probs_today @ transition matrix (today's candle)
    mu: np.ndarray                # mean daily gold log-return per state
    sigma: np.ndarray             # daily gold log-return std per state
    transmat: np.ndarray
    loglik: float
    n_obs: int
    features: list[str]
    gamma: np.ndarray | None = None      # state probabilities for every day
    dates: pd.Index | None = None

    @property
    def current_name(self) -> str:
        return self.names[int(np.argmax(self.probs_next))]

    def probs_by_name(self) -> dict[str, float]:
        return {n: float(p) for n, p in zip(self.names, self.probs_next)}


def build_features(
    returns: pd.Series,
    macro: pd.DataFrame | None,
    rv_window: int = 10,
    min_coverage: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """Feature matrix indexed by session label. Macro columns are optional."""
    df = pd.DataFrame({"ret": returns})
    df["rv"] = returns.rolling(rv_window).std()
    used = ["ret", "rv"]
    if macro is not None:
        m = macro.reindex(df.index).ffill(limit=3)
        for col in ("dxy_ret", "tnx_chg"):
            if col in m.columns and m[col].iloc[rv_window:].notna().mean() >= min_coverage:
                df[col] = m[col].fillna(0.0)       # a missing day = "no change"
                used.append(col)
    df = df.dropna(subset=["ret", "rv"])
    return df[used], used


def map_regimes(means_z: np.ndarray, vars_z: np.ndarray | None = None) -> list[str]:
    """Name each state by the rank of its mean realised vol (standardised col 1)."""
    K = means_z.shape[0]
    if K != 3:
        raise ValueError("Regime naming supports exactly 3 states.")
    order = np.argsort(means_z[:, 1])             # low -> high realised volatility
    names = [""] * K
    names[order[0]], names[order[1]], names[order[2]] = REGIME_CALM, REGIME_TREND, REGIME_SHOCK
    return names


def fit_regime_model(
    features: pd.DataFrame,
    n_inits: int = 6,
    n_iter: int = 150,
    seed: int = 7,
) -> RegimeResult:
    X = features.values.astype(float)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-12
    Z = (X - mean) / std

    hmm = GaussianHMM(n_states=3, n_iter=n_iter, n_inits=n_inits, seed=seed, vol_col=1).fit(Z)
    today = hmm.gamma_[-1]
    nxt = today @ hmm.transmat_
    mu = hmm.means_[:, 0] * std[0] + mean[0]
    sigma = np.sqrt(hmm.vars_[:, 0]) * std[0]
    return RegimeResult(
        names=map_regimes(hmm.means_, hmm.vars_),
        probs_today=today, probs_next=nxt, mu=mu, sigma=sigma,
        transmat=hmm.transmat_, loglik=hmm.loglik_, n_obs=len(X),
        features=list(features.columns), gamma=hmm.gamma_, dates=features.index,
    )
