"""Pillar 3 (plus glue for pillars 1 and 2): forecast today's D1 Close, High and Low.

Pipeline for one forecast
-------------------------
1. Daily log-returns of the NY-close candles (futures: roll-like outliers neutralised).
2. HMM regime mixture for today -> regime volatility and regime drift.
3. Daily volatility sigma_D = blend(regime vol, EWMA vol), clipped to a sane band.
4. Drift mu_D = shrunk(regime drift) + shrunk(Kalman velocity), capped at 0.2 x sigma_D.
5. Jump parameters from days whose |return| exceeds 2.5 x robust (MAD) sigma.
6. Intraday variance profile (by New York clock hour) from the hourly bars.
7. Merton jump-diffusion on the REMAINING hourly segments of the session, starting
   from the latest price, with Brownian-bridge extremes for High/Low and the already
   realised session High/Low merged in.

Everything is an estimate: the reported P15-P85 band is a 70% prediction interval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .data_fetcher import DataBundle, DataError, session_end_utc
from .notes import INFO, WARN, Note
from .signal_and_regime import build_features, fit_regime_model, kalman_filtered_path, kalman_trend

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class Forecast:
    open_price: float                  # session open (or reference price if unknown)
    ref_price: float                   # latest price at run time
    open_known: bool
    close_q: tuple[float, float, float]
    high_q: tuple[float, float, float]
    low_q: tuple[float, float, float]
    p_up: float                        # P(close > open)
    bias: str                          # Bullish / Netral / Bearish
    regime: str
    regime_probs: dict[str, float]
    sigma_d: float                     # daily vol used (log-return std)
    mu_d: float                        # daily drift used (log-return)
    lam: float                         # jump intensity per day after multipliers
    kalman_velocity: float             # $/day
    kalman_se: float
    kalman_label: str                  # naik / turun / datar
    vol_ratio: float                   # sigma_D vs its 30-day average
    vol_label: str                     # lebih lebar / normal / lebih sempit
    features_used: list[str]
    remaining_var_frac: float          # share of the day's variance still to come
    n_paths: int
    realized_high: Optional[float]
    realized_low: Optional[float]
    avg_range_5: float = 0.0           # mean High-Low of the last 5 complete sessions ($)
    avg_range_20: float = 0.0          # same over 20 sessions ($)
    notes: list[str] = field(default_factory=list)
    diag: dict = field(default_factory=dict)      # every intermediate number (for the report)


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return max(1.4826 * float(np.median(np.abs(x - np.median(x)))), 1e-6)


def ewma_vol_series(r: np.ndarray, lam: float, init_n: int = 20) -> np.ndarray:
    """RiskMetrics EWMA; element i is the vol forecast for day i+1 using returns up to i."""
    r = np.asarray(r, dtype=float)
    v_prev = float(np.var(r[:init_n]))
    out = np.empty(len(r))
    for i, x in enumerate(r):
        v_prev = lam * v_prev + (1.0 - lam) * x * x
        out[i] = v_prev
    return np.sqrt(out)


def neutralise_roll(r: pd.Series) -> tuple[pd.Series, int]:
    """Futures only: zero out roll-like outliers (|r| > max(5 robust sigma, 2.5%))."""
    thr = max(5 * robust_sigma(r.values), 0.025)
    bad = r.abs() > thr
    return r.where(~bad, 0.0), int(bad.sum())


def calibrate_jumps(r: np.ndarray, jcfg: dict) -> tuple[float, float, float, float]:
    """Return (lambda_per_day, mu_J, sigma_J, robust_sigma)."""
    med, s = float(np.median(r)), robust_sigma(r)
    mask = np.abs(r - med) > jcfg["mad_threshold"] * s
    lam = float(np.clip(mask.mean(), jcfg["lambda_min"], jcfg["lambda_max"]))
    sig_j = float(np.std(r[mask])) if mask.sum() >= 5 else 3.0 * s
    return lam, 0.0, max(sig_j, 2.0 * s), s


# --------------------------------------------------------------------------
# Intraday structure
# --------------------------------------------------------------------------
def intraday_profile(hourly: Optional[pd.DataFrame], tz: str) -> np.ndarray:
    """Share of daily variance per New York clock hour (sums to 1)."""
    uniform = np.full(24, 1.0 / 24)
    if hourly is None or len(hourly) < 500:
        return uniform
    logp = np.log(hourly["Close"].astype(float))
    r = logp.diff()
    consecutive = (hourly.index.to_series().diff() == pd.Timedelta(hours=1)).values
    r = r[consecutive]
    if len(r) < 400:
        return uniform
    hours = r.index.tz_convert(tz).hour.values
    r2 = r.values ** 2
    r2 = np.minimum(r2, np.quantile(r2, 0.995))               # tame isolated spikes
    prof = np.array([r2[hours == h].mean() if (hours == h).sum() >= 20 else np.nan for h in range(24)])
    if np.isnan(prof).all():
        return uniform
    prof = np.where(np.isnan(prof), np.nanmean(prof), prof)
    return prof / prof.sum()


def segment_cuts(now_utc: pd.Timestamp, end_utc: pd.Timestamp) -> list[pd.Timestamp]:
    """Boundaries of the hourly segments between `now` and the session end."""
    if end_utc <= now_utc:
        raise DataError("Session already closed; nothing left to forecast.")
    cuts = [now_utc]
    t = now_utc.floor("h") + pd.Timedelta(hours=1)
    while t < end_utc:
        cuts.append(t)
        t += pd.Timedelta(hours=1)
    cuts.append(end_utc)
    return cuts


def segment_weights(
    now_utc: pd.Timestamp, end_utc: pd.Timestamp, profile: np.ndarray, tz: str
) -> np.ndarray:
    """Variance weight of each remaining segment between `now` and the session end."""
    cuts = segment_cuts(now_utc, end_utc)
    w = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        hour = a.tz_convert(tz).hour
        w.append(profile[hour] * (b - a).total_seconds() / 3600.0)
    return np.array(w)


def event_segments(
    cuts: list[pd.Timestamp], weights: np.ndarray, event_times: Optional[list[pd.Timestamp]], max_events: int
) -> list[int]:
    """Segment index (0-based) in which each scheduled event lands.

    With known times: the segment containing each event (earliest `max_events` kept).
    Without times (e.g. the --event flag): one event at the variance midpoint of the session.
    """
    n = len(weights)
    if event_times:
        inside = sorted(t for t in event_times if cuts[0] <= t <= cuts[-1])[:max_events]
        return [min(max(int(np.searchsorted(cuts, t, side="right")) - 1, 0), n - 1) for t in inside]
    mid = float(np.searchsorted(np.cumsum(weights), weights.sum() / 2.0))
    return [int(min(mid, n - 1))]


def session_so_far(hourly: Optional[pd.DataFrame], start_utc: pd.Timestamp) -> Optional[dict]:
    if hourly is None:
        return None
    s = hourly[hourly.index >= start_utc]
    if s.empty:
        return None
    return {
        "open": float(s["Open"].iloc[0]),
        "high": float(s["High"].max()),
        "low": float(s["Low"].min()),
        "last": float(s["Close"].iloc[-1]),
    }


# --------------------------------------------------------------------------
# Monte Carlo (Merton jump-diffusion with Brownian-bridge extremes)
# --------------------------------------------------------------------------
def simulate_day(
    p0: float,
    weights: np.ndarray,
    sigma_diff: float,
    mu_d: float,
    lam: float,
    mu_j: float,
    sig_j: float,
    n_paths: int,
    rng: np.random.Generator,
    return_paths: bool = False,
    scheduled: Optional[dict[int, float]] = None,
) -> dict[str, np.ndarray]:
    """Simulate the remaining session. Returns arrays (price units) of close/high/low
    measured over the simulated part only (realised high/low are merged by the caller).

    Per segment j with variance weight w_j:
        dx = mu_d*w_j + sigma_diff*sqrt(w_j)*Z + sum of N~Poisson(lam*w_j) jumps
    """
    x = np.zeros(n_paths)
    run_max = np.zeros(n_paths)
    run_min = np.zeros(n_paths)
    path_log = [x.copy()] if return_paths else None
    scheduled = scheduled or {}
    for j, wj in enumerate(weights):
        z = rng.standard_normal(n_paths)
        a = x
        b = a + mu_d * wj + sigma_diff * np.sqrt(wj) * z
        d = b - a
        s2 = sigma_diff ** 2 * wj
        u1 = 1.0 - rng.random(n_paths)                      # in (0, 1]
        u2 = 1.0 - rng.random(n_paths)
        seg_max = 0.5 * (a + b + np.sqrt(d * d - 2.0 * s2 * np.log(u1)))
        seg_min = 0.5 * (a + b - np.sqrt(d * d - 2.0 * s2 * np.log(u2)))
        n_j = rng.poisson(lam * wj, n_paths)
        jump = mu_j * n_j + sig_j * np.sqrt(n_j) * rng.standard_normal(n_paths)
        if j in scheduled:                                   # scheduled release: extra Gaussian jump
            jump = jump + scheduled[j] * rng.standard_normal(n_paths)
        x = b + jump
        run_max = np.maximum(run_max, np.maximum(seg_max, x))
        run_min = np.minimum(run_min, np.minimum(seg_min, x))
        if return_paths:
            path_log.append(x.copy())
    out = {
        "close": p0 * np.exp(x),
        "high": p0 * np.exp(run_max),
        "low": p0 * np.exp(run_min),
    }
    if return_paths:
        out["paths"] = p0 * np.exp(np.stack(path_log, axis=1).astype(np.float32))   # n_paths x (n_seg+1)
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def forecast_day(
    bundle: DataBundle,
    macro: Optional[pd.DataFrame],
    cfg: dict,
    event_today: bool = False,
    explain: bool = False,
    event_times: Optional[list[pd.Timestamp]] = None,
) -> Forecast:
    m = cfg["model"]
    tz = cfg["market"]["candle_cutoff_tz"]
    hour = cfg["market"]["candle_cutoff_hour"]
    notes: list[str] = []

    close = bundle.candles["Close"].astype(float)
    r = np.log(close).diff().dropna()
    if bundle.is_futures:
        r, n_roll = neutralise_roll(r)
        if n_roll:
            notes.append(Note("Lompatan kontrak dinetralkan",
                              f"{n_roll} return mirip pergantian kontrak di riwayat futures dinetralkan "
                              f"agar tidak menggelembungkan volatilitas.", INFO))

    # ---- Pillar 2: regimes
    feats, used = build_features(r, macro, rv_window=10, min_coverage=m["macro"]["min_coverage"])
    if len(feats) < m["hmm"]["min_history"]:
        raise DataError(f"Only {len(feats)} observations for the HMM; need >= {m['hmm']['min_history']}.")
    regime = fit_regime_model(
        feats, n_inits=m["hmm"]["n_inits"], n_iter=m["hmm"]["n_iter"], seed=m["hmm"]["seed"]
    )
    pi = regime.probs_next
    sigma_regime = float(np.sqrt(np.sum(pi * regime.sigma ** 2)))
    mu_regime = float(np.sum(pi * regime.mu))

    # ---- daily volatility
    vcfg = m["vol"]
    ewma = ewma_vol_series(r.values, vcfg["ewma_lambda"])
    rv_ref = float(r.tail(60).std())
    sigma_raw = vcfg["w_regime"] * sigma_regime + (1 - vcfg["w_regime"]) * float(ewma[-1])
    sigma_d = float(np.clip(sigma_raw, vcfg["floor_mult"] * rv_ref, vcfg["cap_mult"] * rv_ref))
    if sigma_d != sigma_raw:
        notes.append(Note("Volatilitas dibatasi",
                          "Estimasi volatilitas harian dipotong ke batas bawah atau atas yang dianggap wajar.",
                          INFO))

    # ---- Pillar 1: Kalman drift
    kal = kalman_trend(close)
    mu_kal = kal.velocity / float(close.iloc[-1])
    dcfg = m["drift"]
    mu_d = dcfg["k_regime"] * mu_regime + dcfg["k_kalman"] * mu_kal
    mu_d = float(np.clip(mu_d, -dcfg["cap_sigma_mult"] * sigma_d, dcfg["cap_sigma_mult"] * sigma_d))

    # ---- jumps
    jcfg = m["jumps"]
    lam0, mu_j, sig_j, _ = calibrate_jumps(r.values, jcfg)
    mult = 1.0
    if bundle.session_open is None:
        mult *= jcfg["sunday_mult"]
        notes.append(Note("Sesi pasca akhir pekan",
                          "Risiko lompatan dinaikkan karena sesi ini mengikuti akhir pekan.", INFO))
    lam = float(min(lam0 * mult, jcfg["lambda_cap"]))
    sigma_diff = float(np.sqrt(max(sigma_d ** 2 - lam0 * sig_j ** 2, 0.5 * sigma_d ** 2)))

    # ---- intraday structure and realised part of the session
    profile = intraday_profile(bundle.hourly, tz)
    end_utc = session_end_utc(bundle.target_session, tz, hour)
    start_utc = session_end_utc(bundle.target_session - pd.Timedelta(days=1), tz, hour)
    # If the market is still closed (weekend), the candle has not started: simulate the WHOLE
    # session from its open, not the closed hours in between (that would inflate the variance).
    now_eff = max(bundle.now_utc, start_utc)
    weights = segment_weights(now_eff, end_utc, profile, tz)
    so_far = session_so_far(bundle.hourly, start_utc)

    p0 = float(bundle.reference_price)
    open_price = float(bundle.session_open) if bundle.session_open is not None else p0
    realized_high = so_far["high"] if so_far else None
    realized_low = so_far["low"] if so_far else None

    # ---- scheduled high-impact events: extra variance at the release hour
    scheduled: dict[int, float] = {}
    sigma_event = float(np.sqrt(jcfg["event_extra_var"]) * sigma_d)
    if event_today:
        cuts = segment_cuts(now_eff, end_utc)
        for j in event_segments(cuts, weights, event_times, int(jcfg["max_scheduled_events"])):
            scheduled[j] = float(np.sqrt(scheduled.get(j, 0.0) ** 2 + sigma_event ** 2))
        notes.append(Note("Event USD berdampak tinggi hari ini",
                          "Varians tambahan dimasukkan saat rilis, sehingga kisaran lebih lebar.", WARN))

    # ---- Monte Carlo
    seed = int(bundle.target_session.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    sim = simulate_day(p0, weights, sigma_diff, mu_d, lam, mu_j, sig_j, int(m["n_paths"]), rng,
                       return_paths=explain, scheduled=scheduled)
    close_s = sim["close"]
    high_s = sim["high"] if realized_high is None else np.maximum(sim["high"], realized_high)
    low_s = sim["low"] if realized_low is None else np.minimum(sim["low"], realized_low)
    # session extremes can never exclude the open itself
    high_s = np.maximum(high_s, open_price)
    low_s = np.minimum(low_s, open_price)

    qs = list(m["quantiles"])
    qt = lambda a: tuple(float(v) for v in np.quantile(a, qs))
    p_up = float(np.mean(close_s > open_price))

    bcfg = m["bias"]
    bias = "Bullish" if p_up >= bcfg["bull_above"] else "Bearish" if p_up <= bcfg["bear_below"] else "Netral"
    if p_up > 0.70 or p_up < 0.30:
        notes.append(Note("Peluang arah sangat ekstrem",
                          f"Peluang candle naik {p_up:.0%}. Angka seperti ini cenderung muncul bila harga "
                          f"sudah jauh dari open dan sisa sesi pendek; baca sebagai aritmetika harga saat "
                          f"ini, bukan keyakinan arah (backtest tidak menunjukkan kemampuan menebak arah).",
                          WARN))

    recent = ewma[-30:]
    vol_ratio = float(sigma_d / np.mean(recent)) if len(recent) else 1.0
    vol_label = "lebih lebar" if vol_ratio > 1.15 else "lebih sempit" if vol_ratio < 0.87 else "normal"

    mu_raw = dcfg["k_regime"] * mu_regime + dcfg["k_kalman"] * mu_kal
    diag = {
        "n_returns": int(len(r)),
        "kalman": {
            "level": kal.level, "velocity": kal.velocity, "se": kal.velocity_se, "z": kal.z,
            "q_level": kal.q_level, "q_slope": kal.q_slope, "last_close": float(close.iloc[-1]),
            "extrapolated_next": kal.level + kal.velocity,
        },
        "regime": {
            "names": regime.names, "probs_today": [float(x) for x in regime.probs_today],
            "probs_next": [float(x) for x in regime.probs_next], "mu": [float(x) for x in regime.mu],
            "sigma": [float(x) for x in regime.sigma], "transmat": regime.transmat.tolist(),
            "loglik": float(regime.loglik), "n_obs": int(regime.n_obs), "features": list(used),
        },
        "vol": {
            "sigma_regime": sigma_regime, "sigma_ewma": float(ewma[-1]), "rv_ref": rv_ref,
            "w_regime": vcfg["w_regime"], "sigma_raw": float(sigma_raw), "sigma_d": sigma_d,
        },
        "drift": {
            "mu_regime": mu_regime, "mu_kalman": mu_kal, "k_regime": dcfg["k_regime"],
            "k_kalman": dcfg["k_kalman"], "mu_raw": float(mu_raw),
            "cap": float(dcfg["cap_sigma_mult"] * sigma_d), "mu_d": mu_d,
        },
        "jumps": {
            "lam0": lam0, "mu_j": mu_j, "sig_j": sig_j, "mult": mult, "lam": lam,
            "sigma_diff": sigma_diff, "robust_sigma": float(robust_sigma(r.values)),
        },
        "intraday": {
            "profile": [float(x) for x in profile], "now_eff": str(now_eff), "end": str(end_utc),
            "n_segments": int(len(weights)), "remaining": float(weights.sum()),
        },
        "event": {"n": len(scheduled), "sigma_event": sigma_event if scheduled else 0.0,
                  "extra_var": float(jcfg["event_extra_var"]) if scheduled else 0.0},
        "p0": p0,
    }
    if explain:
        lev, vel = kalman_filtered_path(close, kal.q_level, kal.q_slope)
        diag["arrays"] = {
            "close": close_s, "high": high_s, "low": low_s, "paths": sim["paths"],
            "weights": weights, "hist_close": close, "kalman_level": lev, "kalman_velocity": vel,
            "gamma": regime.gamma, "gamma_dates": regime.dates,
        }

    return Forecast(
        open_price=open_price, ref_price=p0, open_known=bundle.session_open is not None,
        close_q=qt(close_s), high_q=qt(high_s), low_q=qt(low_s),
        p_up=p_up, bias=bias, regime=regime.current_name, regime_probs=regime.probs_by_name(),
        sigma_d=sigma_d, mu_d=mu_d, lam=lam,
        kalman_velocity=kal.velocity, kalman_se=kal.velocity_se, kalman_label=kal.label,
        vol_ratio=vol_ratio, vol_label=vol_label, features_used=used,
        remaining_var_frac=float(weights.sum()), n_paths=int(m["n_paths"]),
        realized_high=realized_high, realized_low=realized_low,
        avg_range_5=float((bundle.candles['High'] - bundle.candles['Low']).tail(5).mean()),
        avg_range_20=float((bundle.candles['High'] - bundle.candles['Low']).tail(20).mean()),
        notes=notes, diag=diag,
    )
