"""Walk-forward evaluation (the honest test of the model).

For each past session the model is run exactly as it would have been at ~30 minutes
after the 17:00 New York open, using ONLY bars available at that moment, and the
forecast is compared with what actually happened.

Reported
--------
* Coverage of the P15-P85 band for Close, High and Low (target ~70%).
* Pinball loss of the Close quantiles versus a naive baseline
  (random walk from the current price, EWMA volatility, normal quantiles).
* Brier score of P(close > open) versus a coin flip (0.25), plus hit rate of the
  Bullish/Bearish calls (Netral days excluded).

The last (still forming) hourly bar is neutralised to its open price so the test
does not "see" the rest of that hour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data_fetcher import (
    DataError,
    _bundle_from_hourly,
    build_ny_close_candles,
    session_end_utc,
)
from .quant_engine import calibrate_jumps, ewma_vol_series, forecast_day

log = logging.getLogger(__name__)

QS = (0.15, 0.50, 0.85)


def pinball(y: float, q: float, tau: float) -> float:
    d = y - q
    return max(tau * d, (tau - 1.0) * d)


@dataclass
class BacktestReport:
    n: int
    skipped: int
    cover: dict[str, tuple[float, float, float]]   # name -> (below, inside, above)
    pinball_model: float
    pinball_baseline: float
    brier: float
    brier_baseline: float
    call_hit_rate: float | None
    n_calls: int
    table: pd.DataFrame

    def lines(self) -> list[str]:
        se = lambda p: 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / max(self.n, 1))
        out = [f"Sessions evaluated: {self.n} (skipped {self.skipped})",
               "",
               "Coverage of the P15-P85 band (target: 15% below / 70% inside / 15% above)"]
        for k, (b, i, a) in self.cover.items():
            out.append(f"  {k:5s}: below {b:5.1%} | inside {i:5.1%} (95% CI +/-{se(i):.1%}) | above {a:5.1%}")
        gain = 1.0 - self.pinball_model / self.pinball_baseline if self.pinball_baseline else float("nan")
        out += [
            "",
            f"Close pinball loss: model {self.pinball_model:.3f} vs baseline {self.pinball_baseline:.3f}"
            f"  ({gain:+.1%} vs baseline; positive = model better)",
            f"Direction Brier: model {self.brier:.4f} vs coin-flip {self.brier_baseline:.4f}"
            f"  (lower is better; differences below ~0.01 are noise)",
        ]
        if self.call_hit_rate is not None:
            out.append(f"Bullish/Bearish call hit rate: {self.call_hit_rate:.1%} over {self.n_calls} calls")
        else:
            out.append("No Bullish/Bearish calls (all Netral).")
        return out


def run_backtest(
    hourly_full: pd.DataFrame, cfg: dict, n_sessions: int = 120, symbol: str = "XAU/USD",
    is_futures: bool = False, macro: pd.DataFrame | None = None, progress: bool = False,
) -> BacktestReport:
    tz, hour = cfg["market"]["candle_cutoff_tz"], cfg["market"]["candle_cutoff_hour"]
    end_probe = hourly_full.index[-1] + pd.Timedelta(hours=2)
    truth_all = build_ny_close_candles(hourly_full, tz, hour, end_probe)
    truth = truth_all[truth_all["complete"]]
    labels = truth.index[-n_sessions:]

    rows = []
    skipped = 0
    for i, label in enumerate(labels):
        start = session_end_utc(label - pd.Timedelta(days=1), tz, hour)
        now = start + pd.Timedelta(minutes=30)
        cut = hourly_full[hourly_full.index <= now].copy()
        if cut.empty:
            skipped += 1
            continue
        last_ts = cut.index[-1]
        if last_ts + pd.Timedelta(hours=1) > now:                 # bar still forming at `now`
            cut.loc[last_ts, ["High", "Low", "Close"]] = cut.loc[last_ts, "Open"]
        try:
            bundle = _bundle_from_hourly(symbol, is_futures, cfg, now, loader=lambda c=cut: c)
            fc = forecast_day(bundle, macro, cfg)
        except DataError as exc:
            log.debug("skip %s: %s", label.date(), exc)
            skipped += 1
            continue

        actual = truth.loc[label]
        # baseline: random walk from the current price with EWMA volatility
        r = np.log(bundle.candles["Close"]).diff().dropna().values
        sig = float(ewma_vol_series(r, cfg["model"]["vol"]["ewma_lambda"])[-1])
        sig_rem = sig * np.sqrt(fc.remaining_var_frac)
        base_q = [fc.ref_price * np.exp(norm.ppf(t) * sig_rem) for t in QS]

        rows.append({
            "session": label.date(), "open": fc.open_price, "ref": fc.ref_price,
            "c15": fc.close_q[0], "c50": fc.close_q[1], "c85": fc.close_q[2],
            "h15": fc.high_q[0], "h85": fc.high_q[2], "l15": fc.low_q[0], "l85": fc.low_q[2],
            "p_up": fc.p_up, "bias": fc.bias, "regime": fc.regime,
            "a_close": float(actual["Close"]), "a_high": float(actual["High"]), "a_low": float(actual["Low"]),
            "a_open": float(actual["Open"]),
            "pin_model": float(np.mean([pinball(actual["Close"], q, t) for q, t in zip(fc.close_q, QS)])),
            "pin_base": float(np.mean([pinball(actual["Close"], q, t) for q, t in zip(base_q, QS)])),
        })
        if progress and (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(labels)} sessions", flush=True)

    if not rows:
        raise DataError("Backtest produced no forecasts.")
    df = pd.DataFrame(rows)

    def split(actual: str, lo: str, hi: str) -> tuple[float, float, float]:
        below = float((df[actual] < df[lo]).mean())
        above = float((df[actual] > df[hi]).mean())
        return below, 1.0 - below - above, above

    cover = {
        "close": split("a_close", "c15", "c85"),
        "high": split("a_high", "h15", "h85"),
        "low": split("a_low", "l15", "l85"),
    }
    outcome = (df["a_close"] > df["open"]).astype(float)
    brier = float(((df["p_up"] - outcome) ** 2).mean())
    calls = df[df["bias"] != "Netral"]
    hit = None
    if len(calls):
        pred_up = (calls["bias"] == "Bullish")
        hit = float((pred_up == (calls["a_close"] > calls["open"])).mean())
    return BacktestReport(
        n=len(df), skipped=skipped, cover=cover,
        pinball_model=float(df["pin_model"].mean()), pinball_baseline=float(df["pin_base"].mean()),
        brier=brier, brier_baseline=0.25, call_hit_rate=hit, n_calls=len(calls), table=df,
    )


# --------------------------------------------------------------------------
# Event study: do high-impact event days really have more variance?
# --------------------------------------------------------------------------
# FOMC decision days (second day of each meeting, 14:00 New York), from the Federal Reserve calendar.
FOMC_DECISION_DAYS = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]


def event_study(candles: pd.DataFrame, event_dates: list[str], cfg: dict,
                n_boot: int = 5000, seed: int = 0) -> dict:
    """Compare close-to-close variance on event days with normal days.

    The model adds extra variance `event_extra_var x sigma_D^2` at the release hour, so the day's
    variance becomes (1 + event_extra_var) x normal. The observed variance ratio therefore maps
    directly to the recommended `model.jumps.event_extra_var = ratio - 1` (never below 0), with a
    bootstrap 95% interval.
    """
    r = np.log(candles["Close"].astype(float)).diff().dropna()
    is_event = r.index.isin(pd.DatetimeIndex(event_dates))
    r_ev, r_nm = r[is_event].values, r[~is_event].values
    if len(r_ev) < 5:
        raise DataError(f"Only {len(r_ev)} event days inside the history; need at least 5.")

    nm_var = float(np.mean(r_nm ** 2))
    ratio = float(np.mean(r_ev ** 2) / nm_var)
    rng = np.random.default_rng(seed)
    boots = np.array([np.mean(rng.choice(r_ev ** 2, size=len(r_ev), replace=True)) / nm_var for _ in range(n_boot)])
    lo, hi = (float(x) for x in np.quantile(boots, [0.025, 0.975]))

    rg = ((candles["High"] - candles["Low"]) / candles["Close"]).reindex(r.index)
    range_ratio = float(rg[is_event].mean() / rg[~is_event].mean())
    configured = float(cfg["model"]["jumps"]["event_extra_var"])
    return {
        "n_event": int(len(r_ev)), "n_normal": int(len(r_nm)),
        "var_ratio": ratio, "ci": (lo, hi), "range_ratio": range_ratio,
        "recommended_extra_var": max(ratio - 1.0, 0.0),
        "extra_var_ci": (max(lo - 1.0, 0.0), max(hi - 1.0, 0.0)),
        "configured_extra_var": configured, "configured_ratio": 1.0 + configured,
    }
