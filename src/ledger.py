"""Forecast ledger: a public-safe track record of the bot's own forecasts.

What is stored (data/predictions_log.csv, one row per target session)
--------------------------------------------------------------------
* the forecast bands as FRACTIONS of the session's base price (e.g. +0.0042), never as prices;
* probability, regime, volatility (our own model outputs);
* after the session is complete: only 0/1 hit flags (was Close/High/Low inside its P15-P85
  band? did the candle close above its base price?).

No raw prices and no realised returns are stored. The base price needed to settle a row is
recovered from the candles available at settlement time:
  * open known at forecast time  -> base = that candle's Open
  * weekend run (open unknown)   -> base = previous candle's Close (Friday close)

This keeps the file safe for a public repository while still giving live coverage,
the probability-direction Brier score, and a heartbeat commit that keeps the scheduled
workflow alive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data_fetcher import DataBundle
from .quant_engine import Forecast

COLUMNS = [
    "session", "created_utc", "source", "quality", "open_known",
    "c15", "c50", "c85", "h15", "h50", "h85", "l15", "l50", "l85",
    "p_up", "bias", "regime", "sigma_d",
    "settled", "close_hit", "high_hit", "low_hit", "up_actual", "event",
]
OPTIONAL_COLS = ["event"]       # columns added after the first release; old files get NaN
FLOAT_COLS = ["c15", "c50", "c85", "h15", "h50", "h85", "l15", "l50", "l85", "p_up", "sigma_d"]
FLAG_COLS = ["settled", "close_hit", "high_hit", "low_hit", "up_actual", "open_known", "event"]


def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})


def load_ledger(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return empty_ledger()
    df = pd.read_csv(path, dtype={"session": str, "created_utc": str, "source": str,
                                  "quality": str, "bias": str, "regime": str})
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Ledger is missing columns: {missing}")
    for c in FLOAT_COLS + FLAG_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[COLUMNS].reset_index(drop=True)


def save_ledger(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_values("session").reset_index(drop=True)
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp, index=False, float_format="%.6f")
    tmp.replace(path)                                   # atomic: never leaves a half-written file


def upsert_forecast(
    ledger: pd.DataFrame, fc: Forecast, bundle: DataBundle, now_utc: pd.Timestamp,
    event_flag: bool = False,
) -> pd.DataFrame:
    """Add today's forecast (replacing a re-run for the same session)."""
    base = float(fc.open_price)
    rel = lambda v: float(v) / base - 1.0
    row = {
        "session": bundle.target_session.strftime("%Y-%m-%d"),
        "created_utc": now_utc.strftime("%Y-%m-%dT%H:%MZ"),
        "source": "futures" if bundle.is_futures else "spot",
        "quality": bundle.quality,
        "open_known": int(fc.open_known),
        "c15": rel(fc.close_q[0]), "c50": rel(fc.close_q[1]), "c85": rel(fc.close_q[2]),
        "h15": rel(fc.high_q[0]), "h50": rel(fc.high_q[1]), "h85": rel(fc.high_q[2]),
        "l15": rel(fc.low_q[0]), "l50": rel(fc.low_q[1]), "l85": rel(fc.low_q[2]),
        "p_up": float(fc.p_up), "bias": fc.bias, "regime": fc.regime, "sigma_d": float(fc.sigma_d),
        "settled": 0, "close_hit": np.nan, "high_hit": np.nan, "low_hit": np.nan, "up_actual": np.nan,
        "event": int(event_flag),
    }
    keep = ledger[ledger["session"] != row["session"]]
    new = pd.DataFrame([row], columns=COLUMNS)
    if keep.empty:
        return new
    return pd.concat([keep, new], ignore_index=True)


def settle(ledger: pd.DataFrame, candles: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill the hit flags of every unsettled row whose session is now a complete candle."""
    ledger = ledger.copy()
    idx = candles.index
    settled_now = 0
    for i in ledger.index[ledger["settled"] != 1]:
        row = ledger.loc[i]
        s = pd.Timestamp(row["session"])
        if s not in idx:
            continue
        pos = idx.get_loc(s)
        if int(row["open_known"]) == 1:
            base = float(candles["Open"].iloc[pos])
        else:
            if pos == 0:
                continue
            base = float(candles["Close"].iloc[pos - 1])
        c = candles.iloc[pos]
        inside = lambda lo, hi, v: float(base * (1 + row[lo]) <= v <= base * (1 + row[hi]))
        ledger.loc[i, "close_hit"] = inside("c15", "c85", float(c["Close"]))
        ledger.loc[i, "high_hit"] = inside("h15", "h85", float(c["High"]))
        ledger.loc[i, "low_hit"] = inside("l15", "l85", float(c["Low"]))
        ledger.loc[i, "up_actual"] = float(float(c["Close"]) > base)
        ledger.loc[i, "settled"] = 1
        settled_now += 1
    return ledger, settled_now


def live_accuracy(ledger: pd.DataFrame, window: int = 30, min_n: int = 10) -> dict:
    """Coverage of the P15-P85 bands over the last `window` settled sessions."""
    s = ledger[ledger["settled"] == 1].sort_values("session").tail(window)
    n = len(s)
    if n == 0:
        return {"n": 0, "ready": False, "min_n": min_n, "close": np.nan, "high": np.nan,
                "low": np.nan, "brier": np.nan}
    brier = float(((s["p_up"] - s["up_actual"]) ** 2).mean())
    return {
        "n": n, "ready": n >= min_n, "min_n": min_n,
        "close": float(s["close_hit"].mean()), "high": float(s["high_hit"].mean()),
        "low": float(s["low_hit"].mean()), "brier": brier,
    }
