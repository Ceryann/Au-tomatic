"""Compare the bot's D1 candles (Yahoo, 17:00 NY boundary) with an MT5 D1 export.

How to export from MT5
----------------------
1. MT5 -> View -> Symbols -> "Bars" tab.
2. Symbol: your gold symbol (e.g. XAUUSD), Timeframe: Daily (D1).
3. Date range: last 60+ days -> "Request" -> "Export Bars" -> save as CSV.

Then run (needs internet, because it downloads from Yahoo):
    python tools/compare_mt5.py path/to/XAUUSD_Daily.csv --days 30

What to look for
----------------
* Day alignment : "best lag" must be 0. Otherwise the candle boundary differs.
* Level offset  : a stable mean difference of a few dollars is normal (different
  feeds / spot vs broker). Large or noisy differences mean a boundary problem.
* High/Low gaps : Yahoo hourly bars can miss brief spikes, so Yahoo High is often
  slightly below MT5 High and Yahoo Low slightly above MT5 Low.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_mt5_csv(path: str) -> pd.DataFrame:
    """Read an MT5 'Export Bars' CSV (tab or comma separated, <DATE> style headers)."""
    raw = pd.read_csv(path, sep=None, engine="python")
    raw.columns = [str(c).strip("<> ").strip().title() for c in raw.columns]
    if "Date" not in raw.columns:
        raise ValueError(f"No <DATE> column found. Columns: {list(raw.columns)}")
    if "Time" in raw.columns:
        raise ValueError("This looks like an intraday export; please export timeframe D1.")
    raw["Date"] = pd.to_datetime(raw["Date"].astype(str).str.strip(), format="%Y.%m.%d", errors="coerce")
    if raw["Date"].isna().all():
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    out = raw.dropna(subset=["Date"]).set_index("Date")[["Open", "High", "Low", "Close"]].astype(float)
    out.index.name = "session"
    return out.sort_index()


def compare(bot: pd.DataFrame, mt5: pd.DataFrame, days: int = 30) -> dict:
    """Return diagnostics comparing bot candles with MT5 candles (bot - MT5)."""
    joined = pd.merge(
        bot[["Open", "High", "Low", "Close"]], mt5,
        left_index=True, right_index=True, how="inner", suffixes=("_bot", "_mt5"),
    ).tail(days)
    if joined.empty:
        raise ValueError("No overlapping dates between bot candles and the MT5 export.")

    diffs = pd.DataFrame({c: joined[f"{c}_bot"] - joined[f"{c}_mt5"] for c in ["Open", "High", "Low", "Close"]})

    # Day alignment: correlate close-to-close returns at lags -1, 0, +1.
    b_ret = bot["Close"].pct_change()
    m_ret = mt5["Close"].pct_change()
    lag_corr = {}
    for lag in (-1, 0, 1):
        pair = pd.concat([b_ret.shift(lag), m_ret], axis=1, join="inner").dropna().tail(max(days, 20))
        lag_corr[lag] = float(pair.corr().iloc[0, 1]) if len(pair) > 5 else float("nan")
    best_lag = max(lag_corr, key=lambda k: -1 if pd.isna(lag_corr[k]) else lag_corr[k])

    return {
        "n": len(joined),
        "diffs": diffs,
        "mean": diffs.mean(),
        "median_abs": diffs.abs().median(),
        "max_abs": diffs.abs().max(),
        "lag_corr": lag_corr,
        "best_lag": best_lag,
    }


def print_report(r: dict) -> None:
    print(f"\nCompared {r['n']} overlapping D1 candles (bot - MT5, in dollars)\n")
    print(r["diffs"].round(2).tail(10).to_string())
    print("\nMean difference      :", r["mean"].round(2).to_dict())
    print("Median |difference|  :", r["median_abs"].round(2).to_dict())
    print("Max |difference|     :", r["max_abs"].round(2).to_dict())
    print("Return correlation by lag:", {k: round(v, 3) for k, v in r["lag_corr"].items()})

    if r["best_lag"] != 0:
        print(f"\n[!] Best alignment is lag {r['best_lag']:+d}: candle labels/boundary differ from MT5.")
    else:
        print("\n[ok] Day alignment matches (lag 0).")
    close_med = float(r["median_abs"]["Close"])
    if close_med > 3.0:
        print(f"[!] Median Close gap is ${close_med:.2f}: check symbol (spot vs futures) or broker feed.")
    else:
        print(f"[ok] Median Close gap is ${close_med:.2f}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mt5_csv")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    import yaml

    from src.data_fetcher import get_gold_bundle

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bundle = get_gold_bundle(cfg)
    print("Bot source:", bundle.source_label)
    for w in bundle.warnings:
        print("  warning:", w)
    print_report(compare(bundle.candles, read_mt5_csv(args.mt5_csv), args.days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
