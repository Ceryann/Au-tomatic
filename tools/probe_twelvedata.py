"""Test whether Twelve Data gives true spot XAU/USD hourly bars on a FREE key,
and how closely 17:00-New-York D1 candles built from them match your MT5 export.

Setup (PowerShell, from the au-tomatic folder, venv active):
    $env:TWELVEDATA_API_KEY = "paste-your-key-here"
    python tools/probe_twelvedata.py "C:\\Users\\...\\XAUUSD_D1.csv"

NEVER paste the key into chat, code, or a commit. It only lives in this terminal session
(and later in GitHub Secrets).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from compare_mt5 import compare, print_report, read_mt5_csv  # noqa: E402
from src.data_fetcher import _clean_raw, build_ny_close_candles  # noqa: E402

URL = "https://api.twelvedata.com/time_series"


def fetch_1h(
    api_key: str, symbol: str = "XAU/USD", outputsize: int = 5000, end_date: str | None = None
) -> pd.DataFrame:
    """Up to `outputsize` hourly bars (UTC) ending at `end_date` (default: now).

    5000 bars is roughly 7 months of gold, so older history needs several pages.
    """
    params = {
        "symbol": symbol, "interval": "1h", "outputsize": outputsize,
        "timezone": "UTC", "format": "JSON", "apikey": api_key,
    }
    if end_date:
        params["end_date"] = end_date
    try:
        resp = requests.get(URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # mask the key if it appears in the error text
        raise RuntimeError(str(exc).replace(api_key, "***")) from None

    if data.get("status") == "error" or "values" not in data:
        raise RuntimeError(f"Twelve Data says: code={data.get('code')} message={data.get('message')}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close"]].astype(float)
    df["Volume"] = 0.0
    return df


def fetch_1h_paged(api_key: str, pages: int = 4, symbol: str = "XAU/USD") -> tuple[pd.DataFrame, list[str]]:
    """Walk backwards page by page. Returns (bars, notes). Stops cleanly if a page fails."""
    frames, notes, end = [], [], None
    for i in range(pages):
        try:
            df = fetch_1h(api_key, symbol=symbol, end_date=end)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"page {i + 1} stopped: {exc}")
            break
        if df.empty:
            break
        frames.append(df)
        if len(df) < 5000:
            notes.append(f"page {i + 1} returned {len(df)} bars: start of available history reached.")
            break
        end = (df.index[0] - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
        if i < pages - 1:
            time.sleep(1)  # stay far below the 8 requests/minute limit
    if not frames:
        raise RuntimeError("; ".join(notes) or "no data returned")
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")], notes


def main() -> int:
    key = os.environ.get("TWELVEDATA_API_KEY")
    if not key:
        print('Set the key first:  $env:TWELVEDATA_API_KEY = "your-key"')
        return 2

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tz, hour = cfg["market"]["candle_cutoff_tz"], cfg["market"]["candle_cutoff_hour"]

    try:
        raw, notes = fetch_1h_paged(key, pages=4)
        hourly = _clean_raw(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")
        print("If the message mentions your plan/symbol, XAU/USD hourly may not be on the free plan.")
        return 1

    span_days = (hourly.index[-1] - hourly.index[0]).days
    print(f"Twelve Data XAU/USD 1h: {len(hourly)} bars, {hourly.index[0]} -> {hourly.index[-1]}")
    print(f"History depth: {span_days} days (~{span_days / 365:.1f} years). Target: ~700 days.")
    for n in notes:
        print("note:", n)
    print(f"Last close: {hourly['Close'].iloc[-1]:.2f}")

    if len(sys.argv) > 1:
        mt5 = read_mt5_csv(sys.argv[1])
        candles = build_ny_close_candles(hourly, tz, hour, pd.Timestamp.now(tz="UTC"))
        candles = candles[candles["complete"]]
        print_report(compare(candles, mt5, days=30))
        print("\nGood sign: lag +0 and median Close gap near $1-3 (spot vs your broker).")
    else:
        print("Add your MT5 CSV path as an argument to compare with your broker candles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
