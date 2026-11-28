"""Probe which Yahoo symbols return usable hourly gold data, and (optionally)
how closely each one matches your MT5 D1 candles.

Usage (needs internet):
    python tools/probe_symbols.py
    python tools/probe_symbols.py "C:\\path\\to\\XAUUSD_D1.csv"

Without the CSV it only reports availability. With the CSV it also builds
17:00-New-York D1 candles per symbol and reports the median |Close| gap versus MT5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import yaml  # noqa: E402

from src.data_fetcher import _clean_raw, _download, build_ny_close_candles  # noqa: E402

# Candidates that might carry gold. ETFs (GLD/IAU) are deliberately excluded:
# they only trade during US hours and are not priced in $/oz.
CANDIDATES = ["XAUUSD=X", "XAU=X", "XAUUSD", "GC=F", "MGC=F"]


def probe(symbol: str, days: int = 45) -> dict:
    try:
        raw = _download(symbol, "1h", days)
        hourly = _clean_raw(raw)
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "ok": False, "note": str(exc)[:70]}
    return {
        "symbol": symbol,
        "ok": True,
        "rows": len(hourly),
        "first": hourly.index[0],
        "last": hourly.index[-1],
        "last_close": float(hourly["Close"].iloc[-1]),
        "hourly": hourly,
    }


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tz, hour = cfg["market"]["candle_cutoff_tz"], cfg["market"]["candle_cutoff_hour"]
    mt5 = None
    if len(sys.argv) > 1:
        from compare_mt5 import compare, read_mt5_csv

        mt5 = read_mt5_csv(sys.argv[1])
        print(f"MT5 export: {len(mt5)} candles, last {mt5.index[-1].date()} close {mt5['Close'].iloc[-1]:.2f}\n")

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for sym in CANDIDATES:
        print(f"probing {sym} ...", flush=True)
        r = probe(sym)
        if not r["ok"]:
            rows.append((sym, "NO DATA", "-", "-", "-", "-", r["note"]))
            continue
        gap, lag = "-", "-"
        if mt5 is not None:
            try:
                c = build_ny_close_candles(r["hourly"], tz, hour, now)
                c = c[c["complete"]]
                res = compare(c, mt5, days=20)
                gap = f"{res['median_abs']['Close']:.2f}"
                lag = f"{res['best_lag']:+d}"
            except Exception as exc:  # noqa: BLE001
                gap = f"err: {str(exc)[:30]}"
        rows.append((sym, "ok", r["rows"], r["last"].strftime("%m-%d %H:%M"), f"{r['last_close']:.2f}", gap, lag))

    print("\nsymbol      status   rows  last bar (UTC)  last close  median|dClose| vs MT5  lag")
    for sym, st, n, last, lc, gap, lag in rows:
        print(f"{sym:<10}  {st:<7}  {str(n):>4}  {last:<14}  {lc:>10}  {gap:>20}  {lag}")
    print("\nA good spot candidate: status ok, rows > 500 for 45 days, gap near $1-3, lag +0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
