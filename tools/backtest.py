"""Walk-forward backtest on REAL history. Takes about a minute and uses 3-4 Twelve Data credits.

PowerShell (venv active, from the au-tomatic folder):
    $env:TWELVEDATA_API_KEY = "your-key"
    python tools/backtest.py --sessions 120

Options:
    --sessions N   how many past sessions to evaluate (default 120)
    --run-after-open-h H   hours after the candle opened to forecast at (default 0.5; try 6 for a delayed cron)
    --yahoo        use Yahoo GC=F futures instead (no API key needed; levels are futures)
    --macro        also use DXY / US10Y features (needs Yahoo; slower)
    --set K=V      override a config value, repeatable, e.g.
                   --set model.drift.k_kalman=1 --set model.drift.cap_sigma_mult=10
                   (full trend extrapolation: tests whether extrapolating the Kalman trend helps)

Read the report like this:
  * "inside" should be near 70%. 60-80% is acceptable for ~120 sessions.
  * Direction Brier close to 0.25 means no directional skill - that is NORMAL for daily gold.
  * Pinball gain vs baseline > 0 means the model beats a plain random walk.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fetcher import (  # noqa: E402
    DataError,
    _clean_raw,
    _download,
    _with_retry,
    fetch_macro_daily,
    fetch_twelvedata_hourly,
)
from src.evaluation import run_backtest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, default=120)
    ap.add_argument("--run-after-open-h", type=float, default=0.5,
                    help="Hours after the candle opened at which each session is forecast (default 0.5, the cron's nominal time; use 6 to mimic the 5-6 hour delay GitHub adds to scheduled runs).")
    ap.add_argument("--yahoo", action="store_true")
    ap.add_argument("--macro", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="K=V", help="Override config, e.g. model.drift.k_kalman=1")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "out"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for item in args.set:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            print(f"Unknown config key: {key}")
            return 2
        node[parts[-1]] = yaml.safe_load(raw)
        print(f"override: {key} = {node[parts[-1]]}")
    now = pd.Timestamp.now(tz="UTC")

    try:
        if args.yahoo:
            sym = cfg["market"]["yahoo_symbols"][0]
            raw = _with_retry(lambda: _download(sym, "1h", 725), 3, 5, f"{sym} 1h")
            hourly, symbol, fut = _clean_raw(raw), sym, True
        else:
            key = os.environ.get(cfg["twelvedata"]["api_key_env"], "").strip()
            if not key:
                print('Set the key first:  $env:TWELVEDATA_API_KEY = "your-key"   (or use --yahoo)')
                return 2
            hourly = _clean_raw(fetch_twelvedata_hourly(cfg, now, key))
            symbol, fut = cfg["market"]["twelvedata_symbol"], False
    except DataError as exc:
        print(f"Data download failed: {exc}")
        return 1

    macro = None
    if args.macro:
        macro, notes = fetch_macro_daily(cfg, now)
        for n in notes:
            print("note:", n)

    span = (hourly.index[-1] - hourly.index[0]).days
    print(f"{symbol}: {len(hourly)} hourly bars, {span} days. Evaluating the last {args.sessions} sessions...")
    report = run_backtest(
        hourly, cfg, n_sessions=args.sessions, symbol=symbol, is_futures=fut, macro=macro, progress=True,
        run_after_open_h=args.run_after_open_h,
    )
    print()
    print("\n".join(report.lines()))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report.table.to_csv(out / "backtest_table.csv", index=False)
    print(f"\nPer-session table saved to {out / 'backtest_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
