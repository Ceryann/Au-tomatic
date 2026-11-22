"""Event study: how much more variable is gold on FOMC decision days? (uses ~4 Twelve Data credits)

PowerShell (venv active, from the au-tomatic folder):
    $env:TWELVEDATA_API_KEY = "your-key"
    python tools/event_study.py

It compares close-to-close variance on FOMC decision days (2024-2026) with all other days, gives a
bootstrap 95% interval, and shows the `model.jumps.event_extra_var` that matches it. With only ~20 event days the interval is wide: treat the result as a sanity check, not a proof.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fetcher import DataError, _clean_raw, build_ny_close_candles, fetch_twelvedata_hourly  # noqa: E402
from src.evaluation import FOMC_DECISION_DAYS, event_study  # noqa: E402


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    key = os.environ.get(cfg["twelvedata"]["api_key_env"], "").strip()
    if not key:
        print('Set the key first:  $env:TWELVEDATA_API_KEY = "your-key"')
        return 2
    now = pd.Timestamp.now(tz="UTC")
    try:
        hourly = _clean_raw(fetch_twelvedata_hourly(cfg, now, key))
        tz, hour = cfg["market"]["candle_cutoff_tz"], cfg["market"]["candle_cutoff_hour"]
        c = build_ny_close_candles(hourly, tz, hour, now)
        candles = c[c["complete"]]
        dates = [d for d in FOMC_DECISION_DAYS if pd.Timestamp(d) <= candles.index[-1]]
        res = event_study(candles, dates, cfg)
    except DataError as exc:
        print(f"Failed: {exc}")
        return 1

    lo, hi = res["ci"]
    print(f"History: {candles.index[0].date()} -> {candles.index[-1].date()} ({res['n_normal'] + res['n_event']} sessions)")
    print(f"FOMC decision days found: {res['n_event']}  |  normal days: {res['n_normal']}\n")
    print(f"Variance ratio (FOMC day / normal day): {res['var_ratio']:.2f}   95% interval {lo:.2f} to {hi:.2f}")
    print(f"Mean High-Low range ratio             : {res['range_ratio']:.2f}\n")
    cur = res["configured_extra_var"]
    rec = res["recommended_extra_var"]
    elo, ehi = res["extra_var_ci"]
    print(f"Configured model.jumps.event_extra_var = {cur:.2f}  (event days {1 + cur:.2f}x as variable)")
    print(f"Observed value implies event_extra_var  = {rec:.2f}  (95% interval {elo:.2f} to {ehi:.2f})\n")
    if lo <= res["configured_ratio"] <= hi:
        print("[ok] The configured value is consistent with the FOMC-day data.")
        print("     With so few event days the interval is wide: keep the default unless it is far off.")
    elif res["configured_ratio"] < lo:
        print(f"[!] FOMC days look MORE variable than the model assumes. Consider event_extra_var: {rec:.2f}")
    else:
        print(f"[!] FOMC days look LESS variable than the model assumes. Consider event_extra_var: {rec:.2f}")
    if lo <= 1.0:
        print("Note: the interval includes 1.0, so 'FOMC days are more variable' is not statistically clear yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
