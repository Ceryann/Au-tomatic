"""Task A1 probe: how many hourly bars does Twelve Data give on US-holiday sessions?

Counts bars only (no price levels are printed). Uses ~4 API credits.

Setup (PowerShell, from the au-tomatic folder, venv active):
    $env:TWELVEDATA_API_KEY = "paste-your-key-here"      # never paste the key into chat
    python tools/probe_holidays.py

Read the output, then adjust `data.holidays` / `data.holiday_min_bars` in config.yaml.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dateutil.easter import easter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fetcher import (  # noqa: E402
    DataError, build_ny_close_candles, fetch_twelvedata_hourly, holiday_map,
)


def us_holiday_dates(start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, str]:
    """Candidate US holidays in [start, end]: federal (observed) + Good Friday + Dec 24 + Dec 31."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    cal = USFederalHolidayCalendar()
    out = {d.normalize(): n for d, n in cal.holidays(start, end, return_name=True).items()}
    for year in range(start.year, end.year + 1):
        gf = pd.Timestamp(easter(year)) - pd.Timedelta(days=2)
        out[gf] = "Good Friday"
        out[pd.Timestamp(year=year, month=12, day=24)] = "Christmas Eve"
        out[pd.Timestamp(year=year, month=12, day=31)] = "New Year's Eve"
    return {d: n for d, n in out.items() if start <= d <= end and d.weekday() < 5}


def next_weekday(d: pd.Timestamp) -> pd.Timestamp:
    d += pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def holiday_report(candles: pd.DataFrame) -> list[tuple[str, str, str, object]]:
    """Rows (session, holiday, which, bars or 'ABSENT') for each holiday session and the next one."""
    rows = []
    for day, name in sorted(us_holiday_dates(candles.index[0], candles.index[-1]).items()):
        for which, label in (("holiday", day), ("next", next_weekday(day))):
            bars = int(candles.loc[label, "bars"]) if label in candles.index else "ABSENT"
            rows.append((label.strftime("%Y-%m-%d %a"), name, which, bars))
    return rows


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    key = os.environ.get(cfg["twelvedata"]["api_key_env"], "").strip()
    if not key:
        print(f"Set ${cfg['twelvedata']['api_key_env']} first (see the docstring).")
        return 2
    now = pd.Timestamp.now(tz="UTC")
    try:
        hourly = fetch_twelvedata_hourly(cfg, now, key)
    except DataError as exc:
        print(f"Fetch failed: {exc}")
        return 1
    m = cfg["market"]
    c = build_ny_close_candles(hourly, m["candle_cutoff_tz"], m["candle_cutoff_hour"], now)
    c = c[c["complete"]]
    print(f"{len(c)} complete sessions, {c.index[0].date()} -> {c.index[-1].date()}")
    print("\nBars per session, all history (normal gold day ~ 23):")
    print(c["bars"].describe().round(1).to_string())

    print("\nHoliday sessions (label = NY date the candle closes) and the session after:")
    print(f"{'session':<16}{'holiday':<24}{'which':<9}bars")
    for row in holiday_report(c):
        print(f"{row[0]:<16}{row[1]:<24}{row[2]:<9}{row[3]}")

    print("\n12 thinnest sessions overall (to spot thin days that are NOT on this list):")
    for label, bars in c["bars"].nsmallest(12).items():
        print(f"  {label.date()} {label.day_name()[:3]}  {int(bars)} bars")

    full = pd.bdate_range(c.index[0], c.index[-1]).difference(c.index)
    print(f"\nWeekdays with no candle at all: {len(full)}")
    print("  " + ", ".join(d.strftime('%Y-%m-%d') for d in full[:40]))

    listed = holiday_map(cfg)
    print(f"\nconfig.yaml currently lists {len(listed)} holiday session(s); "
          f"holiday_min_bars = {cfg['data'].get('holiday_min_bars')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
