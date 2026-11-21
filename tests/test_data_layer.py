"""Offline tests for Sprint 1. No network, no pytest-only features.

Run either way:
    pytest -q
    python tests/test_data_layer.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src import data_fetcher as df_mod  # noqa: E402
from src.data_fetcher import (  # noqa: E402
    DataError,
    build_ny_close_candles,
    get_gold_bundle,
    last_complete_label,
    session_label_for,
    target_label,
    validate_candles,
)
from src.email_dispatcher import build_data_check_email, build_error_email  # noqa: E402

TZ, HOUR = "America/New_York", 17
NY = ZoneInfo(TZ)


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def load_cfg() -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["data"]["retries"] = 1          # no sleeping in tests
    cfg["data"]["backoff_seconds"] = 0
    cfg["twelvedata"]["min_interval_seconds"] = 0
    return cfg


def make_hourly(first_sunday: str, weeks: int, seed: int = 0, start_price: float = 2000.0) -> pd.DataFrame:
    """Synthetic 1h bars: Sunday 17:00 NY -> Friday 17:00 NY each week."""
    rng = np.random.default_rng(seed)
    stamps = []
    sun = pd.Timestamp(first_sunday)
    for w in range(weeks):
        start = (sun + pd.Timedelta(weeks=w, hours=17)).tz_localize(NY).tz_convert("UTC")
        end = (sun + pd.Timedelta(weeks=w, days=5, hours=17)).tz_localize(NY).tz_convert("UTC")
        stamps.extend(pd.date_range(start, end, freq="1h", inclusive="left"))
    idx = pd.DatetimeIndex(stamps)
    n = len(idx)
    close = start_price * np.exp(np.cumsum(rng.normal(0, 0.0007, n)))
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0003, n)))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 100.0}, index=idx
    )


# Tue 2026-01-13 22:30 UTC = 17:30 EST. Week starts Sunday 2026-01-11.
NOW = utc("2026-01-13 22:30")
FIRST_SUNDAY = (pd.Timestamp("2026-01-11") - pd.Timedelta(weeks=110)).strftime("%Y-%m-%d")


def hourly_until(now: pd.Timestamp, seed: int = 1) -> pd.DataFrame:
    h = make_hourly(FIRST_SUNDAY, 111, seed=seed)
    return h[h.index <= now]


KEY_ENV = "TWELVEDATA_API_KEY"


class FakeYahoo:
    """Replace data_fetcher._download; also removes any real Twelve Data key from the env."""

    def __init__(self, handler):
        self.handler = handler

    def __enter__(self):
        self._orig = df_mod._download
        self._key = os.environ.pop(KEY_ENV, None)
        df_mod._download = self.handler
        return self

    def __exit__(self, *exc):
        df_mod._download = self._orig
        if self._key is not None:
            os.environ[KEY_ENV] = self._key


class FakeTD:
    """Replace data_fetcher._td_request with a pager over `hourly` (newest bars first)."""

    def __init__(self, hourly=None, error=None):
        self.hourly, self.error, self.calls = hourly, error, []

    def _handler(self, params):
        self.calls.append(params.get("end_date"))
        if self.error:
            raise DataError(self.error)
        end = pd.Timestamp(params["end_date"], tz="UTC")
        sl = self.hourly[self.hourly.index <= end].tail(int(params["outputsize"]))
        if sl.empty:
            return {"code": 400, "message": "no data", "status": "error"}
        vals = [
            {"datetime": ts.strftime("%Y-%m-%d %H:%M:%S"), "open": repr(float(r.Open)),
             "high": repr(float(r.High)), "low": repr(float(r.Low)), "close": repr(float(r.Close))}
            for ts, r in sl.iloc[::-1].iterrows()
        ]
        return {"values": vals, "status": "ok"}

    def __enter__(self):
        self._orig_req, self._orig_dl = df_mod._td_request, df_mod._download
        self._key = os.environ.get(KEY_ENV)
        os.environ[KEY_ENV] = "TEST-KEY"
        df_mod._td_request = self._handler
        df_mod._download = lambda *a, **k: (_ for _ in ()).throw(DataError("yahoo disabled in this test"))
        return self

    def __exit__(self, *exc):
        df_mod._td_request, df_mod._download = self._orig_req, self._orig_dl
        if self._key is None:
            os.environ.pop(KEY_ENV, None)
        else:
            os.environ[KEY_ENV] = self._key


# ---------------------------------------------------------------- sessions
def test_session_label_winter_and_summer_boundary():
    # Winter: 17:00 EST = 22:00 UTC -> new session labelled next day
    assert session_label_for(utc("2026-01-12 21:00"), TZ, HOUR) == pd.Timestamp("2026-01-12")
    assert session_label_for(utc("2026-01-12 22:00"), TZ, HOUR) == pd.Timestamp("2026-01-13")
    # Summer: 17:00 EDT = 21:00 UTC
    assert session_label_for(utc("2026-07-14 20:00"), TZ, HOUR) == pd.Timestamp("2026-07-14")
    assert session_label_for(utc("2026-07-14 21:00"), TZ, HOUR) == pd.Timestamp("2026-07-15")


def test_sunday_open_belongs_to_monday():
    assert session_label_for(utc("2026-01-11 22:00"), TZ, HOUR) == pd.Timestamp("2026-01-12")
    assert session_label_for(utc("2026-01-11 22:00"), TZ, HOUR).weekday() == 0


def test_last_complete_and_target_labels():
    cases = [
        # now (UTC),            last complete,  target
        ("2026-01-13 22:30", "2026-01-13", "2026-01-14"),  # Tue winter
        ("2026-07-14 22:30", "2026-07-14", "2026-07-15"),  # Tue summer
        ("2026-01-11 22:30", "2026-01-09", "2026-01-12"),  # Sunday reopen -> Monday candle
        ("2026-01-10 12:00", "2026-01-09", "2026-01-12"),  # Saturday
        ("2026-01-09 22:30", "2026-01-09", "2026-01-12"),  # Friday after close
        ("2026-01-13 20:00", "2026-01-12", "2026-01-13"),  # Tue before close
    ]
    for now, last, tgt in cases:
        assert last_complete_label(utc(now), TZ, HOUR) == pd.Timestamp(last), now
        assert target_label(utc(now), TZ, HOUR) == pd.Timestamp(tgt), now


# ------------------------------------------------------------- aggregation
def test_candle_aggregation_is_consistent():
    hourly = make_hourly("2025-01-05", 4)
    candles = build_ny_close_candles(hourly, TZ, HOUR, utc("2025-02-10 12:00"))
    assert len(candles) == 20                       # 4 weeks x 5 sessions
    assert (candles.index.dayofweek < 5).all()
    assert candles["bars"].between(23, 25).all()
    assert (candles["High"] >= candles["Low"]).all()
    # first candle's open == first hourly open; last close == last hourly close
    assert candles["Open"].iloc[0] == hourly["Open"].iloc[0]
    assert candles["Close"].iloc[-1] == hourly["Close"].iloc[-1]
    # High/Low are true extremes of the session
    first = candles.iloc[0]
    sess = hourly[session_label_for(hourly.index, TZ, HOUR) == candles.index[0]]
    assert first["High"] == sess["High"].max() and first["Low"] == sess["Low"].min()


def test_dst_weeks_keep_boundary_at_1700_ny():
    # Week containing the US spring-forward (2026-03-08) must still split at 17:00 NY.
    hourly = make_hourly("2026-03-01", 3)
    candles = build_ny_close_candles(hourly, TZ, HOUR, utc("2026-04-01 00:00"))
    assert len(candles) == 15
    for ts in candles["first_ts"]:
        assert ts.tz_convert(TZ).hour == 17         # every session opens 17:00 NY


def test_incomplete_candle_flagged():
    hourly = make_hourly("2026-01-04", 2)
    now = utc("2026-01-13 22:30")
    candles = build_ny_close_candles(hourly[hourly.index <= now], TZ, HOUR, now)
    assert bool(candles["complete"].iloc[-1]) is False   # Wed 14th session in progress
    assert bool(candles["complete"].iloc[-2]) is True


# -------------------------------------------------------------- validation
def test_validation_rejects_broken_candles():
    cfg = load_cfg()
    hourly = hourly_until(NOW)
    all_c = build_ny_close_candles(hourly, TZ, HOUR, NOW)
    complete = all_c[all_c["complete"]].copy()
    expected = last_complete_label(NOW, TZ, HOUR)
    assert validate_candles(complete, cfg, expected) is not None   # baseline passes

    bad = complete.copy()
    bad.iloc[10, bad.columns.get_loc("High")] = bad.iloc[10]["Low"] - 5
    try:
        validate_candles(bad, cfg, expected)
        raise AssertionError("expected DataError for High < Low")
    except DataError:
        pass

    try:
        validate_candles(complete.iloc[:-1], cfg, expected)   # newest candle missing
        raise AssertionError("expected DataError for stale data")
    except DataError as exc:
        assert "Stale" in str(exc)


# ------------------------------------------------------------- integration
def test_bundle_from_twelvedata_spot_with_paging():
    cfg = load_cfg()
    cfg["twelvedata"]["page_size"] = 3000        # force several pages
    hourly = hourly_until(NOW)
    td = FakeTD(hourly)
    with td:
        b = get_gold_bundle(cfg, now_utc=NOW)

    assert b.symbol == "XAU/USD" and not b.is_futures and b.quality == "nyclose_1h"
    assert len(td.calls) >= 2                                   # paged backwards
    assert b.candles.index[-1] == pd.Timestamp("2026-01-13")
    assert b.target_session == pd.Timestamp("2026-01-14")
    assert abs(b.reference_price - hourly["Close"].iloc[-1]) < 1e-6     # string parsing: ~1e-13 noise
    assert abs(b.session_open - hourly["Open"].iloc[-1]) < 1e-6
    assert b.hourly is not None and len(b.hourly) > 1000
    assert "Spot" in b.source_label
    assert not any("failed" in w or "not set" in w for w in b.warnings)


def test_fallback_to_futures_is_labelled():
    cfg = load_cfg()
    hourly = hourly_until(NOW, seed=2)
    td = FakeTD(error="429 Too Many Requests")
    with td:
        df_mod._download = lambda symbol, interval, days: hourly      # Yahoo works
        b = get_gold_bundle(cfg, now_utc=NOW)

    assert b.is_futures and b.symbol == "GC=F"
    assert any("XAU/USD 1h failed" in w for w in b.warnings)
    assert any("Futures" in w for w in b.warnings)
    subject, html_body, text = build_data_check_email(b, cfg)
    assert "Futures" in subject and "Futures" in html_body


def test_missing_key_is_reported_not_fatal():
    cfg = load_cfg()
    hourly = hourly_until(NOW, seed=2)
    with FakeYahoo(lambda s, i, d: hourly):                            # also clears the env key
        b = get_gold_bundle(cfg, now_utc=NOW)
    assert b.is_futures
    assert any("TWELVEDATA_API_KEY not set" in w for w in b.warnings)


def test_twelvedata_key_never_leaks_in_errors():
    import requests
    orig = requests.get

    def boom(*a, **k):
        raise RuntimeError("HTTPError for url: https://api.twelvedata.com/x?apikey=SECRET123")

    requests.get = boom
    try:
        df_mod._td_request({"apikey": "SECRET123"})
        raise AssertionError("expected DataError")
    except DataError as exc:
        assert "SECRET123" not in str(exc) and "***" in str(exc)
    finally:
        requests.get = orig


def test_daily_fallback_uses_exchange_calendar_date():
    cfg = load_cfg()
    days = pd.bdate_range("2024-06-03", "2026-01-13")
    rng = np.random.default_rng(3)
    close = 2000 * np.exp(np.cumsum(rng.normal(0, 0.007, len(days))))
    daily = pd.DataFrame(
        {"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close, "Volume": 0.0},
        index=days.tz_localize("Europe/London"),
    )

    def handler(symbol, interval, days_):
        if interval == "1h":
            raise DataError("no hourly data")
        return daily

    with FakeYahoo(handler):
        b = get_gold_bundle(cfg, now_utc=NOW)

    assert b.quality == "daily_fallback"
    assert b.candles.index[-1] == pd.Timestamp("2026-01-13")   # not shifted back a day
    assert b.target_session == pd.Timestamp("2026-01-14")
    assert any("Daily fallback" in w for w in b.warnings)


def test_all_sources_stale_raises_instead_of_guessing():
    cfg = load_cfg()
    stale = hourly_until(utc("2026-01-13 22:30"))          # data ends Tuesday
    later = utc("2026-01-14 22:30")                          # but it is already Wednesday evening

    def handler(symbol, interval, days):
        if interval == "1h":
            return stale
        raise DataError("daily unavailable")

    with FakeYahoo(handler):
        try:
            get_gold_bundle(cfg, now_utc=later)
            raise AssertionError("expected DataError")
        except DataError as exc:
            assert "All data sources failed" in str(exc)
            assert "Stale" in str(exc) or "No hourly bars" in str(exc)


def test_sunday_run_without_new_bar_uses_friday_close():
    cfg = load_cfg()
    sunday = utc("2026-01-11 22:30")                      # market just reopened
    hourly = hourly_until(sunday)
    hourly = hourly[hourly.index < utc("2026-01-11 22:00")]   # Yahoo has no new-week bar yet
    assert hourly.index[-1] < utc("2026-01-10")

    with FakeYahoo(lambda s, i, d: hourly):
        b = get_gold_bundle(cfg, now_utc=sunday)

    assert b.quality == "nyclose_1h"
    assert b.target_session == pd.Timestamp("2026-01-12")      # Monday candle
    assert b.candles.index[-1] == pd.Timestamp("2026-01-09")   # Friday
    assert b.session_open is None
    assert b.reference_price == b.candles["Close"].iloc[-1]
    assert any("Weekend gap" in w for w in b.warnings)


def test_sunday_run_with_new_bar_uses_monday_open():
    cfg = load_cfg()
    sunday = utc("2026-01-11 22:30")
    hourly = hourly_until(sunday)                              # includes the 22:00 UTC bar
    assert hourly.index[-1] == utc("2026-01-11 22:00")

    with FakeYahoo(lambda s, i, d: hourly):
        b = get_gold_bundle(cfg, now_utc=sunday)

    assert b.target_session == pd.Timestamp("2026-01-12")
    assert b.session_open == hourly["Open"].iloc[-1]
    assert b.reference_price == hourly["Close"].iloc[-1]
    assert not any("Weekend gap" in w for w in b.warnings)


# ------------------------------------------------------------------- email
def test_emails_render_without_nan():
    cfg = load_cfg()
    hourly = hourly_until(NOW)
    with FakeYahoo(lambda s, i, d: hourly):
        b = get_gold_bundle(cfg, now_utc=NOW)
    subject, html_body, text = build_data_check_email(b, cfg)
    assert "TES DATA OK" in subject and "$" in subject
    assert "BUKAN prediksi" in text
    for blob in (subject, html_body, text):
        assert "nan" not in blob.lower().replace("finansial", "")
    e_subject, e_html, e_text = build_error_email("boom <b>", cfg)
    assert "TIDAK TERSEDIA" in e_subject and "&lt;b&gt;" in e_html


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
