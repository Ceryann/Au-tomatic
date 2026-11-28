"""Offline tests for the forecast ledger (Sprint 3). Run: pytest -q   or   python tests/test_ledger.py"""

from __future__ import annotations

import email
import os
import smtplib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_data_layer as T  # noqa: E402
import test_models as M  # noqa: E402
from src.data_fetcher import get_gold_bundle  # noqa: E402
from src.ledger import (  # noqa: E402
    COLUMNS,
    FLOAT_COLS,
    empty_ledger,
    live_accuracy,
    load_ledger,
    save_ledger,
    settle,
    upsert_forecast,
)
from src.quant_engine import forecast_day  # noqa: E402


def _forecast(now=None, seed=11, hourly=None):
    cfg = M.fast_cfg()
    now = now or T.NOW
    hourly = T.hourly_until(now, seed=seed) if hourly is None else hourly
    b = M.bundle_at(now, hourly, cfg)
    return cfg, b, forecast_day(b, None, cfg)


def _with_target_candle(candles, target, open_, high, low, close):
    out = candles.copy()
    out.loc[pd.Timestamp(target), ["Open", "High", "Low", "Close", "Volume", "bars"]] = [open_, high, low, close, 0.0, 23]
    return out.sort_index()


# ------------------------------------------------------------ privacy by design
def test_ledger_stores_no_prices_only_fractions_and_flags():
    cfg, b, fc = _forecast()
    led = upsert_forecast(empty_ledger(), fc, b, b.now_utc)
    assert list(led.columns) == COLUMNS
    for c in FLOAT_COLS:
        assert abs(float(led.iloc[0][c])) < 1.5, c               # fractions/probabilities, never prices like 2000+
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "l.csv"
        save_ledger(led, p)
        text = p.read_text(encoding="utf-8")
    import re
    assert not re.search(r"\b\d{4}\.\d{1,2}\b", text), text      # nothing like 2069.14
    assert "close_hit" in text and ",spot," not in text           # this test world is futures-labelled
    assert led.iloc[0]["session"] == str(b.target_session.date())


def test_rerun_for_the_same_session_keeps_the_first_row_unless_replace_is_asked():
    """GitHub starts the scheduled run hours late and the user may have run it by hand in the morning:
    the live accuracy must keep the forecast made close to the open, not the last one."""
    import dataclasses
    cfg, b, fc = _forecast()
    led = upsert_forecast(empty_ledger(), fc, b, b.now_utc, elapsed_h=0.5)
    later = b.now_utc + pd.Timedelta(hours=7)
    kept = upsert_forecast(led, fc, b, later, elapsed_h=7.5)
    assert len(kept) == 1 and kept.iloc[0]["created_utc"].endswith("22:30Z") and kept.iloc[0]["elapsed_h"] == 0.5
    forced = upsert_forecast(led, fc, b, later, elapsed_h=7.5, replace=True)
    assert len(forced) == 1 and forced.iloc[0]["created_utc"].endswith("05:30Z") and forced.iloc[0]["elapsed_h"] == 7.5
    tomorrow = dataclasses.replace(b, target_session=b.target_session + pd.Timedelta(days=1))
    both = upsert_forecast(led, fc, tomorrow, later, elapsed_h=0.5)
    assert len(both) == 2, "a different session must still be added"


def test_elapsed_hours_are_stored_and_old_ledgers_without_the_column_still_load():
    cfg, b, fc = _forecast()
    led = upsert_forecast(empty_ledger(), fc, b, b.now_utc, elapsed_h=0.504)
    assert led.iloc[0]["elapsed_h"] == 0.5                                   # rounded to 2 decimals
    assert np.isnan(upsert_forecast(empty_ledger(), fc, b, b.now_utc).iloc[0]["elapsed_h"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "l.csv"
        save_ledger(led, path)
        assert load_ledger(path).iloc[0]["elapsed_h"] == 0.5
        led.drop(columns=["elapsed_h"]).to_csv(path, index=False)            # a file written before the column existed
        back = load_ledger(path)
        assert list(back.columns) == COLUMNS and np.isnan(back.iloc[0]["elapsed_h"])


def test_live_accuracy_counts_runs_by_timing():
    elapsed = [0.5, 0.5, 6.0, 7.5, np.nan, 1.99]
    rows = [{c: np.nan for c in COLUMNS} | {
        "session": f"2026-01-{i + 1:02d}", "settled": 1, "close_hit": 1.0, "high_hit": 1.0, "low_hit": 1.0,
        "p_up": 0.5, "up_actual": 1.0, "elapsed_h": e} for i, e in enumerate(elapsed)]
    led = pd.DataFrame(rows)
    acc = live_accuracy(led, window=30, min_n=3, late_after=2.0)
    assert (acc["n_ontime"], acc["n_late"], acc["n_unknown"]) == (3, 2, 1) and acc["n"] == 6
    acc = live_accuracy(led, window=30, min_n=3, late_after=7.0)
    assert (acc["n_ontime"], acc["n_late"], acc["n_unknown"]) == (4, 1, 1)
    none = live_accuracy(empty_ledger(), 30, 10)
    assert (none["n_ontime"], none["n_late"], none["n_unknown"]) == (0, 0, 0)


# ------------------------------------------------------------------ settlement
def test_settle_scores_hits_and_misses_correctly_when_open_is_known():
    cfg, b, fc = _forecast()
    assert fc.open_known
    led = upsert_forecast(empty_ledger(), fc, b, b.now_utc)
    base = b.session_open
    row = led.iloc[0]
    hit = _with_target_candle(b.candles, b.target_session, base,
                              base * (1 + row.h50), base * (1 + row.l50), base * (1 + row.c50))
    settled, n = settle(led, hit)
    r = settled.iloc[0]
    assert n == 1 and r.settled == 1
    assert (r.close_hit, r.high_hit, r.low_hit) == (1.0, 1.0, 1.0)

    miss = _with_target_candle(b.candles, b.target_session, base,
                               base * (1 + row.h85 + 0.01), base * (1 + row.l15 - 0.01),
                               base * (1 + row.c85 + 0.01))
    r2 = settle(led, miss)[0].iloc[0]
    assert (r2.close_hit, r2.high_hit, r2.low_hit) == (0.0, 0.0, 0.0)
    assert r2.up_actual == 1.0                                   # closed above its base


def test_settle_uses_previous_close_as_base_on_weekend_rows():
    cfg = M.fast_cfg()
    sunday = T.utc("2026-01-11 22:30")
    hourly = T.hourly_until(sunday, seed=11)
    hourly = hourly[hourly.index < T.utc("2026-01-11 22:00")]    # no new-week bar: open unknown
    b = M.bundle_at(sunday, hourly, cfg)
    fc = forecast_day(b, None, cfg)
    assert not fc.open_known
    led = upsert_forecast(empty_ledger(), fc, b, b.now_utc)
    fri_close = float(b.candles["Close"].iloc[-1])
    row = led.iloc[0]
    # Monday's candle opens with a gap, but the base for this row must still be FRIDAY'S CLOSE
    cand = _with_target_candle(b.candles, b.target_session, fri_close * 1.05,
                               fri_close * (1 + row.h50), fri_close * (1 + row.l50), fri_close * (1 + row.c50))
    r = settle(led, cand)[0].iloc[0]
    assert (r.close_hit, r.high_hit, r.low_hit) == (1.0, 1.0, 1.0)


def test_unfinished_sessions_stay_unsettled():
    cfg, b, fc = _forecast()
    led = upsert_forecast(empty_ledger(), fc, b, b.now_utc)
    same, n = settle(led, b.candles)                             # target session not in candles yet
    assert n == 0 and same.iloc[0]["settled"] == 0


# ------------------------------------------------------------- live accuracy
def test_live_accuracy_window_and_ready_flag():
    rows = []
    for i in range(40):
        rows.append({c: np.nan for c in COLUMNS} | {
            "session": f"2026-01-{(i % 28) + 1:02d}" if i < 28 else f"2026-02-{i - 27:02d}",
            "settled": 1, "close_hit": 1.0 if i % 4 else 0.0, "high_hit": 1.0, "low_hit": 0.0,
            "p_up": 0.5, "up_actual": 1.0 if i % 2 else 0.0,
        })
    led = pd.DataFrame(rows)
    acc = live_accuracy(led, window=30, min_n=10)
    assert acc["n"] == 30 and acc["ready"] and acc["high"] == 1.0 and acc["low"] == 0.0
    assert abs(acc["brier"] - 0.25) < 1e-9
    few = live_accuracy(led.head(4), window=30, min_n=10)
    assert few["n"] == 4 and not few["ready"]
    none = live_accuracy(empty_ledger(), 30, 10)
    assert none["n"] == 0 and not none["ready"]


def test_csv_roundtrip_and_bad_files():
    cfg, b, fc = _forecast()
    led = upsert_forecast(empty_ledger(), fc, b, b.now_utc)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x" / "l.csv"
        save_ledger(led, p)
        back = load_ledger(p)
        assert len(back) == 1 and abs(back.iloc[0]["c85"] - led.iloc[0]["c85"]) < 1e-6
        assert load_ledger(Path(tmp) / "missing.csv").empty
        bad = Path(tmp) / "bad.csv"
        bad.write_text("a,b\n1,2\n", encoding="utf-8")
        try:
            load_ledger(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


# ----------------------------------------------------------- end-to-end flow
def _text_of_last_mail() -> str:
    msg = email.message_from_string(T._FakeSMTP.last["message"])
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            return part.get_payload(decode=True).decode("utf-8")
    raise AssertionError("no text part")


def _run_main(args, hourly, env_extra=None):
    import main as main_mod

    env = {"GMAIL_USER": "bot@gmail.com", "GMAIL_APP_PASSWORD": "pw", "EMAIL_TO": "me@gmail.com, me2@gmail.com"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    orig = smtplib.SMTP_SSL
    smtplib.SMTP_SSL = T._FakeSMTP
    try:
        with T.FakeYahoo(lambda s, i, d: hourly):
            return main_mod.main(args)
    finally:
        smtplib.SMTP_SSL = orig
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_two_day_flow_records_then_settles_and_reports_accuracy():
    day1, day2 = T.utc("2026-01-13 22:30"), T.utc("2026-01-14 22:30")
    hourly = T.hourly_until(day2, seed=11)
    with tempfile.TemporaryDirectory() as tmp:
        led_path = str(Path(tmp) / "ledger.csv")
        base_args = ["--no-macro", "--ledger", led_path]

        # a dry run must never write the ledger
        assert _run_main(["--dry-run", "--out", tmp, "--now", "2026-01-13T22:30:00Z"] + base_args, hourly) == 0
        assert not Path(led_path).exists()

        T._FakeSMTP.last = {}
        assert _run_main(["--now", "2026-01-13T22:30:00Z"] + base_args, hourly) == 0
        led = load_ledger(led_path)
        assert len(led) == 1 and led.iloc[0]["settled"] == 0
        assert T._FakeSMTP.last["recipients"] == ["me@gmail.com", "me2@gmail.com"]

        assert _run_main(["--now", "2026-01-14T22:30:00Z"] + base_args, hourly) == 0
        led = load_ledger(led_path)
        assert len(led) == 2
        first = led.sort_values("session").iloc[0]
        assert first["settled"] == 1 and first["close_hit"] in (0.0, 1.0)
        assert "Akurasi live: mengumpulkan data (1/10 hari)" in _text_of_last_mail()


def test_main_keeps_the_on_time_row_when_a_late_run_follows_unless_replace_is_given():
    hourly = T.hourly_until(T.utc("2026-01-14 05:30"), seed=11)
    with tempfile.TemporaryDirectory() as tmp:
        led_path = str(Path(tmp) / "ledger.csv")
        base = ["--no-macro", "--no-news", "--ledger", led_path]

        assert _run_main(["--now", "2026-01-13T22:30:00Z"] + base, hourly) == 0          # the morning run
        row = load_ledger(led_path).iloc[0]
        assert row["created_utc"] == "2026-01-13T22:30Z" and row["elapsed_h"] == 0.5

        T._FakeSMTP.last = {}
        assert _run_main(["--now", "2026-01-14T05:30:00Z"] + base, hourly) == 0          # same session, 7 h later
        led = load_ledger(led_path)
        assert len(led) == 1 and led.iloc[0]["created_utc"] == "2026-01-13T22:30Z", "the on-time row must survive"
        assert led.iloc[0]["elapsed_h"] == 0.5
        assert T._FakeSMTP.last["recipients"], "the late run still sends its email"
        assert "Run terlambat" in _text_of_last_mail()

        assert _run_main(["--now", "2026-01-14T05:30:00Z", "--replace-ledger"] + base, hourly) == 0
        led = load_ledger(led_path)
        assert len(led) == 1 and led.iloc[0]["created_utc"] == "2026-01-14T05:30Z" and led.iloc[0]["elapsed_h"] == 7.5


def test_a_broken_ledger_never_blocks_the_email():
    hourly = T.hourly_until(T.NOW, seed=11)
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "ledger.csv"
        bad.write_text("garbage,not,a,ledger\n1,2,3,4\n", encoding="utf-8")
        T._FakeSMTP.last = {}
        rc = _run_main(["--no-macro", "--ledger", str(bad), "--now", "2026-01-13T22:30:00Z"], hourly)
        assert rc == 0 and T._FakeSMTP.last.get("recipients")          # email still went out
        assert bad.read_text(encoding="utf-8").startswith("garbage")   # and the bad file was not overwritten


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
