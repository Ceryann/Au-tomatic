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
from src import news_sentinel as ns_mod  # noqa: E402
from src.data_fetcher import (  # noqa: E402
    DataError,
    MarketClosedError,
    build_ny_close_candles,
    drop_empty_holiday_sessions,
    get_gold_bundle,
    last_complete_label,
    session_label_for,
    target_label,
    validate_candles,
)
from src.email_dispatcher import (  # noqa: E402
    build_data_check_email,
    build_error_email,
    build_market_closed_email,
)

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


def _offline_http(url, timeout=0):
    raise OSError("network disabled in tests")


class FakeYahoo:
    """Replace data_fetcher._download; also removes any real Twelve Data key from the env."""

    def __init__(self, handler):
        self.handler = handler

    def __enter__(self):
        self._orig = df_mod._download
        self._orig_http = ns_mod._http_get
        self._key = os.environ.pop(KEY_ENV, None)
        df_mod._download = self.handler
        ns_mod._http_get = _offline_http                       # tests must never touch the network
        return self

    def __exit__(self, *exc):
        df_mod._download = self._orig
        ns_mod._http_get = self._orig_http
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
        self._orig_http = ns_mod._http_get
        ns_mod._http_get = _offline_http
        return self

    def __exit__(self, *exc):
        ns_mod._http_get = self._orig_http
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


# --------------------------------------------------------------- recipients
def test_parse_recipients_handles_real_world_input():
    from src.email_dispatcher import parse_recipients

    assert parse_recipients("a@gmail.com, b@gmail.com") == ["a@gmail.com", "b@gmail.com"]
    assert parse_recipients("a@gmail.com;b@gmail.com,") == ["a@gmail.com", "b@gmail.com"]
    assert parse_recipients(" A@Gmail.com a@gmail.com\n<c@x.org> ") == ["A@Gmail.com", "c@x.org"]
    assert parse_recipients("not-an-email, , @, x@y") == []
    assert parse_recipients("") == [] and parse_recipients(None) == []


class _FakeSMTP:
    refuse: dict = {}
    last: dict = {}

    def __init__(self, host, port, context=None, timeout=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        _FakeSMTP.last["login"] = (user, password)

    def sendmail(self, sender, recipients, message):
        _FakeSMTP.last.update(sender=sender, recipients=list(recipients), message=message)
        return dict(_FakeSMTP.refuse)


def _send_with_fake(to, **kw):
    import smtplib
    from src.email_dispatcher import send_email

    orig = smtplib.SMTP_SSL
    smtplib.SMTP_SSL = _FakeSMTP
    try:
        return send_email("Subj", "<b>x</b>", "x", user="bot@gmail.com", password="pw", to=to, **kw)
    finally:
        smtplib.SMTP_SSL = orig


def test_send_email_reaches_every_recipient_and_hides_addresses():
    _FakeSMTP.refuse, _FakeSMTP.last = {}, {}
    ok, bad = _send_with_fake("one@gmail.com, two@gmail.com")
    assert (ok, bad) == (2, 0)
    assert _FakeSMTP.last["recipients"] == ["one@gmail.com", "two@gmail.com"]
    header_block = _FakeSMTP.last["message"].split("\n\n", 1)[0]
    assert "one@gmail.com" not in header_block and "two@gmail.com" not in header_block   # hidden
    assert "To: bot@gmail.com" in header_block


def test_send_email_can_show_recipients_when_configured():
    _FakeSMTP.refuse, _FakeSMTP.last = {}, {}
    _send_with_fake("one@gmail.com, two@gmail.com", hide_recipients=False)
    header_block = _FakeSMTP.last["message"].split("\n\n", 1)[0]
    assert "one@gmail.com" in header_block and "two@gmail.com" in header_block


def test_partial_refusal_is_reported_not_swallowed():
    _FakeSMTP.refuse, _FakeSMTP.last = {"two@gmail.com": (550, b"no such user")}, {}
    ok, bad = _send_with_fake("one@gmail.com, two@gmail.com")
    assert (ok, bad) == (1, 1)
    _FakeSMTP.refuse = {}


def test_no_valid_recipient_raises_clear_error():
    try:
        _send_with_fake(" , ; ")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "no valid email" in str(exc)


# ------------------------------------------------- SMTP resilience (Gmail dropped login from CI)
class _FlakySMTP:
    """Fake SMTP whose behaviour is scripted per class: fail_logins = list of exceptions to raise."""
    events: list = []
    fail_logins: list = []
    last: dict = {}

    def __init__(self, host, port, context=None, timeout=None):
        self.port = port
        self.mode = "ssl" if context is not None and not hasattr(self, "_tls") else "plain"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo(self):
        pass

    def starttls(self, context=None):
        self._tls = True
        self.mode = "starttls"

    def login(self, user, password):
        _FlakySMTP.events.append(self.port)
        _FlakySMTP.last["login"] = (user, password)
        if _FlakySMTP.fail_logins and self.port in (465, 587):
            exc = _FlakySMTP.fail_logins[0]
            if exc[1] == "always" or exc[1] == self.port:
                if exc[1] != "always":
                    _FlakySMTP.fail_logins.pop(0)
                raise exc[0]

    def sendmail(self, sender, recipients, message):
        _FlakySMTP.last["recipients"] = list(recipients)
        return {}


def _send_flaky(fail_logins, **kw):
    import smtplib
    from src.email_dispatcher import send_email

    _FlakySMTP.events, _FlakySMTP.fail_logins, _FlakySMTP.last = [], list(fail_logins), {}
    sleeps: list = []
    o1, o2 = smtplib.SMTP_SSL, smtplib.SMTP
    smtplib.SMTP_SSL = smtplib.SMTP = _FlakySMTP
    try:
        kw.setdefault("password", "pw")
        res = send_email("S", "<b>x</b>", "x", user="bot@gmail.com", to="a@gmail.com, b@gmail.com",
                         sleep=sleeps.append, **kw)
        return res, sleeps
    except Exception as exc:
        return exc, sleeps
    finally:
        smtplib.SMTP_SSL, smtplib.SMTP = o1, o2


def test_smtp_disconnect_is_retried_and_then_succeeds():
    import smtplib
    drop = (smtplib.SMTPServerDisconnected("Connection unexpectedly closed"), 465)
    res, sleeps = _send_flaky([drop, drop], fallback_starttls=False)     # first two SSL logins drop
    assert res == (2, 0), res
    assert _FlakySMTP.events == [465, 465, 465] and sleeps == [5.0, 15.0]


def test_smtp_falls_back_to_starttls_587_when_ssl_drops():
    import smtplib
    res, sleeps = _send_flaky([(smtplib.SMTPServerDisconnected("closed"), 465)])
    assert res == (2, 0), res
    assert _FlakySMTP.events == [465, 587] and sleeps == []              # no pause needed: fallback worked


def test_smtp_auth_failure_is_not_retried():
    import smtplib
    res, sleeps = _send_flaky([(smtplib.SMTPAuthenticationError(535, b"bad credentials"), "always")])
    assert isinstance(res, smtplib.SMTPAuthenticationError)
    assert _FlakySMTP.events == [465] and sleeps == []


def test_smtp_gives_up_after_retries_and_raises_last_error():
    import smtplib
    res, sleeps = _send_flaky([(smtplib.SMTPServerDisconnected("closed"), "always")], retries=3, retry_delay_s=5)
    assert isinstance(res, smtplib.SMTPServerDisconnected)
    assert len(_FlakySMTP.events) == 6                    # 3 attempts x (SSL + STARTTLS)
    assert sleeps == [5, 15]                              # growing pause between attempts


def test_smtp_credentials_are_normalised():
    res, _ = _send_flaky([], password=" abcd efgh\u00a0ijkl mnop\n")
    assert res == (2, 0)
    assert _FlakySMTP.last["login"] == ("bot@gmail.com", "abcdefghijklmnop")


# ------------------------------------------------- Gmail API transport (HTTPS)
class _Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body


class _ScriptedHTTP:
    """Fake requests.post: routes by URL; `send_script` is a list of responses/exceptions for send calls."""
    def __init__(self, token=None, send_script=None):
        self.token = token or _Resp(200, {"access_token": "AT-123"})
        self.send_script = list(send_script or [])
        self.calls: list = []

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        if url.endswith("/token"):
            return self.token
        item = self.send_script.pop(0) if self.send_script else _Resp(200, {"id": "m1"})
        if isinstance(item, Exception):
            raise item
        return item


def _api_send(http, **kw):
    from src.email_dispatcher import send_email_gmail_api
    sleeps: list = []
    kw.setdefault("to", "one@gmail.com, two@gmail.com")
    try:
        res = send_email_gmail_api("Subj", "<b>x</b>", "x", client_id="cid", client_secret="csec",
                                   refresh_token="RT-secret", sleep=sleeps.append, post=http, **kw)
    except Exception as exc:  # noqa: BLE001
        res = exc
    return res, sleeps


def _decode_raw(call):
    import base64
    return base64.urlsafe_b64decode(call[1]["json"]["raw"]).decode("utf-8", "replace")


def test_gmail_api_sends_one_private_copy_per_recipient_with_one_token_call():
    http = _ScriptedHTTP()
    res, _ = _api_send(http)
    assert res == (2, 0), res
    urls = [c[0] for c in http.calls]
    assert urls.count("https://oauth2.googleapis.com/token") == 1 and len(urls) == 3
    sends = [c for c in http.calls if "messages/send" in c[0]]
    assert all(c[1]["headers"]["Authorization"] == "Bearer AT-123" for c in sends)
    m1, m2 = _decode_raw(sends[0]), _decode_raw(sends[1])
    assert "one@gmail.com" in m1 and "two@gmail.com" not in m1      # nobody sees the other address
    assert "two@gmail.com" in m2 and "one@gmail.com" not in m2


def test_gmail_api_shared_header_when_recipients_are_not_hidden():
    http = _ScriptedHTTP()
    res, _ = _api_send(http, hide_recipients=False)
    assert res == (2, 0)
    sends = [c for c in http.calls if "messages/send" in c[0]]
    assert len(sends) == 1
    m = _decode_raw(sends[0])
    assert "one@gmail.com" in m and "two@gmail.com" in m


def test_gmail_api_refresh_token_rejected_is_a_clear_non_retried_error():
    from src.email_dispatcher import EmailAuthError
    http = _ScriptedHTTP(token=_Resp(400, {"error": "invalid_grant", "error_description": "Token has been expired"}))
    res, sleeps = _api_send(http)
    assert isinstance(res, EmailAuthError) and "invalid_grant" in str(res) and "In production" in str(res)
    assert "RT-secret" not in str(res) and len(http.calls) == 1 and sleeps == []


def test_gmail_api_retries_5xx_and_429_then_succeeds_and_gives_up_with_growing_pause():
    from src.email_dispatcher import EmailTransportError
    http = _ScriptedHTTP(send_script=[_Resp(503), _Resp(429), _Resp(200), _Resp(200)])
    res, sleeps = _api_send(http)
    assert res == (2, 0) and sleeps == [5.0, 15.0], (res, sleeps)

    http2 = _ScriptedHTTP(send_script=[_Resp(500)] * 3)
    res2, sleeps2 = _api_send(http2, to="one@gmail.com")
    assert isinstance(res2, EmailTransportError) and sleeps2 == [5.0, 15.0]


def test_gmail_api_retries_network_errors():
    import requests
    http = _ScriptedHTTP(send_script=[requests.ConnectionError("boom"), _Resp(200)])
    res, sleeps = _api_send(http, to="one@gmail.com")
    assert res == (1, 0) and sleeps == [5.0]


def test_gmail_api_one_rejected_recipient_does_not_stop_the_others():
    http = _ScriptedHTTP(send_script=[_Resp(400, {"error": {"status": "INVALID_ARGUMENT"}}), _Resp(200)])
    res, _ = _api_send(http)
    assert res == (1, 1), res


def test_gmail_api_permission_denied_raises_auth_error():
    from src.email_dispatcher import EmailAuthError
    http = _ScriptedHTTP(send_script=[_Resp(403, {"error": {"status": "PERMISSION_DENIED"}})])
    res, _ = _api_send(http)
    assert isinstance(res, EmailAuthError) and "gmail.send" in str(res)


def test_gmail_api_logs_never_contain_tokens_or_addresses():
    import logging
    records: list = []

    class H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    lg = logging.getLogger("src.email_dispatcher")
    h = H()
    lg.addHandler(h)
    try:
        _api_send(_ScriptedHTTP(send_script=[_Resp(503), _Resp(400, {"error": {"status": "X"}}), _Resp(200)]))
    finally:
        lg.removeHandler(h)
    joined = " ".join(records)
    assert records and "RT-secret" not in joined and "AT-123" not in joined and "@gmail.com" not in joined


def test_gmail_api_credentials_need_all_three_and_ignore_stray_whitespace():
    from src.email_dispatcher import gmail_api_credentials
    full = {"GMAIL_OAUTH_CLIENT_ID": " a.apps ", "GMAIL_OAUTH_CLIENT_SECRET": "b\n", "GMAIL_OAUTH_REFRESH_TOKEN": "c d"}
    assert gmail_api_credentials(full) == ("a.apps", "b", "cd")
    assert gmail_api_credentials({**full, "GMAIL_OAUTH_REFRESH_TOKEN": ""}) is None
    assert gmail_api_credentials({}) is None


def test_gmail_api_missing_names_only_the_absent_variables():
    from src.email_dispatcher import gmail_api_missing
    assert gmail_api_missing({}) == ["GMAIL_OAUTH_CLIENT_ID", "GMAIL_OAUTH_CLIENT_SECRET", "GMAIL_OAUTH_REFRESH_TOKEN"]
    env = {"GMAIL_OAUTH_CLIENT_ID": "a", "GMAIL_OAUTH_CLIENT_SECRET": "  ", "GMAIL_OAUTH_REFRESH_TOKEN": "c"}
    assert gmail_api_missing(env) == ["GMAIL_OAUTH_CLIENT_SECRET"]


# ------------------------------------------------- US-holiday awareness (A1)
HOL_NOW = utc("2026-01-15 22:30")          # Thu 17:30 EST; newest complete session = 2026-01-15
HOL_LABEL = pd.Timestamp("2026-01-15")


def _hol_cfg(absent_ok: bool = False, name: str = "Test Holiday") -> dict:
    cfg = load_cfg()
    cfg["data"]["holidays"] = {"2026-01-15": {"name": name, "absent_ok": absent_ok}}
    return cfg


def _thin(hourly: pd.DataFrame, label: pd.Timestamp, keep_every: int) -> pd.DataFrame:
    """Keep every n-th hourly bar of one session (24 bars -> 24/n)."""
    lab = session_label_for(hourly.index, TZ, HOUR)
    idx = hourly.index[lab == label]
    return hourly.drop(index=[ts for i, ts in enumerate(idx) if i % keep_every])


def _complete(hourly: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    c = build_ny_close_candles(hourly, TZ, HOUR, now)
    return c[c["complete"]]


def test_holiday_session_with_eight_bars_passes_with_warning():
    hourly = _thin(hourly_until(HOL_NOW), HOL_LABEL, 3)
    complete = _complete(hourly, HOL_NOW)
    assert int(complete["bars"].iloc[-1]) == 8
    w = validate_candles(complete, _hol_cfg(), last_complete_label(HOL_NOW, TZ, HOUR))
    assert any("Thin liquidity" in x and "Test Holiday" in x for x in w), w


def test_same_thin_session_on_a_normal_date_is_still_rejected():
    hourly = _thin(hourly_until(HOL_NOW), HOL_LABEL, 3)
    complete = _complete(hourly, HOL_NOW)
    cfg = load_cfg()                                   # real config: 2026-01-15 is not a holiday
    try:
        validate_candles(complete, cfg, last_complete_label(HOL_NOW, TZ, HOUR))
        raise AssertionError("expected DataError for 8 bars on a normal date")
    except DataError as exc:
        assert "hourly bars" in str(exc)


def test_holiday_below_floor_is_still_rejected():
    hourly = _thin(hourly_until(HOL_NOW), HOL_LABEL, 8)           # 3 bars < holiday_min_bars (6)
    complete = _complete(hourly, HOL_NOW)
    try:
        validate_candles(complete, _hol_cfg(), last_complete_label(HOL_NOW, TZ, HOUR))
        raise AssertionError("expected DataError below the holiday floor")
    except DataError as exc:
        assert "holiday floor" in str(exc)


def test_absent_holiday_session_is_accepted_only_when_listed():
    hourly = hourly_until(HOL_NOW)
    lab = session_label_for(hourly.index, TZ, HOUR)
    hourly = hourly[lab != HOL_LABEL]                              # no bars at all that session
    complete = _complete(hourly, HOL_NOW)
    expected = last_complete_label(HOL_NOW, TZ, HOUR)
    assert complete.index[-1] == pd.Timestamp("2026-01-14")

    w = validate_candles(complete, _hol_cfg(absent_ok=True), expected)
    assert any("No candle for 2026-01-15" in x for x in w), w

    for cfg in (_hol_cfg(absent_ok=False), load_cfg()):            # listed-but-strict, and unlisted
        try:
            validate_candles(complete, cfg, expected)
            raise AssertionError("expected Stale data error")
        except DataError as exc:
            assert "Stale" in str(exc)


def test_near_empty_absent_ok_candle_is_dropped_from_history():
    hourly = _thin(hourly_until(HOL_NOW), HOL_LABEL, 12)           # 2 bars
    complete = _complete(hourly, HOL_NOW)
    kept, notes = drop_empty_holiday_sessions(complete, _hol_cfg(absent_ok=True))
    assert HOL_LABEL not in kept.index and len(kept) == len(complete) - 1
    assert notes and "excluded" in notes[0]
    kept2, notes2 = drop_empty_holiday_sessions(complete, _hol_cfg(absent_ok=False))
    assert HOL_LABEL in kept2.index and notes2 == []               # strict holidays are never dropped


def test_bundle_accepts_thin_holiday_and_still_rejects_it_on_a_normal_day():
    hourly = _thin(hourly_until(HOL_NOW), HOL_LABEL, 3)
    with FakeTD(hourly):
        b = get_gold_bundle(_hol_cfg(), now_utc=HOL_NOW)
    assert b.candles.index[-1] == HOL_LABEL and b.quality == "nyclose_1h"
    assert any("Thin liquidity" in x for x in b.warnings), b.warnings
    with FakeTD(hourly):
        try:
            get_gold_bundle(load_cfg(), now_utc=HOL_NOW)
            raise AssertionError("expected DataError on a normal date")
        except DataError as exc:
            assert "hourly bars" in str(exc)


def test_bundle_survives_missing_holiday_session_and_notes_holiday_target():
    # Missing session (absent_ok): the run right after it still produces a bundle.
    hourly = hourly_until(HOL_NOW)
    hourly = hourly[session_label_for(hourly.index, TZ, HOUR) != HOL_LABEL]
    with FakeTD(hourly):
        b = get_gold_bundle(_hol_cfg(absent_ok=True), now_utc=HOL_NOW)
    assert b.candles.index[-1] == pd.Timestamp("2026-01-14")
    assert any("No candle for 2026-01-15" in x for x in b.warnings), b.warnings

    # Target session is the holiday (run the evening before): the email carries a caution note.
    now = utc("2026-01-14 22:30")
    with FakeTD(hourly_until(now)):
        b2 = get_gold_bundle(_hol_cfg(), now_utc=now)
    assert b2.target_session == HOL_LABEL
    assert any("US holiday session" in x for x in b2.warnings), b2.warnings


# ------------------------------------------- Market-closed holiday email (instead of an error)
def _no_data_cfg(absent_ok: bool) -> dict:
    cfg = _hol_cfg(absent_ok=absent_ok, name="Christmas Day")
    cfg["data"]["holidays"] = {"2026-01-15": {"name": "Christmas Day", "absent_ok": absent_ok}}
    return cfg


def test_closed_holiday_with_no_data_raises_market_closed():
    now = utc("2026-01-14 22:30")                          # 17:30 EST: target session = 2026-01-15
    assert target_label(now, TZ, HOUR) == HOL_LABEL
    with FakeTD(error="no data (market closed)"):
        try:
            get_gold_bundle(_no_data_cfg(absent_ok=True), now_utc=now)
            raise AssertionError("expected MarketClosedError")
        except MarketClosedError as exc:
            assert isinstance(exc, DataError)                # old callers still see a failure
            assert exc.label == HOL_LABEL and exc.holiday_name == "Christmas Day"
            assert exc.reasons, "the underlying data problems must be kept for the log"


def test_same_failure_without_absent_ok_or_off_holiday_is_still_an_error():
    now = utc("2026-01-14 22:30")
    for cfg in (_no_data_cfg(absent_ok=False), load_cfg()):   # strict holiday, and an ordinary date
        with FakeTD(error="feed down"):
            try:
                get_gold_bundle(cfg, now_utc=now)
                raise AssertionError("expected DataError")
            except MarketClosedError:
                raise AssertionError("must not be reported as a market holiday")
            except DataError as exc:
                assert "All data sources failed" in str(exc)


def test_closed_holiday_with_real_data_still_gets_a_forecast_bundle():
    now = utc("2026-01-14 22:30")
    with FakeTD(hourly_until(now)):
        b = get_gold_bundle(_no_data_cfg(absent_ok=True), now_utc=now)
    assert b.target_session == HOL_LABEL
    assert any("US holiday session" in x for x in b.warnings)


def test_market_closed_email_is_short_and_has_no_numbers_or_error_wording():
    cfg = load_cfg()
    subject, body, plain = build_market_closed_email(pd.Timestamp("2026-12-25"), "Christmas Day", cfg)
    assert subject.startswith(cfg["email"]["subject_prefix"]) and "Pasar libur" in subject
    assert "Jumat, 25 Des 2026" in plain and "Christmas Day" in plain
    for text in (subject, body, plain):
        assert "$" not in text, "no price levels in a closed-market email"
        for bad in ("gagal", "GAGAL", "error", "Error", "TIDAK TERSEDIA"):
            assert bad not in text, bad
    assert len(plain) < 400
    # a holiday name from config is escaped in the HTML part
    _, body2, _ = build_market_closed_email(pd.Timestamp("2026-12-25"), "A <b>&</b> B", cfg)
    assert "A &lt;b&gt;&amp;&lt;/b&gt; B" in body2


def test_main_sends_closed_email_and_exits_zero_on_a_closed_holiday():
    import main as main_mod
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="au_closed_"))
    orig = main_mod.get_gold_bundle
    def boom(cfg, now_utc=None):
        raise MarketClosedError(pd.Timestamp("2026-12-25"), "Christmas Day", ["feed empty"])
    main_mod.get_gold_bundle = boom
    try:
        rc = main_mod.main(["--dry-run", "--out", str(tmp), "--no-ledger"])
        preview = (tmp / "preview.txt").read_text(encoding="utf-8")
    finally:
        main_mod.get_gold_bundle = orig
    assert rc == 0, rc
    assert "Pasar libur hari ini" in preview and "PIPELINE GAGAL" not in preview


def test_main_still_reports_a_plain_data_error_as_failure():
    import main as main_mod
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="au_error_"))
    orig = main_mod.get_gold_bundle
    def boom(cfg, now_utc=None):
        raise DataError("All data sources failed. x")
    main_mod.get_gold_bundle = boom
    try:
        rc = main_mod.main(["--dry-run", "--out", str(tmp), "--no-ledger"])
        preview = (tmp / "preview.txt").read_text(encoding="utf-8")
    finally:
        main_mod.get_gold_bundle = orig
    assert rc == 1 and "PIPELINE GAGAL" in preview


def test_holiday_config_is_sane():
    cfg = load_cfg()
    d = cfg["data"]
    assert 1 <= d["holiday_min_bars"] <= d["min_bars_per_session"]
    for key in d["holidays"]:
        ts = pd.Timestamp(key)                         # must parse
        assert ts.weekday() < 5, f"{key} is a weekend: sessions are only labelled on weekdays"


def test_summarize_recipients_counts_without_exposing_addresses():
    from src.email_dispatcher import parse_recipients, summarize_recipients

    assert summarize_recipients("a@gmail.com, b@gmail.com") == (2, 0, 0)
    assert summarize_recipients("a@gmail.com;b@gmail.com,") == (2, 0, 0)     # trailing separator is not "malformed"
    assert summarize_recipients("a@gmail.com A@GMAIL.com junk @ x@y") == (1, 3, 1)
    assert summarize_recipients("") == (0, 0, 0) and summarize_recipients(None) == (0, 0, 0)
    for raw in ["a@gmail.com, b@gmail.com", "a@x.org a@X.org nope", " <c@x.org>;; d@y", ""]:
        assert summarize_recipients(raw)[0] == len(parse_recipients(raw)), raw   # same rules the sender uses


def _main_logs(argv, env, send_stub=None):
    """Run main.main() with a failing data bundle (no network), capturing the 'au-tomatic' log messages."""
    import logging
    import os
    import main as main_mod

    messages: list[str] = []

    class H(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    lg = logging.getLogger("au-tomatic")
    handler, old_level = H(), lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    saved_env = {k: os.environ.get(k) for k in env}
    orig_bundle, orig_send = main_mod.get_gold_bundle, main_mod.send_email_gmail_api

    def boom(cfg, now_utc=None):
        raise DataError("All data sources failed. x")

    main_mod.get_gold_bundle = boom
    if send_stub is not None:
        main_mod.send_email_gmail_api = send_stub
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        rc = main_mod.main(argv)
    finally:
        main_mod.get_gold_bundle, main_mod.send_email_gmail_api = orig_bundle, orig_send
        lg.removeHandler(handler)
        lg.setLevel(old_level)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return rc, messages


def test_dry_run_logs_recipient_counts_but_never_the_addresses():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="au_rcpt_")
    base = ["--dry-run", "--out", tmp, "--no-ledger", "--no-macro", "--no-news"]
    raw = "zulu.one@kantor.test, zulu.two@kantor.test; ZULU.ONE@kantor.test junk x@y"
    _, logs = _main_logs(base, {"EMAIL_TO": raw})
    text = "\n".join(logs)
    assert "EMAIL_TO: 2 usable recipient(s)." in text, text
    assert "ignored 2 malformed entries and 1 duplicate(s)" in text, text
    assert "zulu" not in text.lower() and "kantor" not in text.lower(), "an address leaked into the log"

    # a secret missing in Actions arrives as an empty string: say so before any real send
    _, logs = _main_logs(base, {"EMAIL_TO": ""})
    text = "\n".join(logs)
    assert "EMAIL_TO: 0 usable recipient(s)." in text and "no usable address" in text, text

    # a local dry run without EMAIL_TO stays quiet
    _, logs = _main_logs(base, {"EMAIL_TO": None})
    assert not any("EMAIL_TO" in m for m in logs), logs


def test_real_send_logs_recipient_count_before_sending():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="au_send_")
    env = {"EMAIL_TO": "zulu.one@kantor.test, zulu.two@kantor.test",
           "GMAIL_OAUTH_CLIENT_ID": "x", "GMAIL_OAUTH_CLIENT_SECRET": "y", "GMAIL_OAUTH_REFRESH_TOKEN": "z",
           "GMAIL_USER": None, "GMAIL_APP_PASSWORD": None}
    sent: list[int] = []

    def stub(*args, **kwargs):                      # no network: pretend both recipients were accepted
        sent.append(1)
        return 2, 0

    rc, logs = _main_logs(["--out", tmp, "--no-ledger", "--no-macro", "--no-news"], env, send_stub=stub)
    text = "\n".join(logs)
    assert sent == [1], "the stub was not used, so the real-send branch did not run"
    assert "EMAIL_TO: 2 usable recipient(s)." in text and "Email sent to 2 recipient(s)." in text, text
    assert text.index("EMAIL_TO: 2 usable") < text.index("Email sent to 2"), "count must be logged before sending"
    assert "zulu" not in text.lower() and "kantor" not in text.lower(), "an address leaked into the log"


def test_readme_secrets_table_matches_workflow():
    """Regression: README kept listing GMAIL_USER / GMAIL_APP_PASSWORD as required secrets after the
    workflow moved to Gmail API OAuth. The secrets table in the README and the secrets the workflows
    actually read must stay identical."""
    import re

    root = Path(__file__).resolve().parents[1]
    used: set[str] = set()
    for wf in (root / ".github" / "workflows").glob("*.yml"):
        used |= set(re.findall(r"secrets\.([A-Z][A-Z0-9_]*)", wf.read_text(encoding="utf-8")))
    used.discard("GITHUB_TOKEN")                      # provided by Actions, never set by hand

    documented: set[str] = set()
    in_table = False
    for line in (root / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| Secret"):
            in_table = True                           # header row of the secrets table
            continue
        if in_table:
            if not line.startswith("|"):
                break
            first_cell = line.split("|")[1]
            documented |= set(re.findall(r"`([A-Z][A-Z0-9_]*)`", first_cell))

    assert used, "no secrets found in the workflows: the test would pass vacuously"
    assert used <= documented, f"used by a workflow but missing in README: {sorted(used - documented)}"
    assert documented <= used, f"listed in README but unused by any workflow: {sorted(documented - used)}"


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
