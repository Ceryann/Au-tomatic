"""Data layer for Au-tomatic (Sprint 1).

Responsibilities
----------------
1. Download gold prices from Yahoo (1h bars, ~2 years).
2. Build D1 candles whose boundary is 17:00 America/New_York
   (the same boundary as the MT5 broker candle of the project owner).
3. Drop the candle that is still forming and validate what is left.
4. Fall back (futures, then daily bars) when the primary source fails,
   always recording *why* so the email can tell the user.

Session label convention
------------------------
A D1 candle is labelled with the calendar date (New York) on which it CLOSES.
Example: the candle that opens Sunday 17:00 NY and closes Monday 17:00 NY is
labelled Monday, exactly like an MT5 D1 candle dated Monday.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DAY = pd.Timedelta(days=1)


class DataError(RuntimeError):
    """Raised when data is missing, stale, or structurally invalid."""


class MarketClosedError(DataError):
    """No usable data AND the target session is a listed market-closed holiday (`absent_ok`).

    Subclasses DataError so any caller that does not know about it still treats it as a failure.
    `main.py` catches it first and sends the short "market closed" email instead of an error email.
    """

    def __init__(self, label: pd.Timestamp, holiday_name: str, reasons: list[str]):
        self.label, self.holiday_name, self.reasons = label, holiday_name, list(reasons)
        super().__init__(f"Market closed for {label.date()} ({holiday_name}); no usable data. "
                         + " | ".join(self.reasons))


@dataclass
class DataBundle:
    """Everything downstream stages need from the data layer."""

    candles: pd.DataFrame          # COMPLETE D1 candles; index = session label
    symbol: str                    # e.g. "XAUUSD=X"
    is_futures: bool
    quality: str                   # "nyclose_1h" | "daily_fallback"
    target_session: pd.Timestamp   # label of the candle being forecast
    reference_price: float         # latest price at run time (~ open of target)
    reference_time_utc: pd.Timestamp
    session_open: Optional[float]  # open of the target candle, if it exists yet
    now_utc: pd.Timestamp
    warnings: list[str] = field(default_factory=list)
    hourly: Optional[pd.DataFrame] = None   # hourly bars <= now (None for daily fallback)

    @property
    def source_label(self) -> str:
        kind = "Futures, basis != spot" if self.is_futures else "Spot"
        how = "1h -> D1 NY-close" if self.quality == "nyclose_1h" else "daily fallback"
        return f"{self.symbol} ({kind}; {how})"


# --------------------------------------------------------------------------
# Session / calendar helpers
# --------------------------------------------------------------------------
def session_label_for(ts, tz: str, cutoff_hour: int):
    """Label(s) of the D1 session containing `ts` (Timestamp or DatetimeIndex).

    Shifting wall-clock New York time by (24 - cutoff_hour) hours moves the
    17:00 boundary to midnight, so normalising yields the closing date.
    """
    local_naive = ts.tz_convert(tz).tz_localize(None)
    return (local_naive + pd.Timedelta(hours=24 - cutoff_hour)).normalize()


def session_end_utc(label, tz: str, cutoff_hour: int):
    """UTC instant at which the session `label` closes (17:00 NY on that date)."""
    return (label + pd.Timedelta(hours=cutoff_hour)).tz_localize(tz).tz_convert("UTC")


def last_complete_label(now_utc: pd.Timestamp, tz: str, cutoff_hour: int) -> pd.Timestamp:
    """Label of the newest weekday session that has already closed."""
    cand = session_label_for(now_utc, tz, cutoff_hour)
    while cand.weekday() >= 5 or session_end_utc(cand, tz, cutoff_hour) > now_utc:
        cand -= DAY
    return cand


def target_label(now_utc: pd.Timestamp, tz: str, cutoff_hour: int) -> pd.Timestamp:
    """Label of the candle we are forecasting (the one in progress / next to open)."""
    cand = session_label_for(now_utc, tz, cutoff_hour)
    while cand.weekday() >= 5:
        cand += DAY
    return cand


# --------------------------------------------------------------------------
# US-holiday awareness (Task A1)
# --------------------------------------------------------------------------
def holiday_map(cfg: dict) -> dict[str, dict]:
    """`data.holidays` as {"YYYY-MM-DD": {"name": str, "absent_ok": bool}} (tolerates missing key)."""
    raw = cfg.get("data", {}).get("holidays") or {}
    out: dict[str, dict] = {}
    for key, val in raw.items():
        val = val or {}
        out[str(key)[:10]] = {
            "name": str(val.get("name", "US holiday")),
            "absent_ok": bool(val.get("absent_ok", False)),
        }
    return out


def holiday_for(label, cfg: dict) -> Optional[dict]:
    """Holiday entry for a session label, or None on an ordinary day."""
    return holiday_map(cfg).get(pd.Timestamp(label).strftime("%Y-%m-%d"))


def _previous_weekday(label: pd.Timestamp) -> pd.Timestamp:
    cand = label - DAY
    while cand.weekday() >= 5:
        cand -= DAY
    return cand


def drop_empty_holiday_sessions(candles: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Remove near-empty candles on `absent_ok` holidays (market effectively closed).

    A candle built from a couple of bars has a meaningless range and would distort the
    volatility estimate, so it is treated exactly like a missing session.
    """
    if "bars" not in candles.columns or candles["bars"].isna().all():
        return candles, []
    d = cfg["data"]
    floor = d.get("holiday_min_bars", d["min_bars_per_session"])
    drop, notes = [], []
    for label, bars in candles["bars"].items():
        h = holiday_for(label, cfg)
        if h and h["absent_ok"] and bars < floor:
            drop.append(label)
            notes.append(
                f"{label.date()} ({h['name']}): only {int(bars)} hourly bar(s); candle excluded "
                f"from history (market effectively closed)."
            )
    if drop:
        candles = candles.drop(index=drop)
    return candles, notes


def holiday_target_note(target: pd.Timestamp, cfg: dict) -> list[str]:
    """Email note when the candle being forecast falls on a listed US holiday."""
    h = holiday_for(target, cfg)
    if not h:
        return []
    return [
        f"US holiday session ({h['name']}): liquidity is usually thin and gaps are more likely; "
        f"treat today's range with extra caution."
    ]


# --------------------------------------------------------------------------
# Download (network) - isolated so tests can replace it
# --------------------------------------------------------------------------
def _download(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch OHLC from Yahoo. Imported lazily so tests need no yfinance."""
    import yfinance as yf

    df = yf.Ticker(symbol).history(
        period=f"{days}d", interval=interval, auto_adjust=False, actions=False
    )
    if df is None or df.empty:
        raise DataError(f"Yahoo returned no data for {symbol} ({interval}).")
    return df


def _clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate and drop unusable rows from a raw download."""
    need = ["Open", "High", "Low", "Close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise DataError(f"Download is missing columns: {missing}")
    out = df.copy()
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    out.index = out.index.tz_convert("UTC")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=need)
    out = out[(out[need] > 0).all(axis=1)]
    if out.empty:
        raise DataError("No valid rows left after cleaning.")
    return out[need + ["Volume"]]


def _with_retry(fn: Callable[[], pd.DataFrame], retries: int, backoff: float, what: str):
    last: Optional[Exception] = None
    for i in range(max(1, retries)):
        try:
            return fn()
        except Exception as exc:  # network, 429, empty frame, parse error...
            last = exc
            wait = backoff * (2 ** i)
            log.warning("%s failed (attempt %d/%d): %s", what, i + 1, retries, exc)
            if i < retries - 1:
                time.sleep(wait)
    raise DataError(f"{what} failed after {retries} attempts: {last}")


# --------------------------------------------------------------------------
# Candle construction
# --------------------------------------------------------------------------
def build_ny_close_candles(
    hourly: pd.DataFrame, tz: str, cutoff_hour: int, now_utc: pd.Timestamp
) -> pd.DataFrame:
    """Aggregate hourly bars into D1 candles with a 17:00 New York boundary.

    Returns one row per weekday session with columns
    Open, High, Low, Close, Volume, bars, first_ts, last_ts, end_utc, complete.
    Bars that land on a weekend label (data glitches) are discarded.
    """
    df = hourly.sort_index()
    tmp = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    tmp["_label"] = session_label_for(df.index, tz, cutoff_hour)
    tmp["_ts"] = df.index

    grouped = tmp.groupby("_label")
    out = pd.DataFrame(
        {
            "Open": grouped["Open"].first(),
            "High": grouped["High"].max(),
            "Low": grouped["Low"].min(),
            "Close": grouped["Close"].last(),
            "Volume": grouped["Volume"].sum(),
            "bars": grouped["Close"].size(),
            "first_ts": grouped["_ts"].min(),
            "last_ts": grouped["_ts"].max(),
        }
    )
    out.index.name = "session"

    weekend = out.index.dayofweek >= 5
    if weekend.any():
        log.debug("Discarding %d weekend-labelled session(s).", int(weekend.sum()))
        out = out[~weekend]

    out["end_utc"] = (out.index + pd.Timedelta(hours=cutoff_hour)).tz_localize(tz).tz_convert("UTC")
    out["complete"] = out["end_utc"] <= now_utc
    return out


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_candles(
    candles: pd.DataFrame,
    cfg: dict,
    expected_last: pd.Timestamp,
    check_bars: bool = True,
) -> list[str]:
    """Raise DataError on hard failures; return soft warnings otherwise."""
    d = cfg["data"]
    warnings: list[str] = []

    if len(candles) < d["min_sessions"]:
        raise DataError(f"Only {len(candles)} complete candles; need >= {d['min_sessions']}.")

    ohlc = candles[["Open", "High", "Low", "Close"]]
    if ohlc.isna().any().any():
        raise DataError("NaN found in OHLC.")
    if (candles["High"] < candles["Low"]).any():
        raise DataError("Found candle(s) with High < Low.")
    tol = 1e-6 * candles["Close"]
    if (candles["High"] + tol < candles[["Open", "Close"]].max(axis=1)).any() or (
        candles["Low"] - tol > candles[["Open", "Close"]].min(axis=1)
    ).any():
        raise DataError("Found candle(s) where High/Low do not bound Open/Close.")

    last_label = candles.index[-1]
    if last_label != expected_last:
        # An `absent_ok` holiday session may legitimately be missing: accept candles that end on
        # the previous weekday, but only for the holidays listed in config (never for plain gaps).
        cand, accepted = expected_last, False
        while cand > last_label:
            h = holiday_for(cand, cfg)
            if not (h and h["absent_ok"]):
                break
            cand = _previous_weekday(cand)
            if cand == last_label:
                accepted = True
                warnings.append(
                    f"No candle for {expected_last.date()} ({h['name']}): market effectively closed; "
                    f"history ends on {last_label.date()}."
                )
                break
        if not accepted:
            raise DataError(
                f"Stale data: newest complete candle is {last_label.date()}, "
                f"expected {expected_last.date()}."
            )

    if check_bars:
        bars = int(candles["bars"].iloc[-1])
        need = d["min_bars_per_session"]
        if bars < need:
            h = holiday_for(last_label, cfg)
            floor = min(need, d.get("holiday_min_bars", need))
            if h and bars >= floor:
                warnings.append(
                    f"Thin liquidity on {last_label.date()} ({h['name']}): {bars} hourly bars "
                    f"(normal >= {need}); accepted as a US-holiday session."
                )
            else:
                extra = f"; holiday floor {floor} for {h['name']}" if h else ""
                raise DataError(
                    f"Newest complete candle has only {bars} hourly bars "
                    f"(need >= {need}{extra})."
                )

    rets = candles["Close"].pct_change().abs().dropna()
    big = rets[rets > d["max_abs_daily_return"]]
    if len(big):
        days = ", ".join(str(i.date()) for i in big.index[-3:])
        warnings.append(
            f"{len(big)} daily move(s) above {d['max_abs_daily_return']:.0%} "
            f"(latest: {days}); check for roll/gap artefacts."
        )

    full = pd.bdate_range(candles.index[0], candles.index[-1])
    missing = len(full.difference(candles.index))
    if missing > d["max_missing_weekdays"]:
        warnings.append(f"{missing} weekdays missing from history (holidays or data gaps).")

    return warnings


# --------------------------------------------------------------------------
# Twelve Data: true spot XAU/USD
# --------------------------------------------------------------------------
TD_URL = "https://api.twelvedata.com/time_series"


def _td_request(params: dict) -> dict:
    """One HTTP call to Twelve Data. Isolated so tests can replace it. Masks the key."""
    import requests

    key = str(params.get("apikey", ""))
    try:
        resp = requests.get(TD_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace(key, "***") if key else str(exc)
        raise DataError(f"Twelve Data request failed: {msg}") from None


def _td_page(api_key: str, symbol: str, end_date: str, size: int) -> pd.DataFrame:
    data = _td_request(
        {
            "symbol": symbol, "interval": "1h", "outputsize": size,
            "end_date": end_date, "timezone": "UTC", "format": "JSON", "apikey": api_key,
        }
    )
    if data.get("status") == "error" or "values" not in data:
        raise DataError(f"Twelve Data error {data.get('code')}: {data.get('message')}")
    df = pd.DataFrame(data["values"])
    if df.empty:
        raise DataError("Twelve Data returned an empty page.")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns=str.title)
    for c in ("Open", "High", "Low", "Close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Volume"] = 0.0
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_twelvedata_hourly(cfg: dict, now_utc: pd.Timestamp, api_key: str) -> pd.DataFrame:
    """Page backwards until `history_days` is covered, the history ends, or max_pages is hit."""
    td, d = cfg["twelvedata"], cfg["data"]
    symbol = cfg["market"]["twelvedata_symbol"]
    wanted_start = now_utc - pd.Timedelta(days=d["history_days"])

    frames: list[pd.DataFrame] = []
    end = now_utc.strftime("%Y-%m-%d %H:%M:%S")
    for i in range(int(td["max_pages"])):
        retries = d["retries"] if i == 0 else 1     # a later "no more data" must not burn retries
        try:
            page = _with_retry(
                lambda e=end: _td_page(api_key, symbol, e, int(td["page_size"])),
                retries, d["backoff_seconds"], f"Twelve Data page {i + 1}",
            )
        except DataError:
            if frames:
                log.warning("Twelve Data page %d unavailable; using %d page(s).", i + 1, len(frames))
                break
            raise
        frames.append(page)
        oldest = page.index[0]
        if oldest <= wanted_start or len(page) < int(td["page_size"]):
            break
        end = (oldest - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(float(td["min_interval_seconds"]))

    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


# --------------------------------------------------------------------------
# Sources -> DataBundle
# --------------------------------------------------------------------------
def _weekend_window(now_utc: pd.Timestamp, tz: str) -> bool:
    """True from Friday 17:00 New York through all of Sunday (market closed or just reopening)."""
    ny = now_utc.tz_convert(tz)
    return ny.weekday() in (5, 6) or (ny.weekday() == 4 and ny.hour >= 17)


def _check_feed_freshness(last_bar: pd.Timestamp, now_utc: pd.Timestamp, tz: str, max_hours: float) -> None:
    """Reject a feed whose newest bar is too old while the market should be trading.

    Skipped while the market is closed (Fri 17:00 NY -> Sun 17:00 NY). On Sunday
    evening a missing new-week bar is tolerated here; the caller adds a warning.
    """
    ny = now_utc.tz_convert(tz)
    closed = (
        ny.weekday() == 5
        or (ny.weekday() == 4 and ny.hour >= 17)
        or (ny.weekday() == 6 and ny.hour < 17)
    )
    if closed or ny.weekday() == 6:
        return
    age_h = (now_utc - last_bar).total_seconds() / 3600
    if age_h > max_hours:
        raise DataError(f"Stale feed: newest hourly bar is {age_h:.1f}h old (max {max_hours}h).")


def _bundle_from_hourly(
    symbol: str, is_futures: bool, cfg: dict, now_utc: pd.Timestamp,
    loader: Optional[Callable[[], pd.DataFrame]] = None,
) -> DataBundle:
    m, d = cfg["market"], cfg["data"]
    tz, hour = m["candle_cutoff_tz"], m["candle_cutoff_hour"]

    if loader is None:
        raw = _with_retry(
            lambda: _download(symbol, "1h", d["history_days"]),
            d["retries"], d["backoff_seconds"], f"{symbol} 1h download",
        )
    else:
        raw = loader()
    hourly = _clean_raw(raw)
    hourly = hourly[hourly.index <= now_utc]  # never use bars from the "future"
    if hourly.empty:
        raise DataError(f"{symbol}: no hourly bars at or before {now_utc}.")

    _check_feed_freshness(hourly.index[-1], now_utc, tz, d["max_stale_hours"])

    candles_all = build_ny_close_candles(hourly, tz, hour, now_utc)
    complete = candles_all[candles_all["complete"]]
    complete, holiday_notes = drop_empty_holiday_sessions(complete, cfg)
    expected = last_complete_label(now_utc, tz, hour)
    warnings = holiday_notes + validate_candles(complete, cfg, expected_last=expected, check_bars=True)

    target = target_label(now_utc, tz, hour)
    warnings += holiday_target_note(target, cfg)
    if target in candles_all.index and not candles_all.loc[target, "complete"]:
        cur = candles_all.loc[target]
        reference_price = float(hourly["Close"].iloc[-1])
        reference_time = cur["last_ts"]
        session_open = float(cur["Open"])
    else:
        # Legitimate only while the market is closed (Fri 17:00 NY -> Sun 17:00 NY) or on
        # Sunday evening before the feed publishes its first bar of the new week.
        if not _weekend_window(now_utc, tz):
            h = holiday_for(target, cfg)
            why = f" ({h['name']}: the market may be closed today)" if h else ""
            raise DataError(f"No hourly bars yet for the target session {target.date()}{why}.")
        last = complete.iloc[-1]
        reference_price = float(last["Close"])
        reference_time = last["last_ts"]
        session_open = None
        warnings.append(
            "Weekend gap: the new week has no bars yet; reference price is Friday's close "
            "and the forecast covers the whole Monday candle."
        )

    if is_futures:
        warnings.append(
            "Source is Futures (GC=F): price basis differs from spot by tens of dollars. "
            "Use the ranges relative to your MT5 open, not the absolute levels."
        )

    return DataBundle(
        candles=complete[["Open", "High", "Low", "Close", "Volume", "bars"]],
        symbol=symbol, is_futures=is_futures, quality="nyclose_1h",
        target_session=target, reference_price=reference_price,
        reference_time_utc=reference_time, session_open=session_open,
        now_utc=now_utc, warnings=warnings, hourly=hourly,
    )


def _bundle_from_daily(symbol: str, is_futures: bool, cfg: dict, now_utc: pd.Timestamp) -> DataBundle:
    """Plan B: Yahoo daily bars. Boundary is NOT exactly 17:00 NY, so it is flagged."""
    m, d = cfg["market"], cfg["data"]
    tz, hour = m["candle_cutoff_tz"], m["candle_cutoff_hour"]

    raw = _with_retry(
        lambda: _download(symbol, "1d", d["history_days"]),
        d["retries"], d["backoff_seconds"], f"{symbol} 1d download",
    )
    # Daily bars are stamped at local midnight of the exchange; keep that calendar
    # date as the label (converting to New York time would shift it back a day).
    idx = raw.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    raw = raw.copy()
    raw.index = idx.normalize()
    df = _clean_raw(raw)               # treats the naive index as UTC
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index.dayofweek < 5]
    df.index.name = "session"

    expected = last_complete_label(now_utc, tz, hour)
    complete = df[df.index <= expected].copy()
    complete["bars"] = np.nan
    warnings = validate_candles(complete, cfg, expected_last=expected, check_bars=False)
    warnings += holiday_target_note(target_label(now_utc, tz, hour), cfg)
    warnings.append(
        "Daily fallback: Yahoo daily bars do not end exactly at 17:00 New York; "
        "levels may differ from your MT5 candle."
    )
    if is_futures:
        warnings.append("Source is Futures (GC=F): basis differs from spot.")

    last_bar = df.iloc[-1]
    return DataBundle(
        candles=complete[["Open", "High", "Low", "Close", "Volume", "bars"]],
        symbol=symbol, is_futures=is_futures, quality="daily_fallback",
        target_session=target_label(now_utc, tz, hour),
        reference_price=float(last_bar["Close"]),
        reference_time_utc=now_utc, session_open=None,
        now_utc=now_utc, warnings=warnings, hourly=None,
    )


def get_gold_bundle(cfg: dict, now_utc: Optional[pd.Timestamp] = None) -> DataBundle:
    """Try each source in order; the first that validates wins.

    Order: Twelve Data spot 1h -> Yahoo futures 1h -> Yahoo futures daily.
    Every failed attempt is reported in `bundle.warnings`.
    """
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")

    failures: list[str] = []
    attempts: list[tuple[str, Callable[[], DataBundle]]] = []

    td_symbol = cfg["market"]["twelvedata_symbol"]
    key = os.environ.get(cfg["twelvedata"]["api_key_env"], "").strip()
    if key:
        attempts.append((
            f"{td_symbol} 1h",
            lambda: _bundle_from_hourly(
                td_symbol, False, cfg, now,
                loader=lambda: fetch_twelvedata_hourly(cfg, now, key),
            ),
        ))
    else:
        failures.append(
            f"{cfg['twelvedata']['api_key_env']} not set: spot XAU/USD unavailable, using fallback."
        )

    for sym in cfg["market"]["yahoo_symbols"]:
        fut = sym.endswith("=F")
        attempts.append((f"{sym} 1h", lambda sym=sym, fut=fut: _bundle_from_hourly(sym, fut, cfg, now)))
    for sym in cfg["market"]["yahoo_symbols"]:
        fut = sym.endswith("=F")
        attempts.append((f"{sym} 1d", lambda sym=sym, fut=fut: _bundle_from_daily(sym, fut, cfg, now)))

    for label, build in attempts:
        try:
            bundle = build()
        except DataError as exc:
            msg = f"{label} failed: {exc}"
            log.warning(msg)
            failures.append(msg)
            continue
        bundle.warnings = failures + bundle.warnings
        return bundle

    target = target_label(now, cfg["market"]["candle_cutoff_tz"], cfg["market"]["candle_cutoff_hour"])
    h = holiday_for(target, cfg)
    if h and h["absent_ok"]:       # a listed closed-market day with no data is expected, not a fault
        raise MarketClosedError(target, h["name"], failures)
    raise DataError("All data sources failed. " + " | ".join(failures))


# --------------------------------------------------------------------------
# Optional macro features (DXY, US10Y) - never allowed to break the pipeline
# --------------------------------------------------------------------------
def fetch_macro_daily(cfg: dict, now_utc: pd.Timestamp) -> tuple[Optional[pd.DataFrame], list[str]]:
    """Daily DXY log-return and US10Y change keyed by calendar date.

    Returns (frame_or_None, notes). Any failure yields (None, [note]) so the model
    simply runs on gold-only features.
    """
    mc = cfg["model"]["macro"]
    if not mc.get("enabled", False):
        return None, []
    d = cfg["data"]
    closes: dict[str, pd.Series] = {}
    for name, symbol in mc["symbols"].items():
        try:
            raw = _with_retry(
                lambda s=symbol: _download(s, "1d", d["history_days"]),
                2, d["backoff_seconds"], f"{symbol} 1d download",
            )
            idx = raw.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            raw = raw.copy()
            raw.index = idx.normalize()
            clean = _clean_raw(raw)
            clean.index = clean.index.tz_localize(None)
            closes[name] = clean["Close"][~clean.index.duplicated(keep="last")]
        except DataError as exc:
            return None, [f"Macro data unavailable ({symbol}); regime uses gold-only features. ({exc})"]

    frame = pd.DataFrame(closes).sort_index()
    frame = frame[frame.index <= now_utc.tz_localize(None).normalize()]
    out = pd.DataFrame(index=frame.index)
    if "dxy" in frame:
        out["dxy_ret"] = np.log(frame["dxy"]).diff()
    if "tnx" in frame:
        out["tnx_chg"] = frame["tnx"].diff()
    return out, []
