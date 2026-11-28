"""News and calendar radar (Sprint 4). Every external call is optional and fails softly.

What it does
------------
1. Economic calendar (ForexFactory public JSON feed): finds USD events in the target session.
   High-impact events raise the jump intensity (lambda) of the Monte Carlo. If the feed is
   down, a static list of FOMC decision days (from the Federal Reserve calendar) is used.
2. Headlines (RSS): scored with a GOLD-specific lexicon. Note: generic sentiment tools such as
   VADER treat "war" or "tensions" as negative, but for gold those are bullish, so we do not use them.
3. Display rule (to keep the email short): headlines are shown only if there is a calendar event
   in the last/next 24h window, OR the aggregate score is extreme (|score| >= threshold).

Honest limits
-------------
* The lexicon is a transparent heuristic. It is NOT validated: no historical news archive
  exists to backtest it, so sentiment never changes the numbers, it is only shown.
* The event multiplier on lambda is a judgement parameter; `tools/event_study.py` measures
  it on past FOMC days.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .data_fetcher import DataBundle, session_end_utc

log = logging.getLogger(__name__)

UA = {"User-Agent": "Au-tomatic/1.0 (personal research bot)"}


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class Event:
    title: str
    time_utc: pd.Timestamp
    impact: str            # High / Medium / Low / Holiday
    country: str
    source: str = "feed"   # feed | static


@dataclass
class Headline:
    title: str
    source: str
    link: str
    published: Optional[pd.Timestamp]
    score: float = 0.0


@dataclass
class NewsReport:
    events: list[Event] = field(default_factory=list)          # upcoming events to display
    event_today: bool = False                                   # high-impact event still to come
    high_events: list[Event] = field(default_factory=list)      # those events (their times feed the model)
    headlines: list[Headline] = field(default_factory=list)    # headlines to display (may be empty)
    agg_score: float = 0.0
    n_headlines: int = 0
    show_news: bool = False
    calendar_ok: bool = False
    news_ok: bool = False
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# HTTP (isolated so tests can replace it)
# --------------------------------------------------------------------------
def _http_get(url: str, timeout: float) -> str:
    import requests

    resp = requests.get(url, headers=UA, timeout=timeout)
    resp.raise_for_status()
    return resp.text


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------
def parse_calendar(raw: str) -> list[Event]:
    """Parse the ForexFactory JSON feed: [{title, country, date, impact, ...}, ...]."""
    data = json.loads(raw)
    out: list[Event] = []
    for it in data:
        try:
            ts = pd.Timestamp(it["date"])
            ts = ts.tz_localize("America/New_York") if ts.tzinfo is None else ts
            out.append(Event(
                title=str(it.get("title", "")).strip(), time_utc=ts.tz_convert("UTC"),
                impact=str(it.get("impact", "")).strip().title(), country=str(it.get("country", "")).strip(),
            ))
        except Exception:  # noqa: BLE001 - one malformed row must not drop the whole feed
            continue
    return out


def static_events(cfg: dict) -> list[Event]:
    out = []
    for e in cfg["news"].get("static_events", []):
        ts = pd.Timestamp(f"{e['date']} {e.get('time_ny', '14:00')}").tz_localize("America/New_York")
        out.append(Event(e["title"], ts.tz_convert("UTC"), e.get("impact", "High"), "USD", source="static"))
    return out


def fetch_events(cfg: dict) -> tuple[list[Event], bool, list[str]]:
    """Return (events, feed_ok, notes). Falls back to the static list when every feed fails."""
    n, notes = cfg["news"], []
    events: list[Event] = []
    ok = False
    for url in n["calendar_urls"]:
        try:
            events += parse_calendar(_http_get(url, n["timeout_seconds"]))
            ok = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Calendar feed failed (%s).", type(exc).__name__)   # e.g. next-week feed not yet published
    if not ok:
        events = static_events(cfg)
        notes.append("kalender ekonomi tidak tersedia; memakai jadwal FOMC bawaan")
    seen, unique = set(), []                      # this-week and next-week feeds can overlap
    for e in events:
        key = (e.title.lower(), e.time_utc, e.country)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique, ok, notes


def events_between(events: list[Event], start: pd.Timestamp, end: pd.Timestamp,
                   impacts: list[str], currency: str) -> list[Event]:
    keep = [e for e in events
            if e.country == currency and e.impact in impacts and start <= e.time_utc <= end]
    return sorted(keep, key=lambda e: e.time_utc)


# --------------------------------------------------------------------------
# Headlines
# --------------------------------------------------------------------------
def parse_rss(xml_text: str, default_source: str) -> list[Headline]:
    root = ET.fromstring(xml_text)
    out: list[Headline] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        source = (item.findtext("source") or "").strip()
        if not source and " - " in title:                       # Google News: "Headline - Publisher"
            title, source = title.rsplit(" - ", 1)
        pub = pd.to_datetime(item.findtext("pubDate"), utc=True, errors="coerce")
        out.append(Headline(title.strip(), source or default_source, (item.findtext("link") or "").strip(),
                            None if pd.isna(pub) else pub))
    return out


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())[:60]


def fetch_headlines(cfg: dict, now_utc: pd.Timestamp) -> tuple[list[Headline], bool, list[str]]:
    n, notes = cfg["news"], []
    heads: list[Headline] = []
    ok = False
    for url in n["rss_urls"]:
        try:
            heads += parse_rss(_http_get(url, n["timeout_seconds"]), "RSS")
            ok = True
        except Exception as exc:  # noqa: BLE001
            log.warning("RSS feed failed (%s).", type(exc).__name__)
    if not ok:
        notes.append("radar berita tidak tersedia")
    cutoff = now_utc - pd.Timedelta(hours=n["max_age_hours"])
    seen, fresh = set(), []
    for h in heads:
        if h.published is not None and h.published < cutoff:
            continue
        if not is_relevant(h.title):
            continue
        key = _norm(h.title)
        if key in seen:
            continue
        seen.add(key)
        fresh.append(h)
    return fresh, ok, notes


# --------------------------------------------------------------------------
# Gold-specific sentiment lexicon (+ = bullish for gold, - = bearish for gold)
# --------------------------------------------------------------------------
_DOWN = r"(?:falls?|fell|drops?|dropped|slides?|slid|slumps?|tumbles?|plunges?|declines?|sinks?|retreats?|slips?|dips?|sheds?|loses?|weakens?)"
_UP = r"(?:rallies|rally|jumps?|surges?|climbs?|rises?|rose|gains?|rebounds?|advances?|soars?|spikes?|strengthens?)"
_GOLD = r"(?:gold|xau/usd|xauusd|bullion)(?: \(xau/usd\))?(?: (?:price|prices|futures))?"
# up to two filler words between subject and verb, but never across a clause boundary such as "as" or "while"
_GAP = r"(?:(?!as\b|while\b|but\b|and\b|after\b|amid\b|on\b|with\b|even\b)[\w-]+ ){0,2}"

BULLISH = {
    r"rate[- ]cuts?": 0.6, r"cuts? (?:interest )?rates?": 0.6, r"dovish": 0.6, r"(?:monetary )?easing": 0.4,
    r"weak(?:er)? (?:us )?(?:jobs|payrolls|nfp|data|labou?r)": 0.5,
    rf"(?:us )?(?:dollar|dollar index|dxy|greenback) {_GAP}{_DOWN}": 0.4, r"weaker dollar": 0.4,
    r"(?:treasury )?yields? (?:\w+ ){0,2}(?:fall|drop|slide|decline|dip|retreat|ease|slip)s?": 0.4,
    r"safe[- ]haven": 0.4, r"geopolitical|tensions?|conflict|missile|sanctions?|\bwar\b": 0.4,
    r"central banks? (?:\w+ ){0,2}(?:buy|buying|purchases?)": 0.5,
    r"sticky inflation|inflation (?:rises?|accelerates?|heats? up|jumps?)": 0.3,
    r"recession (?:fears?|risks?|worries)": 0.4,
    rf"{_GOLD} {_GAP}{_UP}": 0.3, r"record[- ]high|all[- ]time high|breakout|bulls? (?:in control|take)": 0.3,
}
BEARISH = {
    r"rate[- ]hikes?": 0.6, r"(?:hike|raise|raising) (?:interest )?rates?": 0.6, r"hawkish": 0.6,
    r"tightening": 0.4, r"higher for longer": 0.5,
    r"strong(?:er)? (?:us )?(?:jobs|payrolls|nfp|data|labou?r)": 0.5,
    rf"(?:us )?(?:dollar|dollar index|dxy|greenback) {_GAP}{_UP}": 0.4, r"stronger dollar": 0.4,
    r"(?:treasury )?yields? (?:\w+ ){0,2}(?:rise|jump|surge|climb|spike|advance)s?": 0.4,
    r"risk[- ]on": 0.3, r"profit[- ]taking": 0.3,
    rf"{_GOLD} {_GAP}{_DOWN}": 0.3,
    r"sell[- ]?off": 0.3, r"(?:yearly|monthly|multi[- ](?:week|month|year)|\d+[- ](?:week|month)) lows?|lowest since": 0.3,
    r"ceasefire|de-?escalation|peace (?:deal|talks)": 0.3,
}
NEGATORS = {"no", "not", "without", "unlikely", "fade", "fades", "fading", "fewer", "ease", "eases",
            "easing", "recede", "recedes", "dim", "dims", "dashed", "rule", "rules", "ruled", "against"}


def _negated(text: str, start: int, end: int) -> bool:
    before = re.findall(r"[a-z']+", text[:start])[-3:]
    after = re.findall(r"[a-z']+", text[end:])[:3]
    return any(w in NEGATORS for w in before + after)


_MINING_NOISE = re.compile(
    r"\b(?:drill(?:ing|s)?|resources?|ounces?|oz|tonnes?|royalt(?:y|ies)|miners?|mining|mines?|explorer|"
    r"exploration|feasibility|private placement|ipo|acquir(?:es?|ition)|permits?|concession|jewell?ery|"
    r"coins?|tax plan|dividend|quarterly results|earnings)\b", re.I)
_MARKET_TERMS = re.compile(
    r"\b(?:xau|xauusd|spot gold|gold (?:price|prices|futures)|bullion|fed|federal reserve|fomc|dollar|dxy|"
    r"yields?|inflation|payrolls?|cpi|pce|rate[- ](?:cut|hike)s?)\b", re.I)
_ON_TOPIC = re.compile(
    r"gold|xau|bullion|\bfed\b|federal reserve|fomc|central bank|dollar|dxy|treasury|yields?|inflation|"
    r"payrolls?|rate[- ](?:cut|hike)s?", re.I)


def is_relevant(title: str) -> bool:
    """Keep headlines about the gold MARKET; drop mining-company and corporate news."""
    if not _ON_TOPIC.search(title):
        return False
    return not (_MINING_NOISE.search(title) and not _MARKET_TERMS.search(title))


def score_headline(title: str) -> float:
    """Gold-direction score in [-1, 1]. Positive = bullish for gold."""
    text = title.lower()
    total = 0.0
    for table, sign in ((BULLISH, 1.0), (BEARISH, -1.0)):
        for pattern, weight in table.items():
            for m in re.finditer(pattern, text):
                s = sign * weight
                total += -s if _negated(text, m.start(), m.end()) else s
    return float(np.clip(total, -1.0, 1.0))


def aggregate(headlines: list[Headline], top_n: int = 5) -> float:
    scored = sorted((h for h in headlines if h.score != 0.0), key=lambda h: -abs(h.score))[:top_n]
    return float(np.mean([h.score for h in scored])) if scored else 0.0


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def session_window(bundle: DataBundle, cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(effective_now, session_end) for the target candle; weekend runs start at the open."""
    tz, hour = cfg["market"]["candle_cutoff_tz"], cfg["market"]["candle_cutoff_hour"]
    end = session_end_utc(bundle.target_session, tz, hour)
    start = session_end_utc(bundle.target_session - pd.Timedelta(days=1), tz, hour)
    return max(bundle.now_utc, start), end


def build_news_report(cfg: dict, bundle: DataBundle) -> NewsReport:
    """Never raises: every failure becomes a note."""
    rep = NewsReport()
    try:
        n = cfg["news"]
        now_eff, end = session_window(bundle, cfg)
        events, rep.calendar_ok, notes = fetch_events(cfg)
        rep.notes += notes
        cur = n["event_currency"]
        rep.events = events_between(events, now_eff, end, n["display_impacts"], cur)
        rep.high_events = events_between(events, now_eff, end, n["lambda_impacts"], cur)
        rep.event_today = bool(rep.high_events)
        recent = events_between(events, bundle.now_utc - pd.Timedelta(hours=24), end, n["display_impacts"], cur)

        heads, rep.news_ok, notes = fetch_headlines(cfg, bundle.now_utc)
        rep.notes += notes
        for h in heads:
            h.score = score_headline(h.title)
        rep.n_headlines = len(heads)
        rep.agg_score = aggregate(heads)
        rep.show_news = bool(recent) or abs(rep.agg_score) >= n["extreme_score"]
        if rep.show_news:
            pick = sorted((h for h in heads if abs(h.score) >= 0.2), key=lambda h: -abs(h.score))
            rep.headlines = pick[: n["max_headlines"]]
        log.info("News radar: calendar_ok=%s news_ok=%s events=%d headlines=%d show=%s",
                 rep.calendar_ok, rep.news_ok, len(rep.events), rep.n_headlines, rep.show_news)
    except Exception as exc:  # noqa: BLE001
        log.warning("News radar skipped (%s: %s).", type(exc).__name__, exc)
        rep.notes.append("radar berita dilewati karena galat")
    return rep
