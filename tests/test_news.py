"""Offline tests for the calendar and news radar (Sprint 4). Never touches the network."""

from __future__ import annotations

import json
import os
import smtplib
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_data_layer as T  # noqa: E402
import test_ledger as L  # noqa: E402
import test_models as M  # noqa: E402
from src import news_sentinel as ns  # noqa: E402
from src.email_dispatcher import build_forecast_email  # noqa: E402
from src.ledger import load_ledger  # noqa: E402
from src.news_sentinel import (  # noqa: E402
    Headline,
    aggregate,
    build_news_report,
    events_between,
    parse_calendar,
    parse_rss,
    score_headline,
    session_window,
)
from src.quant_engine import forecast_day  # noqa: E402

# T.NOW = Tue 2026-01-13 22:30 UTC -> target session Wed 2026-01-14 (closes 22:00 UTC)
CAL = json.dumps([
    {"title": "Core CPI m/m", "country": "USD", "date": "2026-01-14T08:30:00-05:00", "impact": "High"},
    {"title": "Unemployment Claims", "country": "USD", "date": "2026-01-14T08:30:00-05:00", "impact": "Medium"},
    {"title": "German Retail", "country": "EUR", "date": "2026-01-14T05:00:00-05:00", "impact": "High"},
    {"title": "Bank Holiday", "country": "USD", "date": "2026-01-14T00:00:00-05:00", "impact": "Holiday"},
    {"title": "Far future CPI", "country": "USD", "date": "2026-02-11T08:30:00-05:00", "impact": "High"},
    {"title": "Broken row", "country": "USD"},
])
CAL_QUIET = json.dumps([
    {"title": "Far future CPI", "country": "USD", "date": "2026-02-11T08:30:00-05:00", "impact": "High"},
])


def rss(*items: tuple[str, str]) -> str:
    body = "".join(
        f"<item><title>{t}</title><link>http://example.com/{i}</link><pubDate>{d}</pubDate></item>"
        for i, (t, d) in enumerate(items)
    )
    return f"<rss><channel>{body}</channel></rss>"


FRESH = "Tue, 13 Jan 2026 18:00:00 GMT"
STALE = "Mon, 05 Jan 2026 10:00:00 GMT"
NEUTRAL = rss(("Gold price today: what to know - Kitco", FRESH), ("Markets wrap - Reuters", FRESH))
BULLISH = rss(
    ("Fed signals rate cuts as dollar falls - Reuters", FRESH),
    ("Gold climbs as geopolitical tensions escalate - Bloomberg", FRESH),
    ("Central banks buying gold at record pace - FT", FRESH),
    ("Safe-haven demand lifts gold - CNBC", FRESH),
)


class Http:
    """Replace news_sentinel._http_get with routes: substring of URL -> body or Exception."""

    def __init__(self, calendar=None, rss_body=None):
        self.calendar, self.rss_body = calendar, rss_body

    def _get(self, url, timeout=0):
        body = self.calendar if "faireconomy" in url else self.rss_body
        if body is None or isinstance(body, Exception):
            raise body or OSError("feed down")
        return body

    def __enter__(self):
        self._orig = ns._http_get
        ns._http_get = self._get
        return self

    def __exit__(self, *exc):
        ns._http_get = self._orig


def _bundle():
    cfg = M.fast_cfg()
    return cfg, M.bundle_at(T.NOW, T.hourly_until(T.NOW, seed=11), cfg)


# ------------------------------------------------------------------ calendar
def test_calendar_parsing_converts_new_york_time_and_skips_broken_rows():
    ev = parse_calendar(CAL)
    assert len(ev) == 5                                           # "Broken row" dropped
    cpi = next(e for e in ev if e.title == "Core CPI m/m")
    assert cpi.time_utc == pd.Timestamp("2026-01-14 13:30", tz="UTC")   # 08:30 EST
    assert cpi.impact == "High" and cpi.country == "USD"


def test_events_are_filtered_by_currency_impact_and_session_window():
    cfg, b = _bundle()
    now_eff, end = session_window(b, cfg)
    assert now_eff == b.now_utc and end == pd.Timestamp("2026-01-14 22:00", tz="UTC")
    ev = parse_calendar(CAL)
    high = events_between(ev, now_eff, end, ["High"], "USD")
    assert [e.title for e in high] == ["Core CPI m/m"]           # no EUR, no holiday, no far-future
    both = events_between(ev, now_eff, end, ["High", "Medium"], "USD")
    assert {e.title for e in both} == {"Core CPI m/m", "Unemployment Claims"}


def test_weekend_window_covers_the_whole_monday_session():
    cfg = M.fast_cfg()
    sat = T.utc("2026-01-10 12:00")
    b = M.bundle_at(sat, T.hourly_until(sat, seed=11), cfg)
    start, end = session_window(b, cfg)
    assert start == pd.Timestamp("2026-01-11 22:00", tz="UTC") and end == pd.Timestamp("2026-01-12 22:00", tz="UTC")


def test_overlapping_calendar_feeds_do_not_duplicate_events():
    cfg, b = _bundle()
    with Http(calendar=CAL, rss_body=NEUTRAL):                     # both URLs return the same events
        rep = build_news_report(cfg, b)
    titles = [e.title for e in rep.events]
    assert titles.count("Core CPI m/m") == 1 and len(titles) == len(set(titles))


def test_static_fomc_fallback_is_used_only_when_every_feed_fails():
    cfg, b = _bundle()
    cfg["news"]["static_events"] = [
        {"date": "2026-01-14", "time_ny": "14:00", "title": "FOMC rate decision", "impact": "High"}]
    with Http(calendar=OSError("down"), rss_body=NEUTRAL):
        rep = build_news_report(cfg, b)
    assert not rep.calendar_ok and rep.event_today
    assert rep.events and rep.events[0].source == "static"
    assert any("FOMC bawaan" in n for n in rep.notes)
    with Http(calendar=CAL_QUIET, rss_body=NEUTRAL):               # feed works, nothing today
        rep2 = build_news_report(cfg, b)
    assert rep2.calendar_ok and not rep2.event_today and not rep2.events


# ------------------------------------------------------------------ headlines
def test_rss_parsing_splits_publisher_dedupes_and_drops_stale_items():
    cfg, b = _bundle()
    xml = rss(
        ("Fed signals rate cuts as dollar falls - Reuters", FRESH),
        ("Fed signals rate cuts as dollar falls - Reuters", FRESH),         # duplicate
        ("Old story about gold - Kitco", STALE),                           # older than 36h
    )
    heads = parse_rss(xml, "RSS")
    assert heads[0].title == "Fed signals rate cuts as dollar falls" and heads[0].source == "Reuters"
    with Http(calendar=CAL_QUIET, rss_body=xml):
        fresh, ok, _ = ns.fetch_headlines(cfg, b.now_utc)
    assert ok and len(fresh) == 1


def test_gold_lexicon_gets_direction_right_including_negation():
    assert score_headline("Fed signals rate cuts as dollar falls") > 0.5
    assert score_headline("Gold climbs as geopolitical tensions escalate") > 0.5     # war news is BULLISH for gold
    assert score_headline("Strong payrolls lift rate hike bets, gold slides") < -0.5
    assert score_headline("Gold falls as dollar rises") < -0.5
    assert score_headline("Rate cut hopes fade as inflation sticks") < 0             # negated -> bearish
    assert score_headline("Fed rules out rate cuts this year") < 0
    assert score_headline("Gold price today: what to know") == 0.0
    assert -1.0 <= score_headline("rate cuts dovish rate cuts dovish safe haven war war") <= 1.0


def test_lexicon_covers_commentary_style_headlines_without_double_counting():
    # styles seen in real feeds: selloff / bears / yearly lows / "continue to fall" / hyphenated rate-cut
    assert score_headline("Gold price forecast: deepening selloff nears major support") < 0
    assert score_headline("Gold outlook: XAU/USD bears face critical test at yearly lows") < 0
    assert score_headline("Gold rallies as Fed rate-cut bets build and dollar slides") > 0.8
    assert score_headline("Fed rate-hike fears lift Treasury yields; gold slips") < -0.5
    assert abs(score_headline("Will gold prices continue to fall? Trend analysis") + 0.3) < 1e-9   # counted once
    assert score_headline("Dollar index climbs as traders trim Fed cut bets") < 0


def test_price_action_words_do_not_cross_clause_boundaries():
    """Regression: 'rises' (dollar) used to be read as 'gold ... rises' across the word 'as'."""
    assert abs(score_headline("Gold falls as dollar rises") + 0.7) < 1e-9        # -0.3 (gold) -0.4 (dollar)
    assert abs(score_headline("Gold rises as dollar falls") - 0.7) < 1e-9
    assert abs(score_headline("Gold edges up while yields fall") - 0.4) < 1e-9   # only the yields clause scores


def test_relevance_filter_drops_mining_company_and_unrelated_news():
    from src.news_sentinel import is_relevant

    assert not is_relevant("Miner increases gold resource to 3.5 million ounces, targets self-funded PEA")
    assert not is_relevant("Gold tax plan could reach 12% as miners face proposed royalty")
    assert not is_relevant("Local bakery wins award for best croissant")
    assert is_relevant("Will gold prices continue to fall? Trend analysis")        # market piece, kept
    assert is_relevant("Rate-cut hopes fade as inflation sticks")                  # Fed driver without the word gold
    assert is_relevant("Miners rally as gold price hits record high")              # mentions the price, kept


def test_default_news_feeds_are_google_news_queries_and_all_encoded():
    cfg = M.fast_cfg()
    urls = cfg["news"]["rss_urls"]
    assert len(urls) >= 2 and all(u.startswith("https://news.google.com/rss/search?q=") for u in urls)
    assert not any(" " in u or "(" in u for u in urls)                              # properly URL-encoded
    assert not any("fxstreet" in u for u in urls)


def test_irrelevant_headlines_never_reach_the_report():
    cfg, b = _bundle()
    xml = rss(("Miner increases gold resource to 3.5 million ounces - Crux", FRESH),
              ("Gold falls as dollar rises - Reuters", FRESH))
    with Http(calendar=CAL_QUIET, rss_body=xml):
        rep = build_news_report(cfg, b)
    assert rep.n_headlines == 1                                    # the mining item was dropped


def test_aggregate_uses_the_strongest_headlines():
    hs = [Headline("a", "s", "", None, 0.9), Headline("b", "s", "", None, -0.1), Headline("c", "s", "", None, 0.0)]
    assert abs(aggregate(hs) - 0.4) < 1e-9 and aggregate([]) == 0.0


# --------------------------------------------------------------- display rules
def test_quiet_day_shows_no_headlines():
    cfg, b = _bundle()
    with Http(calendar=CAL_QUIET, rss_body=NEUTRAL):
        rep = build_news_report(cfg, b)
    assert rep.news_ok and not rep.show_news and rep.headlines == []


def test_event_day_shows_up_to_three_headlines_with_signs():
    cfg, b = _bundle()
    with Http(calendar=CAL, rss_body=BULLISH):
        rep = build_news_report(cfg, b)
    assert rep.event_today and rep.show_news
    assert 1 <= len(rep.headlines) <= cfg["news"]["max_headlines"]
    assert all(h.score > 0 for h in rep.headlines)


def test_extreme_sentiment_alone_shows_headlines():
    cfg, b = _bundle()
    with Http(calendar=CAL_QUIET, rss_body=BULLISH):
        rep = build_news_report(cfg, b)
    assert not rep.event_today and abs(rep.agg_score) >= cfg["news"]["extreme_score"] and rep.show_news


def test_every_source_down_never_raises_and_is_reported():
    cfg, b = _bundle()
    with Http(calendar=OSError("x"), rss_body=OSError("y")):
        rep = build_news_report(cfg, b)
    assert not rep.news_ok and not rep.calendar_ok and not rep.show_news
    assert any("radar berita tidak tersedia" in n for n in rep.notes)


# ------------------------------------------------------------------- the email
def test_email_shows_event_widening_news_and_graceful_messages():
    cfg, b = _bundle()
    fc = forecast_day(b, None, cfg, event_today=True)
    with Http(calendar=CAL, rss_body=BULLISH):
        rep = build_news_report(cfg, b)
    _, html_doc, text = build_forecast_email(b, fc, cfg, news=rep)
    assert "Core CPI m/m" in html_doc and "20:30 WIB" in html_doc          # 13:30 UTC = 20:30 WIB
    assert "Kisaran dilebarkan" in html_doc and "EVENT USD HARI INI" in text
    assert "[Bullish Emas]" in html_doc and "BERITA PENGGERAK" in text

    with Http(calendar=CAL_QUIET, rss_body=NEUTRAL):
        quiet = build_news_report(cfg, b)
    _, h2, t2 = build_forecast_email(b, forecast_day(b, None, cfg), cfg, news=quiet)
    assert "Tidak ada katalis berita signifikan" in h2 and "Event USD" not in h2

    with Http(calendar=OSError("x"), rss_body=OSError("y")):
        down = build_news_report(cfg, b)
    _, h3, t3 = build_forecast_email(b, forecast_day(b, None, cfg), cfg, news=down)
    assert "Radar berita tidak tersedia" in h3 and "Tidak ada katalis" not in h3

    _, h4, _ = build_forecast_email(b, forecast_day(b, None, cfg), cfg, news=None)   # radar off
    assert "Berita" not in h4
    for blob in (html_doc, text, h2, h3):
        assert " nan" not in blob.lower() and "None" not in blob


def test_event_day_widens_the_forecast_band_through_the_model():
    cfg, b = _bundle()
    base = forecast_day(b, None, cfg)
    ev = forecast_day(b, None, cfg, event_today=True)
    assert (ev.close_q[2] - ev.close_q[0]) > (base.close_q[2] - base.close_q[0])
    assert any("High-impact USD event" in n for n in ev.notes)


# ------------------------------------------------------------------ end to end
def test_main_runs_radar_marks_the_ledger_and_survives_dead_feeds():
    import main as main_mod

    hourly = T.hourly_until(T.NOW, seed=11)
    env = {"GMAIL_USER": "bot@gmail.com", "GMAIL_APP_PASSWORD": "pw", "EMAIL_TO": "me@gmail.com"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    orig_smtp = smtplib.SMTP_SSL
    smtplib.SMTP_SSL = T._FakeSMTP
    try:
        with tempfile.TemporaryDirectory() as tmp:
            led = str(Path(tmp) / "ledger.csv")
            args = ["--no-macro", "--ledger", led, "--now", "2026-01-13T22:30:00Z"]
            with T.FakeYahoo(lambda s, i, d: hourly):
                with Http(calendar=CAL, rss_body=BULLISH):                  # event day
                    assert main_mod.main(args) == 0
                body = L._text_of_last_mail()
                assert "EVENT USD HARI INI" in body and "Core CPI m/m" in body
                assert int(load_ledger(led).iloc[0]["event"]) == 1          # flagged for later analysis

                T._FakeSMTP.last = {}
                with Http(calendar=OSError("x"), rss_body=OSError("y")):    # all feeds dead
                    assert main_mod.main(args + ["--no-ledger"]) == 0
                assert "Radar berita tidak tersedia" in L._text_of_last_mail()
    finally:
        smtplib.SMTP_SSL = orig_smtp
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_old_ledger_without_event_column_still_loads():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, b = _bundle()
        from src.ledger import COLUMNS, empty_ledger, save_ledger, upsert_forecast

        led = upsert_forecast(empty_ledger(), forecast_day(b, None, cfg), b, b.now_utc)
        p = Path(tmp) / "old.csv"
        save_ledger(led.drop(columns=["event"]), p)                         # a Sprint 3 style file
        back = load_ledger(p)
        assert list(back.columns) == COLUMNS and pd.isna(back.iloc[0]["event"])


# ---------------------------------------------------------------- event study
def _daily_world(seed, planted_mult, n_days=700, n_events=22):
    import numpy as np

    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-06-01", periods=n_days)
    ev_idx = rng.choice(np.arange(30, n_days), size=n_events, replace=False)
    sigma = np.full(n_days, 0.008)
    sigma[ev_idx] *= planted_mult
    r = rng.normal(0, sigma)
    close = 2000 * np.exp(np.cumsum(r))
    candles = pd.DataFrame({"Close": close, "High": close * 1.004, "Low": close * 0.996,
                            "Open": close}, index=days)
    return candles, [str(days[i].date()) for i in ev_idx]


def test_event_study_recovers_the_planted_variance_ratio_with_enough_event_days():
    from src.evaluation import event_study

    cfg = M.fast_cfg()
    contains, detected = 0, 0
    for seed in range(20):                                           # no cherry-picked seed
        candles, dates = _daily_world(200 + seed, planted_mult=1.6, n_days=1400, n_events=90)   # truth = 2.56
        res = event_study(candles, dates, cfg, n_boot=800)
        lo, hi = res["ci"]
        contains += lo <= 2.56 <= hi
        detected += lo > 1.0
        assert abs(res["recommended_extra_var"] - max(res["var_ratio"] - 1, 0)) < 1e-12
    assert contains >= 16 and detected >= 19, (contains, detected)   # ~95% coverage, high power


def test_event_study_with_few_event_days_is_honestly_wide():
    from src.evaluation import event_study

    cfg = M.fast_cfg()
    widths = []
    for seed in range(6):
        candles, dates = _daily_world(300 + seed, planted_mult=1.6, n_events=22)
        lo, hi = event_study(candles, dates, cfg, n_boot=800)["ci"]
        widths.append(hi - lo)
    assert min(widths) > 1.0          # about 20 FOMC days cannot pin the ratio down: the tool must say so


def test_event_study_finds_nothing_when_there_is_no_effect():
    from src.evaluation import event_study

    cfg = M.fast_cfg()
    false_alarms = 0
    for seed in range(20):
        candles, dates = _daily_world(100 + seed, planted_mult=1.0, n_days=1400, n_events=90)
        false_alarms += event_study(candles, dates, cfg, n_boot=600)["ci"][0] > 1.0
    assert false_alarms <= 3                                          # ~2.5% expected


def test_event_study_refuses_tiny_samples():
    from src.data_fetcher import DataError
    from src.evaluation import event_study

    candles, dates = _daily_world(5, planted_mult=1.6)
    try:
        event_study(candles, dates[:3], M.fast_cfg())
        raise AssertionError("expected DataError")
    except DataError:
        pass


# ------------------------------------------------- scheduled event jump (engine)
def test_scheduled_jump_adds_the_intended_variance_at_the_right_hour():
    import numpy as np

    from src.quant_engine import event_segments, segment_cuts, simulate_day

    now = pd.Timestamp("2026-01-13 22:30", tz="UTC")
    end = pd.Timestamp("2026-01-14 22:00", tz="UTC")
    cuts = segment_cuts(now, end)
    w = np.full(len(cuts) - 1, 1.0 / (len(cuts) - 1))
    # CPI at 13:30 UTC lands in the segment that starts 13:00 UTC
    j = event_segments(cuts, w, [pd.Timestamp("2026-01-14 13:30", tz="UTC")], max_events=2)[0]
    assert cuts[j] <= pd.Timestamp("2026-01-14 13:30", tz="UTC") < cuts[j + 1]
    # events outside the window are ignored; no times -> one event near the variance midpoint
    assert event_segments(cuts, w, [pd.Timestamp("2026-02-01", tz="UTC")], 2) == []
    mid = event_segments(cuts, w, None, 2)
    assert len(mid) == 1 and abs(mid[0] - len(w) / 2) <= 1

    base = simulate_day(100.0, w, 0.006, 0.0, 0.0, 0.0, 0.0, 60000, np.random.default_rng(1))
    ev = simulate_day(100.0, w, 0.006, 0.0, 0.0, 0.0, 0.0, 60000, np.random.default_rng(1), scheduled={j: 0.006})
    v0, v1 = np.log(base["close"] / 100).var(), np.log(ev["close"] / 100).var()
    assert abs(v1 / v0 - 2.0) < 0.08                  # extra variance = sigma_event^2 = another 100%
    assert (ev["high"] - ev["low"]).mean() > (base["high"] - base["low"]).mean()


def test_event_flag_without_times_still_widens_the_band():
    cfg, b = _bundle()
    base = forecast_day(b, None, cfg)
    flagged = forecast_day(b, None, cfg, event_today=True)             # the --event CLI path (no times)
    assert (flagged.close_q[2] - flagged.close_q[0]) > (base.close_q[2] - base.close_q[0])


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
