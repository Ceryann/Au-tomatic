"""Check which calendar and news feeds work from YOUR machine (no API key, no credits).

    python tools/probe_news.py

Feeds change over time. A failing feed is simply skipped by the bot, but this shows which ones to
replace in config.yaml (news.calendar_urls / news.rss_urls).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import news_sentinel as ns  # noqa: E402


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    n = cfg["news"]
    print("=== Economic calendar ===")
    for url in n["calendar_urls"]:
        try:
            ev = ns.parse_calendar(ns._http_get(url, n["timeout_seconds"]))
            usd = [e for e in ev if e.country == n["event_currency"] and e.impact in ("High", "Medium")]
            print(f"OK    {len(ev):3d} events, {len(usd):2d} USD High/Medium   {url}")
            for e in usd[:5]:
                print(f"        {e.time_utc:%a %d %b %H:%M} UTC  {e.impact:6s} {e.title}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {type(exc).__name__}: {str(exc)[:70]}   {url}")

    print("\n=== News feeds (RSS) ===")
    for url in n["rss_urls"]:
        try:
            heads = ns.parse_rss(ns._http_get(url, n["timeout_seconds"]), "RSS")
            rel = [h for h in heads if ns.is_relevant(h.title)]
            for h in rel:
                h.score = ns.score_headline(h.title)
            scored = sorted((h for h in rel if h.score != 0), key=lambda h: -abs(h.score))
            print(f"OK    {len(heads):3d} items, {len(rel):3d} relevant, {len(scored):3d} with a score   {url[:80]}")
            for h in scored[:5]:
                print(f"        [{h.score:+.1f}] {h.title[:95]}")
            if not scored:
                for h in rel[:3]:
                    print(f"        [ 0.0] {h.title[:95]}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {type(exc).__name__}: {str(exc)[:70]}   {url[:80]}")
    print("\nScores: positive = bullish for gold, negative = bearish (heuristic lexicon, not validated).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
