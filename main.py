"""Au-tomatic (Au = Aurum) - entry point. Sprint 2: data -> models -> forecast email.

Usage
-----
    python main.py --dry-run            # no email; writes out/preview.html + out/preview.txt
    python main.py                      # sends email (needs env vars below)
    python main.py --dry-run --now 2026-11-13T22:30:00Z   # replay a past moment
    python main.py --dry-run --data-check                 # only the Sprint 1 data check
    python main.py --dry-run --no-macro                   # skip DXY/US10Y features
    python main.py --dry-run --no-news                    # skip the calendar/news radar
    python main.py --dry-run --event                      # simulate a high-impact event day
    python main.py --dry-run --explain                    # also write out/report.html (every computation step)

Environment variables (GitHub Secrets in CI):
    TWELVEDATA_API_KEY                         (spot XAU/USD; without it the futures fallback is used)
    GMAIL_USER, GMAIL_APP_PASSWORD, EMAIL_TO   (EMAIL_TO may be comma-separated)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.data_fetcher import DataError, fetch_macro_daily, get_gold_bundle
from src.email_dispatcher import (
    build_data_check_email,
    build_error_email,
    build_forecast_email,
    send_email,
)
from src.ledger import live_accuracy, load_ledger, save_ledger, settle, upsert_forecast
from src.news_sentinel import build_news_report
from src.quant_engine import forecast_day

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("au-tomatic")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Au-tomatic: XAUUSD daily candle range notifier")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--dry-run", action="store_true", help="Write preview files instead of emailing.")
    p.add_argument("--now", default=None, help="Override 'now' (ISO, UTC) to replay a moment.")
    p.add_argument("--out", default=str(ROOT / "out"), help="Directory for dry-run previews.")
    p.add_argument("--data-check", action="store_true", help="Only run the Sprint 1 data check email.")
    p.add_argument("--no-macro", action="store_true", help="Skip DXY / US10Y features.")
    p.add_argument("--event", action="store_true", help="Treat today as a high-impact event day.")
    p.add_argument("--explain", action="store_true", help="Also write out/report.html and out/report.txt.")
    p.add_argument("--no-news", action="store_true", help="Skip the calendar and news radar.")
    p.add_argument("--no-ledger", action="store_true", help="Do not read or write the forecast ledger.")
    p.add_argument("--ledger", default=None, help="Ledger CSV path (default from config.yaml).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    now = pd.Timestamp(args.now, tz="UTC") if args.now else None

    failed = False
    fc = None
    ledger = None
    event_flag = False
    ledger_path = Path(args.ledger) if args.ledger else ROOT / cfg["ledger"]["path"]
    try:
        bundle = get_gold_bundle(cfg, now_utc=now)
        log.info("Data OK: %s | target session %s", bundle.source_label, bundle.target_session.date())
        if args.data_check:
            subject, html_body, text_body = build_data_check_email(bundle, cfg)
        else:
            macro = None
            if not args.no_macro:
                macro, macro_notes = fetch_macro_daily(cfg, bundle.now_utc)
                bundle.warnings.extend(macro_notes)
            news = None
            if cfg["news"].get("enabled", True) and not args.no_news:
                news = build_news_report(cfg, bundle)          # never raises; failures become notes
            event_flag = bool(args.event or (news is not None and news.event_today))
            event_times = [e.time_utc for e in news.high_events] if news is not None else None
            fc = forecast_day(bundle, macro, cfg, event_today=event_flag, explain=args.explain,
                              event_times=event_times)
            log.info(  # relative numbers only: Actions logs of a public repo are public
                "Forecast: close half-width about $%.0f | P(up) %.0f%% | regime %s",
                (fc.close_q[2] - fc.close_q[0]) / 2, 100 * fc.p_up, fc.regime,
            )
            accuracy = None
            if not args.no_ledger:
                try:  # the ledger is a bonus: it must never block the forecast email
                    ledger = load_ledger(ledger_path)
                    ledger, n_set = settle(ledger, bundle.candles)
                    accuracy = live_accuracy(ledger, cfg["ledger"]["window"], cfg["ledger"]["min_n"])
                    log.info("Ledger: %d rows, settled %d new, live n=%d", len(ledger), n_set, accuracy["n"])
                except Exception as exc:  # noqa: BLE001
                    ledger = None
                    log.warning("Ledger skipped (%s: %s).", type(exc).__name__, exc)
            subject, html_body, text_body = build_forecast_email(bundle, fc, cfg, accuracy=accuracy, news=news)
            if args.explain:
                # An optional extra must never block the forecast email itself.
                try:
                    from src.report import build_report

                    rep_html, rep_txt = build_report(bundle, fc, cfg)
                    out_dir = Path(args.out)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "report.html").write_text(rep_html, encoding="utf-8")
                    (out_dir / "report.txt").write_text(rep_txt, encoding="utf-8")
                    log.info("Computation report written to %s", out_dir / "report.html")
                except Exception as exc:  # noqa: BLE001
                    hint = " Run: python -m pip install -r requirements.txt" if isinstance(exc, ImportError) else ""
                    log.warning("Computation report skipped (%s: %s).%s", type(exc).__name__, exc, hint)
    except DataError as exc:
        failed = True
        log.error("Pipeline failed: %s", exc)
        subject, html_body, text_body = build_error_email(str(exc), cfg)
    except Exception as exc:  # unexpected bug: still tell the user, never stay silent
        failed = True
        log.exception("Unexpected error")
        subject, html_body, text_body = build_error_email(f"Unexpected error: {exc!r}", cfg)

    if args.dry_run:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "preview.html").write_text(html_body, encoding="utf-8")
        (out / "preview.txt").write_text(f"SUBJECT: {subject}\n\n{text_body}\n", encoding="utf-8")
        in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
        log.info("Dry run: wrote %s and preview.txt.%s", out / "preview.html",
                 "" if in_ci else f" Subject: {subject}")      # subject contains price levels
    else:
        user = os.environ.get("GMAIL_USER")
        password = os.environ.get("GMAIL_APP_PASSWORD")
        to = os.environ.get("EMAIL_TO")
        if not (user and password and to):
            log.error("Missing GMAIL_USER / GMAIL_APP_PASSWORD / EMAIL_TO environment variables.")
            return 2
        try:
            ok, bad = send_email(
                subject, html_body, text_body,
                user=user, password=password, to=to,
                host=cfg["email"]["smtp_host"], port=int(cfg["email"]["smtp_port"]),
                hide_recipients=bool(cfg["email"].get("hide_recipients", True)),
            )
        except ValueError as exc:
            log.error("%s Check the EMAIL_TO secret.", exc)
            return 2
        # intentionally no addresses in logs (public repo)
        log.info("Email sent to %d recipient(s).", ok)
        if bad:
            log.error("%d recipient(s) were refused by the mail server. Check the EMAIL_TO secret.", bad)
            failed = True
        if ok and fc is not None and ledger is not None:        # record only what was actually sent
            try:
                save_ledger(upsert_forecast(ledger, fc, bundle, bundle.now_utc, event_flag=event_flag), ledger_path)
                log.info("Ledger updated.")
            except Exception as exc:  # noqa: BLE001
                log.warning("Ledger not saved (%s: %s).", type(exc).__name__, exc)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
