"""Au-tomatic (Au = Aurum) - entry point. Sprint 2: data -> models -> forecast email.

Usage
-----
    python main.py --dry-run            # no email; writes out/preview.html + out/preview.txt
    python main.py                      # sends email (needs env vars below)
    python main.py --dry-run --now 2026-11-13T22:30:00Z   # replay a past moment
    python main.py --dry-run --data-check                 # only the Sprint 1 data check
    python main.py --dry-run --no-macro                   # skip DXY/US10Y features
    python main.py --dry-run --event                      # simulate a high-impact event day

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
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    now = pd.Timestamp(args.now, tz="UTC") if args.now else None

    failed = False
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
            fc = forecast_day(bundle, macro, cfg, event_today=args.event)
            log.info(
                "Forecast: close %.2f-%.2f | P(up) %.0f%% | regime %s",
                fc.close_q[0], fc.close_q[2], 100 * fc.p_up, fc.regime,
            )
            subject, html_body, text_body = build_forecast_email(bundle, fc, cfg)
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
        log.info("Dry run: wrote %s and preview.txt. Subject: %s", out / "preview.html", subject)
    else:
        user = os.environ.get("GMAIL_USER")
        password = os.environ.get("GMAIL_APP_PASSWORD")
        to = os.environ.get("EMAIL_TO")
        if not (user and password and to):
            log.error("Missing GMAIL_USER / GMAIL_APP_PASSWORD / EMAIL_TO environment variables.")
            return 2
        send_email(
            subject, html_body, text_body,
            user=user, password=password, to=to,
            host=cfg["email"]["smtp_host"], port=int(cfg["email"]["smtp_port"]),
        )
        log.info("Email sent.")  # intentionally no address in logs (public repo)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
