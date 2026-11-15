"""Entry point. Sprint 1: data pipeline + data-check email.

Usage
-----
    python main.py --dry-run            # no email; writes out/preview.html + out/preview.txt
    python main.py                      # sends email (needs env vars below)
    python main.py --dry-run --now 2026-11-13T22:30:00Z   # replay a past moment

Environment variables (GitHub Secrets in CI):
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

from src.data_fetcher import DataError, get_gold_bundle
from src.email_dispatcher import build_data_check_email, build_error_email, send_email

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("sentinel")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAUUSD Quant Sentinel")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--dry-run", action="store_true", help="Write preview files instead of emailing.")
    p.add_argument("--now", default=None, help="Override 'now' (ISO, UTC) to replay a moment.")
    p.add_argument("--out", default=str(ROOT / "out"), help="Directory for dry-run previews.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    now = pd.Timestamp(args.now, tz="UTC") if args.now else None

    failed = False
    try:
        bundle = get_gold_bundle(cfg, now_utc=now)
        subject, html_body, text_body = build_data_check_email(bundle, cfg)
        log.info("Data OK: %s | target session %s", bundle.source_label, bundle.target_session.date())
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
