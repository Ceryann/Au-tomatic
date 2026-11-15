"""Email layer: builds multipart (HTML + plain text) messages and sends via Gmail SMTP.

Sprint 1 only produces a *data check* email (no forecast yet). The layout is the
final single-column, 600px, inline-CSS design so Sprint 2 only swaps the content.
"""

from __future__ import annotations

import html
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

import pandas as pd

from .data_fetcher import DataBundle

DISCLAIMER = "Estimasi statistik, bukan nasihat finansial."


def _fmt_price(x: float) -> str:
    return f"${x:,.2f}"


def _to_local(ts: pd.Timestamp, tz: str) -> str:
    return ts.tz_convert(ZoneInfo(tz)).strftime("%d %b %Y %H:%M")


def _wrap_html(title: str, body_rows: str, footer: str) -> str:
    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#f4f4f5;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f5;">
<tr><td align="center" style="padding:16px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"
 style="max-width:600px;background:#ffffff;border-radius:8px;font-family:Arial,Helvetica,sans-serif;color:#111827;">
<tr><td style="padding:20px 20px 8px 20px;font-size:18px;font-weight:bold;">{html.escape(title)}</td></tr>
{body_rows}
<tr><td style="padding:12px 20px 20px 20px;font-size:11px;color:#6b7280;border-top:1px solid #e5e7eb;">{footer}</td></tr>
</table>
</td></tr></table>
</body></html>"""


def _row(label: str, value: str) -> str:
    return (
        '<tr><td style="padding:4px 20px;font-size:14px;">'
        f'<span style="color:#6b7280;">{html.escape(label)}</span><br>'
        f'<span style="font-size:16px;font-weight:bold;">{value}</span></td></tr>'
    )


def build_data_check_email(bundle: DataBundle, cfg: dict) -> tuple[str, str, str]:
    """Return (subject, html, plain_text) for the Sprint 1 pipeline check."""
    tz = cfg["email"]["display_tz"]
    last = bundle.candles.iloc[-1]
    last_label = bundle.candles.index[-1].date()
    futures_tag = " (Futures, basis != spot)" if bundle.is_futures else ""

    subject = (
        f"{cfg['email']['subject_prefix']} TES DATA OK | "
        f"Close {last_label}: {_fmt_price(float(last['Close']))}{futures_tag}"
    )

    open_txt = _fmt_price(bundle.session_open) if bundle.session_open else "belum ada"
    bars_txt = "-" if pd.isna(last["bars"]) else f"{int(last['bars'])} bar 1h"

    lines = [
        ("Sesi yang akan diprediksi", str(bundle.target_session.date())),
        ("Harga acuan saat ini", f"{_fmt_price(bundle.reference_price)}"
                                 f" ({_to_local(bundle.reference_time_utc, tz)} WIB)"),
        ("Open sesi ini", open_txt),
        (f"Candle D1 lengkap terakhir ({last_label})",
         f"O {_fmt_price(float(last['Open']))} · H {_fmt_price(float(last['High']))}<br>"
         f"L {_fmt_price(float(last['Low']))} · C {_fmt_price(float(last['Close']))}"),
        ("Jumlah candle / bar", f"{len(bundle.candles)} candle · {bars_txt}"),
        ("Sumber data", html.escape(bundle.source_label)),
    ]

    plain = [
        "TES PIPELINE DATA (Sprint 1) - ini BUKAN prediksi.",
        "",
        f"Sesi yang akan diprediksi : {bundle.target_session.date()}",
        f"Harga acuan saat ini      : {_fmt_price(bundle.reference_price)} "
        f"({_to_local(bundle.reference_time_utc, tz)} WIB)",
        f"Open sesi ini             : {open_txt}",
        f"Candle D1 terakhir {last_label} : O {last['Open']:.2f} H {last['High']:.2f} "
        f"L {last['Low']:.2f} C {last['Close']:.2f}",
        f"Jumlah candle             : {len(bundle.candles)}",
        f"Sumber data               : {bundle.source_label}",
    ]

    rows = "".join(_row(k, v) for k, v in lines)
    if bundle.warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in bundle.warnings)
        rows += (
            '<tr><td style="padding:8px 20px;font-size:13px;color:#92400e;background:#fffbeb;">'
            f"<b>Peringatan</b><ul style='margin:4px 0 0 18px;padding:0;'>{items}</ul></td></tr>"
        )
        plain += ["", "PERINGATAN:"] + [f"- {w}" for w in bundle.warnings]
    plain += ["", DISCLAIMER]

    body = _wrap_html(
        "Tes pipeline data (Sprint 1) - bukan prediksi",
        rows,
        f"Dibuat {_to_local(bundle.now_utc, tz)} WIB. {DISCLAIMER}",
    )
    return subject, body, "\n".join(plain)


def build_error_email(error_message: str, cfg: dict) -> tuple[str, str, str]:
    """Email sent instead of numbers when the pipeline fails (never send stale numbers)."""
    subject = f"{cfg['email']['subject_prefix']} PREDIKSI TIDAK TERSEDIA | pipeline gagal"
    safe = html.escape(error_message)
    rows = (
        '<tr><td style="padding:8px 20px;font-size:14px;">'
        "Prediksi hari ini <b>tidak dikirim</b> karena data tidak lolos validasi atau "
        "pipeline gagal. Jangan memakai angka lama.</td></tr>"
        f'<tr><td style="padding:8px 20px;font-size:13px;color:#991b1b;background:#fef2f2;">{safe}</td></tr>'
    )
    body = _wrap_html("Pipeline gagal", rows, DISCLAIMER)
    plain = (
        "PIPELINE GAGAL - prediksi hari ini tidak dikirim.\n\n"
        f"Alasan: {error_message}\n\n{DISCLAIMER}"
    )
    return subject, body, plain


def send_email(
    subject: str,
    html_body: str,
    text_body: str,
    *,
    user: str,
    password: str,
    to: str,
    host: str = "smtp.gmail.com",
    port: int = 465,
) -> None:
    """Send a multipart/alternative message over SMTP-SSL. Never logs addresses."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.attach(MIMEText(text_body, "plain", "utf-8"))   # fallback first
    msg.attach(MIMEText(html_body, "html", "utf-8"))    # preferred part last

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
        server.login(user, password)
        server.sendmail(user, [a.strip() for a in to.split(",")], msg.as_string())
