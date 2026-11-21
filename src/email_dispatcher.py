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
from .quant_engine import Forecast

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
<tr><td style="padding:16px 20px 0 20px;font-size:11px;letter-spacing:1px;color:#b45309;">AU-TOMATIC &middot; AURUM</td></tr>
<tr><td style="padding:4px 20px 8px 20px;font-size:18px;font-weight:bold;">{html.escape(title)}</td></tr>
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


def _usd(x: float, decimals: int = 0) -> str:
    return f"${x:,.{decimals}f}"


def _signed(x: float) -> str:
    sign = "+" if x >= 0 else "\u2212"
    return f"{sign}${abs(x):,.0f}"


_BIAS_COLOR = {"Bullish": "#15803d", "Bearish": "#b91c1c", "Netral": "#4b5563"}


def build_forecast_email(bundle: DataBundle, fc: Forecast, cfg: dict) -> tuple[str, str, str]:
    """Final forecast email: short, one column, readable on a phone screen."""
    tz = cfg["email"]["display_tz"]
    c15, c50, c85 = fc.close_q
    h15, h50, h85 = fc.high_q
    l15, l50, l85 = fc.low_q
    o = fc.open_price
    futures = bundle.is_futures
    tag = " (Futures, basis != spot)" if futures else ""

    # The bias word is only informative when it leaves the Netral zone, so it is omitted otherwise.
    core = f"Close {_usd(c15)}\u2013{_usd(c85)} | H ~{_usd(h50)} | L ~{_usd(l50)}"
    if fc.bias != "Netral":
        core = f"{fc.bias} | {core}"
    subject = f"{cfg['email']['subject_prefix']} {core}{tag}"

    open_label = "Open candle" if fc.open_known else "Harga acuan (open belum ada)"
    spot_label = "XAUUSD spot" if not futures else "Futures GC=F (bukan spot)"
    color = _BIAS_COLOR.get(fc.bias, "#4b5563")

    kal_txt = (
        f"Momentum Kalman: {fc.kalman_label} "
        f"({fc.kalman_velocity:+.1f} $/hari)".replace("-", "\u2212")
    )
    context = [
        kal_txt,
        f"Rezim pasar: {fc.regime}; kisaran hari ini {fc.vol_label} dibanding 30 hari terakhir.",
    ]
    context.append(
        f"Range High\u2013Low rata-rata: {_usd(fc.avg_range_5)} (5 hari), {_usd(fc.avg_range_20)} (20 hari); "
        f"perkiraan hari ini {_usd(fc.high_q[1] - fc.low_q[1])}."
    )
    if "dxy_ret" in fc.features_used or "tnx_chg" in fc.features_used:
        context.append("Rezim juga memperhitungkan DXY dan US10Y.")

    rows = [
        (
            '<tr><td style="padding:8px 20px 0 20px;font-size:13px;color:#6b7280;">'
            f"Kisaran close D1 hari ini (70%)</td></tr>"
            '<tr><td style="padding:0 20px;font-size:26px;font-weight:bold;">'
            f"{_usd(c15, 2)} \u2013 {_usd(c85, 2)}</td></tr>"
            '<tr><td style="padding:0 20px 8px 20px;font-size:14px;">'
            f"{_signed(c15 - o)} s/d {_signed(c85 - o)} dari open ({_usd(o, 2)})</td></tr>"
        ),
        _row("Perkiraan High", f"{_usd(h50)} <span style='font-weight:normal;font-size:13px;'>"
                               f"(kisaran {_usd(h15)} \u2013 {_usd(h85)}; {_signed(h50 - o)} dari open)</span>"),
        _row("Perkiraan Low", f"{_usd(l50)} <span style='font-weight:normal;font-size:13px;'>"
                              f"(kisaran {_usd(l15)} \u2013 {_usd(l85)}; {_signed(l50 - o)} dari open)</span>"),
        _row("Median close / peluang candle naik",
             f"{_usd(c50)} &middot; <span style='color:{color};'>{fc.p_up:.0%} ({html.escape(fc.bias)})</span>"),
    ]
    rows_html = "".join(rows)
    rows_html += (
        '<tr><td style="padding:6px 20px;font-size:13px;color:#374151;">'
        "<ul style='margin:0 0 0 18px;padding:0;'>"
        + "".join(f"<li>{html.escape(c)}</li>" for c in context)
        + "</ul></td></tr>"
    )

    all_warnings = list(bundle.warnings) + list(fc.notes)
    if all_warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in all_warnings)
        rows_html += (
            '<tr><td style="padding:8px 20px;font-size:12px;color:#92400e;background:#fffbeb;">'
            f"<b>Catatan</b><ul style='margin:4px 0 0 18px;padding:0;'>{items}</ul></td></tr>"
        )

    footer = (
        f"{html.escape(spot_label)} &middot; sumber: {html.escape(bundle.source_label)} &middot; "
        f"dibuat {_to_local(bundle.now_utc, tz)} WIB. "
        "Radar berita/kalender belum aktif. Model belum divalidasi walk-forward. "
        f"{DISCLAIMER}"
    )
    body = _wrap_html("Perkiraan candle D1 XAUUSD hari ini", rows_html, footer)

    plain = [
        f"AU-TOMATIC - perkiraan candle D1 XAUUSD ({bundle.target_session.date()})",
        "",
        f"Kisaran close (70%) : {_usd(c15, 2)} - {_usd(c85, 2)}",
        f"  relatif open      : {_signed(c15 - o)} s/d {_signed(c85 - o)} (open {_usd(o, 2)})",
        f"Perkiraan High      : {_usd(h50)} (kisaran {_usd(h15)} - {_usd(h85)})",
        f"Perkiraan Low       : {_usd(l50)} (kisaran {_usd(l15)} - {_usd(l85)})",
        f"Median close        : {_usd(c50)} | peluang candle naik {fc.p_up:.0%} ({fc.bias})",
        "",
    ] + [f"- {c}" for c in context]
    if all_warnings:
        plain += ["", "CATATAN:"] + [f"- {w}" for w in all_warnings]
    plain += ["", f"Sumber: {bundle.source_label}", DISCLAIMER]
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
