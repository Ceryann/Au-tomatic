"""Email layer: builds multipart (HTML + plain text) messages and sends via Gmail SMTP.

Sprint 1 only produces a *data check* email (no forecast yet). The layout is the
final single-column, 600px, inline-CSS design so Sprint 2 only swaps the content.
"""

from __future__ import annotations

import base64
import html
import logging
import re
import smtplib
import ssl
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

import pandas as pd

from .data_fetcher import DataBundle, session_end_utc
from .news_sentinel import NewsReport
from .notes import WARN, note_level, note_title
from .quant_engine import Forecast

log = logging.getLogger(__name__)

# Transient network/server hiccups worth retrying (a wrong password is NOT one of them).
_RETRYABLE = (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError,
              ConnectionError, TimeoutError, ssl.SSLError)

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


def news_time(ts: pd.Timestamp, tz: str) -> str:
    return ts.tz_convert(ZoneInfo(tz)).strftime("%H:%M")


def _usd(x: float, decimals: int = 0) -> str:
    return f"${x:,.{decimals}f}"


def _signed(x: float) -> str:
    sign = "+" if x >= 0 else "\u2212"
    return f"{sign}${abs(x):,.0f}"


_BIAS_COLOR = {"Bullish": "#15803d", "Bearish": "#b91c1c", "Netral": "#4b5563"}


# --------------------------------------------------------------------------
# Forecast-email building blocks: timing banner, range bar, grouped notes
# --------------------------------------------------------------------------
_BAR_TRACK, _BAR_RANGE, _BAR_OPEN, _BAR_REF = "#e5e7eb", "#f59e0b", "#374151", "#2563eb"
_BAR_CELLS = 50                      # 50 cells x 2% = the full width; plain table cells survive Gmail


def _hm(hours: float) -> str:
    """'7 jam 57 menit', '45 menit' (never negative)."""
    minutes = max(int(round(hours * 60)), 0)
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h} jam {m} menit"
    return f"{h} jam" if h else f"{m} menit"


def run_timing(bundle: DataBundle, fc: Forecast, cfg: dict) -> dict | None:
    """Where in the candle the run happened: hours since it opened, hours left, share of the day's
    variance still to come. None if anything is missing (the email then simply omits the timing)."""
    try:
        tz, hour = cfg["market"]["candle_cutoff_tz"], cfg["market"]["candle_cutoff_hour"]
        start = session_end_utc(bundle.target_session - pd.Timedelta(days=1), tz, hour)
        end = session_end_utc(bundle.target_session, tz, hour)
        elapsed = (bundle.now_utc - start).total_seconds() / 3600.0
        left = (end - bundle.now_utc).total_seconds() / 3600.0
        late_after = float(cfg["email"].get("late_run_hours", 2.0))
        return {"elapsed_h": max(elapsed, 0.0), "left_h": max(left, 0.0),
                "rem_frac": float(fc.remaining_var_frac), "late": elapsed >= late_after}
    except Exception as exc:  # noqa: BLE001 - optional block, must never break the main email
        log.warning("Run timing unavailable (%s).", type(exc).__name__)
        return None


def _ref_time_txt(bundle: DataBundle, tz: str) -> str:
    """Clock time of the reference price; with the date too when it is not from the last hours."""
    age_h = (bundle.now_utc - bundle.reference_time_utc).total_seconds() / 3600.0
    return _to_local(bundle.reference_time_utc, tz) if age_h > 12 else news_time(bundle.reference_time_utc, tz)


def _basis_text(bundle: DataBundle, timing: dict | None, tz: str) -> str:
    base = (f"Dihitung {news_time(bundle.now_utc, tz)} WIB dari harga terakhir "
            f"{_usd(bundle.reference_price, 2)} (jam {_ref_time_txt(bundle, tz)} WIB)")
    if timing is None:
        return base + "."
    return (f"{base}; sisa sesi sekitar {_hm(timing['left_h'])}, "
            f"kira-kira {timing['rem_frac']:.0%} variansi harian masih tersisa.")


def _late_text(bundle: DataBundle, timing: dict, tz: str) -> str:
    return (f"Run terlambat. Dibuat {news_time(bundle.now_utc, tz)} WIB, {_hm(timing['elapsed_h'])} setelah "
            f"candle dibuka. Kisaran di bawah dihitung dari harga terakhir dan sisa sesi, bukan kisaran "
            f"pagi, jadi bisa berbeda dari email sebelumnya hari ini.")


def _late_banner_html(text: str) -> str:
    head, _, rest = text.partition(". ")
    return ('<tr><td style="padding:6px 20px;font-size:13px;color:#92400e;background:#fffbeb;'
            f'border-top:1px solid #fde68a;border-bottom:1px solid #fde68a;"><b>{html.escape(head)}.</b> '
            f'{html.escape(rest)}</td></tr>')


def _range_bar_html(lo: float, hi: float, o: float, ref: float, open_known: bool) -> str:
    """One thin bar: the 70% close range in amber, with the open and the latest price as markers.
    Plain table cells with a background colour, so it renders in Gmail without images or SVG."""
    pts = [lo, hi, o, ref]
    span = max(pts) - min(pts)
    if not span > 0:
        return ""
    a, b = min(pts) - span * 0.10, max(pts) + span * 0.10

    def idx(x: float) -> int:
        return min(_BAR_CELLS - 1, max(0, int((x - a) / (b - a) * _BAR_CELLS)))

    i_lo, i_hi, i_o, i_ref = idx(lo), idx(hi), idx(o), idx(ref)
    cells = []
    for i in range(_BAR_CELLS):
        color = (_BAR_REF if i == i_ref else _BAR_OPEN if (open_known and i == i_o)
                 else _BAR_RANGE if i_lo <= i <= i_hi else _BAR_TRACK)
        cells.append(f'<td width="2%" height="12" bgcolor="{color}" '
                     f'style="background:{color};font-size:0;line-height:0;">&nbsp;</td>')
    open_txt = (f'<span style="color:{_BAR_OPEN};">&#9632;</span> open {_usd(o, 2)}'
                if open_known else "")
    legend = (f'<span style="color:{_BAR_RANGE};">&#9632;</span> kisaran close 70% &nbsp; '
              + (open_txt + " &nbsp; " if open_txt else "")
              + f'<span style="color:{_BAR_REF};">&#9632;</span> harga terakhir {_usd(ref, 2)}')
    return ('<tr><td style="padding:2px 20px 4px 20px;">'
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            'style="table-layout:fixed;border-collapse:collapse;"><tr>' + "".join(cells) + "</tr></table>"
            f'<div style="font-size:11px;color:#6b7280;padding-top:4px;">{legend}</div></td></tr>')


def _bias_basis_text(bundle: DataBundle, fc: Forecast, late: bool) -> str:
    """One plain sentence under the headline: what the Bullish/Bearish/Netral word is actually based on.
    The word is a thresholded P(close > open) given the latest price, so it follows where the price sits
    relative to the open; it is not a direction call (the backtest finds no directional skill)."""
    rem = f"sisa variansi harian {fc.remaining_var_frac:.0%}"
    if fc.open_known:
        delta = bundle.reference_price - fc.open_price
        side = ("hampir sama dengan" if abs(delta) < 0.5
                else f"{_usd(abs(delta), 2)} {'di atas' if delta > 0 else 'di bawah'}")
        first = (f"Dasar label: harga terakhir {_usd(bundle.reference_price, 2)}, {side} "
                 f"open {_usd(fc.open_price, 2)}; {rem}.")
    else:
        first = f"Dasar label: candle belum dibuka, peluang dihitung terhadap harga acuan; {rem}."
    second = ("Ini posisi harga di tengah sesi, bukan prediksi arah dari awal sesi." if late else
              "Backtest: arah harian tidak bisa ditebak, jadi label ini ringkasan peluang, bukan sinyal.")
    return f"{first} {second}"


def _dedupe_notes(notes) -> list:
    seen, out = set(), []
    for n in notes:
        if str(n) not in seen:
            seen.add(str(n))
            out.append(n)
    return out


def _notes_html(notes: list) -> str:
    """Grouped notes: 'Perlu diperhatikan' (amber) first, then 'Info' (grey), each with a bold title."""
    if not notes:
        return ""
    groups = [
        ("Perlu diperhatikan", [n for n in notes if note_level(n) == WARN], "#d97706", "#fffbeb", "#92400e"),
        ("Info", [n for n in notes if note_level(n) != WARN], "#9ca3af", "#f3f4f6", "#374151"),
    ]
    out = []
    for heading, items, bar, bg, fg in groups:
        if not items:
            continue
        body = "".join(
            f'<div style="border-left:3px solid {bar};padding:1px 0 1px 10px;margin:6px 0;">'
            + (f"<b>{html.escape(note_title(n))}</b><br>" if note_title(n) else "")
            + f"{html.escape(str(n))}</div>"
            for n in items
        )
        out.append(f'<tr><td style="padding:8px 20px;font-size:12px;color:{fg};background:{bg};">'
                   f'<b style="font-size:11px;letter-spacing:.5px;text-transform:uppercase;">{heading}</b>{body}</td></tr>')
    return "".join(out)


def _notes_plain(notes: list) -> list[str]:
    lines: list[str] = []
    for heading, items in (("PERLU DIPERHATIKAN:", [n for n in notes if note_level(n) == WARN]),
                           ("INFO:", [n for n in notes if note_level(n) != WARN])):
        if items:
            lines += ([""] if lines else []) + [heading] + [
                f"- {note_title(n)}: {n}" if note_title(n) else f"- {n}" for n in items]
    return lines


def build_forecast_email(
    bundle: DataBundle, fc: Forecast, cfg: dict, accuracy: dict | None = None,
    news: NewsReport | None = None,
) -> tuple[str, str, str]:
    """Final forecast email: short, one column, readable on a phone screen."""
    tz = cfg["email"]["display_tz"]
    c15, c50, c85 = fc.close_q
    h15, h50, h85 = fc.high_q
    l15, l50, l85 = fc.low_q
    o = fc.open_price
    futures = bundle.is_futures
    tag = " (Futures, basis != spot)" if futures else ""

    timing = run_timing(bundle, fc, cfg)
    is_late = bool(timing and timing["late"])
    bias_txt = f"{fc.bias} \u00b7 tengah sesi" if is_late else fc.bias     # a late run reads the price so far
    bias_basis = _bias_basis_text(bundle, fc, is_late)

    # The bias word is only informative when it leaves the Netral zone, so it is omitted otherwise.
    core = f"Close {_usd(c15)}\u2013{_usd(c85)} | H ~{_usd(h50)} | L ~{_usd(l50)}"
    if fc.bias != "Netral":
        core = f"{fc.bias}{' (tengah sesi)' if is_late else ''} | {core}"
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
    if accuracy is not None:
        if accuracy.get("ready"):
            context.append(
                f"Akurasi live {accuracy['n']} hari terakhir: close {accuracy['close']:.0%}, "
                f"high {accuracy['high']:.0%}, low {accuracy['low']:.0%} berada di dalam kisaran (target 70%)."
            )
        else:
            context.append(
                f"Akurasi live: mengumpulkan data ({accuracy.get('n', 0)}/{accuracy.get('min_n', 10)} hari)."
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
             f"{_usd(c50)} &middot; <span style='color:{color};'>{fc.p_up:.0%} ({html.escape(bias_txt)})</span>"),
        f'<tr><td style="padding:0 20px 8px 20px;font-size:11px;color:#6b7280;">{html.escape(bias_basis)}</td></tr>',
    ]
    late_txt = _late_text(bundle, timing, tz) if is_late else ""
    basis_txt = _basis_text(bundle, timing, tz)
    try:
        bar_html = _range_bar_html(c15, c85, o, bundle.reference_price, fc.open_known)
    except Exception as exc:  # noqa: BLE001 - decoration only
        log.warning("Range bar skipped (%s).", type(exc).__name__)
        bar_html = ""
    rows_html = (
        (_late_banner_html(late_txt) if late_txt else "")
        + rows[0] + bar_html
        + f'<tr><td style="padding:0 20px 6px 20px;font-size:11px;color:#6b7280;">{html.escape(basis_txt)}</td></tr>'
        + "".join(rows[1:])
    )
    rows_html += (
        '<tr><td style="padding:6px 20px;font-size:13px;color:#374151;">'
        "<ul style='margin:0 0 0 18px;padding:0;'>"
        + "".join(f"<li>{html.escape(c)}</li>" for c in context)
        + "</ul></td></tr>"
    )

    # ---- calendar events and news (only when the radar ran)
    news_plain: list[str] = []
    if news is not None:
        if news.events:
            items = "".join(
                f"<li>{news_time(e_.time_utc, tz)} WIB: {html.escape(e_.title)} ({html.escape(e_.impact)})</li>"
                for e_ in news.events
            )
            widened = (" Kisaran dilebarkan karena risiko lompatan." if news.event_today else "")
            rows_html += (
                '<tr><td style="padding:6px 20px;font-size:13px;color:#374151;">'
                f"<b>Event USD hari ini</b><ul style='margin:2px 0 0 18px;padding:0;'>{items}</ul>"
                f"<span style='font-size:12px;color:#6b7280;'>{widened.strip()}</span></td></tr>"
            )
            news_plain += ["EVENT USD HARI INI:"] + [
                f"- {news_time(e_.time_utc, tz)} WIB: {e_.title} ({e_.impact})" for e_ in news.events
            ] + ([widened.strip()] if widened else [])
        if not news.news_ok:
            body_news, plain_news = "Radar berita tidak tersedia hari ini.", ["Radar berita tidak tersedia hari ini."]
        elif news.show_news and news.headlines:
            li = "".join(
                f"<li>[{'Bullish' if h.score > 0 else 'Bearish'} Emas] "
                + (f"<a href='{html.escape(h.link, quote=True)}'>{html.escape(h.title)}</a>" if h.link else html.escape(h.title))
                + f" ({html.escape(h.source)})</li>" for h in news.headlines
            )
            body_news = f"<ul style='margin:2px 0 0 18px;padding:0;'>{li}</ul>"
            plain_news = ["BERITA PENGGERAK:"] + [
                f"- [{'Bullish' if h.score > 0 else 'Bearish'} Emas] {h.title} ({h.source})" for h in news.headlines
            ]
        else:
            body_news = ("Tidak ada katalis berita signifikan; pergerakan didominasi dinamika "
                         "volatilitas dan momentum kuantitatif.")
            plain_news = [body_news]
        rows_html += (
            '<tr><td style="padding:6px 20px;font-size:13px;color:#374151;">'
            f"<b>Berita</b><br>{body_news}</td></tr>"
        )
        news_plain += plain_news

    all_warnings = list(bundle.warnings) + list(fc.notes)
    if news is not None:
        all_warnings += [n_ for n_ in news.notes if "radar berita" not in n_]
    all_warnings = _dedupe_notes(all_warnings)
    rows_html += _notes_html(all_warnings)

    news_note = (
        "Berita hanya ditampilkan dan tidak mengubah angka; event berisiko tinggi melebarkan kisaran. "
        if news is not None else ""
    )
    footer = (
        f"{html.escape(spot_label)} &middot; sumber: {html.escape(bundle.source_label)} &middot; "
        f"dibuat {_to_local(bundle.now_utc, tz)} WIB. {news_note}{DISCLAIMER}"
    )
    body = _wrap_html("Perkiraan candle D1 XAUUSD hari ini", rows_html, footer)

    plain = [
        f"AU-TOMATIC - perkiraan candle D1 XAUUSD ({bundle.target_session.date()})",
        "",
        f"Kisaran close (70%) : {_usd(c15, 2)} - {_usd(c85, 2)}",
        f"  relatif open      : {_signed(c15 - o)} s/d {_signed(c85 - o)} (open {_usd(o, 2)})",
        f"Perkiraan High      : {_usd(h50)} (kisaran {_usd(h15)} - {_usd(h85)})",
        f"Perkiraan Low       : {_usd(l50)} (kisaran {_usd(l15)} - {_usd(l85)})",
        f"Median close        : {_usd(c50)} | peluang candle naik {fc.p_up:.0%} ({bias_txt})",
        bias_basis,
        basis_txt,
        "",
    ] + [f"- {c}" for c in context]
    if late_txt:
        plain.insert(1, late_txt)
    if news_plain:
        plain += [""] + news_plain
    if all_warnings:
        plain += [""] + _notes_plain(all_warnings)
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


_HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def build_market_closed_email(label: pd.Timestamp, holiday_name: str, cfg: dict) -> tuple[str, str, str]:
    """Short email for a listed market-closed holiday (instead of an error email).

    Carries no price levels and no pipeline details. The holiday name comes from config.yaml.
    """
    date_txt = f"{_HARI[label.weekday()]}, {label.day} {_BULAN[label.month - 1]} {label.year}"
    subject = f"{cfg['email']['subject_prefix']} Pasar libur hari ini | tidak ada perkiraan"
    safe_name = html.escape(holiday_name)
    rows = (
        '<tr><td style="padding:8px 20px;font-size:14px;">'
        f"Hari ini ({html.escape(date_txt)}) pasar emas libur: <b>{safe_name}</b>. "
        "Tidak ada perkiraan candle D1 hari ini.</td></tr>"
        '<tr><td style="padding:4px 20px 8px 20px;font-size:13px;color:#374151;">'
        "Perkiraan berikutnya dikirim pada hari kerja berikutnya.</td></tr>"
    )
    body = _wrap_html("Pasar libur", rows, DISCLAIMER)
    plain = (
        f"AU-TOMATIC - pasar libur hari ini ({date_txt}): {holiday_name}.\n"
        "Tidak ada perkiraan candle D1 hari ini. Perkiraan berikutnya dikirim pada hari kerja berikutnya.\n\n"
        f"{DISCLAIMER}"
    )
    return subject, body, plain


def _scan_recipients(raw: str | None) -> tuple[list[str], int, int]:
    """Split EMAIL_TO on commas, semicolons or whitespace.

    Returns (usable addresses, number of malformed entries, number of duplicates). Empty pieces
    left by a trailing separator are not counted as malformed."""
    seen: set[str] = set()
    out: list[str] = []
    invalid = duplicates = 0
    for part in re.split(r"[,;\s]+", (raw or "").strip()):
        addr = part.strip().strip("<>")
        if not addr:
            continue
        if "@" not in addr or "." not in addr.split("@")[-1]:
            invalid += 1
        elif addr.lower() in seen:
            duplicates += 1
        else:
            seen.add(addr.lower())
            out.append(addr)
    return out, invalid, duplicates


def parse_recipients(raw: str) -> list[str]:
    """Split EMAIL_TO on commas, semicolons or whitespace; drop blanks, junk and duplicates."""
    return _scan_recipients(raw)[0]


def summarize_recipients(raw: str | None) -> tuple[int, int, int]:
    """(usable, malformed, duplicate) counts for EMAIL_TO. Counts only: the addresses themselves
    must never reach a log because the repository is public."""
    usable, invalid, duplicates = _scan_recipients(raw)
    return len(usable), invalid, duplicates


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
    hide_recipients: bool = True,
    retries: int = 3,
    retry_delay_s: float = 5.0,
    fallback_starttls: bool = True,
    sleep=time.sleep,
) -> tuple[int, int]:
    """Send a multipart/alternative message over SMTP. Never logs addresses.

    Returns (accepted, refused) recipient counts. With `hide_recipients` the recipients
    are only in the SMTP envelope (like Bcc), so they cannot see each other's addresses.

    Robustness (Gmail sometimes drops connections from CI runners mid-login):
    each attempt tries SMTP-SSL on `port`, then STARTTLS on 587 when `fallback_starttls`;
    transient errors are retried `retries` times with growing pauses. A rejected login
    (wrong user/app password) is raised at once, because retrying cannot fix it.
    """
    recipients = parse_recipients(to)
    if not recipients:
        raise ValueError("EMAIL_TO contains no valid email address.")
    user = user.strip()
    password = "".join(password.split())      # app passwords have no spaces; secrets often carry stray ones

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = user if hide_recipients else ", ".join(recipients)
    msg.attach(MIMEText(text_body, "plain", "utf-8"))   # fallback first
    msg.attach(MIMEText(html_body, "html", "utf-8"))    # preferred part last
    payload = msg.as_string()

    context = ssl.create_default_context()
    plans = [("ssl", port)] + ([("starttls", 587)] if fallback_starttls and port != 587 else [])
    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        for mode, prt in plans:
            try:
                if mode == "ssl":
                    server = smtplib.SMTP_SSL(host, prt, context=context, timeout=30)
                else:
                    server = smtplib.SMTP(host, prt, timeout=30)
                with server:
                    if mode == "starttls":
                        server.ehlo()
                        server.starttls(context=context)
                        server.ehlo()
                    server.login(user, password)
                    refused = server.sendmail(user, recipients, payload) or {}
                return len(recipients) - len(refused), len(refused)
            except smtplib.SMTPAuthenticationError:
                raise
            except _RETRYABLE as exc:
                last_exc = exc
                log.warning("SMTP attempt %d/%d via %s:%d failed (%s).",
                            attempt, retries, mode, prt, type(exc).__name__)
        if attempt < retries:
            sleep(retry_delay_s * 3 ** (attempt - 1))
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------
# Gmail API over HTTPS (used when Google drops SMTP logins from CI runners)
# --------------------------------------------------------------------------
TOKEN_URL = "https://oauth2.googleapis.com/token"
SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"


class EmailAuthError(RuntimeError):
    """Credentials/permission problem that retrying cannot fix (message says what to redo)."""


class EmailTransportError(RuntimeError):
    """Network or server trouble that persisted after all retries."""


def gmail_api_credentials(env) -> tuple[str, str, str] | None:
    """(client_id, client_secret, refresh_token) from the environment, or None if any is missing."""
    keys = ("GMAIL_OAUTH_CLIENT_ID", "GMAIL_OAUTH_CLIENT_SECRET", "GMAIL_OAUTH_REFRESH_TOKEN")
    vals = tuple("".join(str(env.get(k, "")).split()) for k in keys)
    return vals if all(vals) else None


def gmail_api_missing(env) -> list[str]:
    """Names of the Gmail OAuth variables that are absent or empty (for a precise log line)."""
    keys = ("GMAIL_OAUTH_CLIENT_ID", "GMAIL_OAUTH_CLIENT_SECRET", "GMAIL_OAUTH_REFRESH_TOKEN")
    return [k for k in keys if not "".join(str(env.get(k, "")).split())]


def _mime(subject: str, html_body: str, text_body: str, to_header: str) -> bytes:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["To"] = to_header            # Gmail fills in From with the authenticated account
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    return msg.as_bytes()


def _google_error_code(resp) -> str:
    """Only the short machine code of a Google error (never tokens, never descriptions)."""
    try:
        err = resp.json().get("error")
        return str(err.get("status") if isinstance(err, dict) else err)[:60]
    except Exception:  # noqa: BLE001
        return "unknown"


def send_email_gmail_api(
    subject: str,
    html_body: str,
    text_body: str,
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    to: str,
    hide_recipients: bool = True,
    retries: int = 3,
    retry_delay_s: float = 5.0,
    sleep=time.sleep,
    post=None,
) -> tuple[int, int]:
    """Send through the Gmail API (HTTPS). Returns (accepted, refused) like `send_email`.

    With `hide_recipients` each recipient gets their own message addressed only to them, so
    nobody sees anyone else's address. Transient failures (network, 429, 5xx) are retried
    with growing pauses; an expired/revoked token or missing permission raises EmailAuthError
    at once. A 400 for one recipient counts as refused and does not stop the others.
    Never logs addresses, tokens or Google's error descriptions.
    """
    import requests

    post = post or requests.post
    recipients = parse_recipients(to)
    if not recipients:
        raise ValueError("EMAIL_TO contains no valid email address.")

    def call(url: str, **kw):
        last: Exception | None = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                resp = post(url, timeout=30, **kw)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = EmailTransportError(f"HTTP {resp.status_code}")
                    log.warning("Gmail API attempt %d/%d: HTTP %d.", attempt, retries, resp.status_code)
                else:
                    return resp
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = exc
                log.warning("Gmail API attempt %d/%d: %s.", attempt, retries, type(exc).__name__)
            if attempt < retries:
                sleep(retry_delay_s * 3 ** (attempt - 1))
        raise EmailTransportError(f"Gmail API unreachable after {retries} attempts "
                                  f"({type(last).__name__}).") from last

    tok = call(TOKEN_URL, data={
        "client_id": client_id, "client_secret": client_secret,
        "refresh_token": refresh_token, "grant_type": "refresh_token",
    })
    if tok.status_code != 200:
        code = _google_error_code(tok)
        raise EmailAuthError(
            f"Google refused the OAuth refresh token ({code}). The token is expired or revoked, or the "
            f"client id/secret is wrong: if the consent screen is still in 'Testing', publish it "
            f"('In production') and run tools/gmail_oauth_setup.py again."
        )
    access = tok.json().get("access_token")
    if not access:
        raise EmailAuthError("Google returned no access token.")
    headers = {"Authorization": f"Bearer {access}", "Content-Type": "application/json"}

    messages = ([(r, r) for r in recipients] if hide_recipients
                else [(", ".join(recipients), None)])       # (To header, single recipient or None)
    accepted = refused = 0
    for to_header, single in messages:
        raw = base64.urlsafe_b64encode(_mime(subject, html_body, text_body, to_header)).decode("ascii")
        resp = call(SEND_URL, headers=headers, json={"raw": raw})
        if resp.status_code == 200:
            accepted += 1 if single else len(recipients)
        elif resp.status_code in (401, 403):
            raise EmailAuthError(
                f"Gmail API denied sending (HTTP {resp.status_code}, {_google_error_code(resp)}). Enable "
                f"the Gmail API in the Google Cloud project and grant the gmail.send permission."
            )
        else:
            log.error("Gmail API rejected one message (HTTP %d, %s).", resp.status_code, _google_error_code(resp))
            refused += 1 if single else len(recipients)
    return accepted, refused
