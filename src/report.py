"""Computation report: makes every step of the forecast visible.

`build_report(bundle, forecast, cfg)` returns (html, text). The forecast must have been
produced with `forecast_day(..., explain=True)`. Charts are drawn with matplotlib and
embedded in the HTML as base64 PNG, so the report is a single self-contained file.
"""

from __future__ import annotations

import base64
import html as _html
import io
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data_fetcher import DataBundle  # noqa: E402
from .quant_engine import Forecast  # noqa: E402

REGIME_COLORS = {"Konsolidasi": "#2f9e44", "Trending": "#1971c2", "Shock": "#e03131"}
GOLD = "#b8860b"


def _usd(x: float, d: int = 2) -> str:
    return f"${x:,.{d}f}"


def _pct(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}%"


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def _chart_trend(fc: Forecast, tz_note: str) -> str:
    a, k = fc.diag["arrays"], fc.diag["kalman"]
    close = a["hist_close"].tail(90)
    lev = pd.Series(a["kalman_level"], index=a["hist_close"].index).tail(90)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.scatter(close.index, close.values, s=9, color="#868e96", label="Close harian", zorder=3)
    ax.plot(lev.index, lev.values, color=GOLD, lw=2, label="Level Kalman (disaring)")
    x_next = close.index[-1] + pd.Timedelta(days=3 if close.index[-1].weekday() == 4 else 1)
    ax.plot([close.index[-1], x_next], [k["level"], k["extrapolated_next"]], color="#e8590c", lw=2, ls="--",
            marker="o", label="Ekstrapolasi tren (level + kecepatan)")
    c15, c50, c85 = fc.close_q
    ax.errorbar([x_next], [c50], yerr=[[c50 - c15], [c85 - c50]], fmt="s", color="#1971c2", capsize=6,
                label="Kisaran close model (P15-P85)")
    ax.set_title("Pilar 1 - Tren Kalman dan ekstrapolasinya")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _png_b64(fig)


def _chart_regimes(fc: Forecast) -> str:
    a, rg = fc.diag["arrays"], fc.diag["regime"]
    gamma, dates = a["gamma"], a["gamma_dates"]
    n = min(150, len(dates))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    close = a["hist_close"].reindex(dates).tail(n)
    ax1.plot(close.index, close.values, color="#495057", lw=1.2)
    ax1.set_title("Pilar 2 - Rezim pasar tersembunyi (HMM)")
    ax1.set_ylabel("Close")
    ax1.grid(alpha=0.25)
    order = [rg["names"].index(nm) for nm in ("Konsolidasi", "Trending", "Shock")]
    ax2.stackplot(
        dates[-n:], *[gamma[-n:, i] for i in order],
        labels=["Konsolidasi", "Trending", "Shock"],
        colors=[REGIME_COLORS[nm] for nm in ("Konsolidasi", "Trending", "Shock")], alpha=0.85,
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probabilitas rezim")
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    fig.autofmt_xdate()
    return _png_b64(fig)


def _segment_times(fc: Forecast) -> list[pd.Timestamp]:
    now = pd.Timestamp(fc.diag["intraday"]["now_eff"])
    end = pd.Timestamp(fc.diag["intraday"]["end"])
    cuts = [now]
    t = now.floor("h") + pd.Timedelta(hours=1)
    while t < end:
        cuts.append(t)
        t += pd.Timedelta(hours=1)
    cuts.append(end)
    return cuts


def _chart_paths(fc: Forecast, display_tz: str) -> str:
    a = fc.diag["arrays"]
    paths = a["paths"]
    times = [t.tz_convert(ZoneInfo(display_tz)) for t in _segment_times(fc)]
    x = np.arange(len(times))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rng = np.random.default_rng(1)
    for i in rng.choice(len(paths), size=min(120, len(paths)), replace=False):
        ax.plot(x, paths[i], color="#adb5bd", lw=0.4, alpha=0.5)
    q15, q50, q85 = (np.quantile(paths, q, axis=0) for q in (0.15, 0.50, 0.85))
    ax.fill_between(x, q15, q85, color="#74c0fc", alpha=0.45, label="Pita P15-P85")
    ax.plot(x, q50, color="#1864ab", lw=2, label="Median")
    ax.axhline(fc.open_price, color="#e8590c", ls="--", lw=1, label="Open / harga acuan")
    if fc.realized_high is not None:
        ax.axhline(fc.realized_high, color="#2f9e44", ls=":", lw=1, label="High sejauh ini")
        ax.axhline(fc.realized_low, color="#e03131", ls=":", lw=1, label="Low sejauh ini")
    step = max(1, len(times) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([t.strftime("%a %H:%M") for t in times[::step]], rotation=30, fontsize=8)
    ax.set_title(f"Pilar 3 - Monte Carlo {len(paths):,} jalur (jam {display_tz})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    return _png_b64(fig)


def _chart_dist(fc: Forecast) -> str:
    a = fc.diag["arrays"]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    for ax, key, title, q in (
        (axes[0], "close", "Close", fc.close_q),
        (axes[1], "high", "High", fc.high_q),
        (axes[2], "low", "Low", fc.low_q),
    ):
        ax.hist(a[key], bins=60, color="#91a7ff", edgecolor="none")
        for v, ls in zip(q, ("--", "-", "--")):
            ax.axvline(v, color="#c92a2a", ls=ls, lw=1.2)
        ax.axvline(fc.open_price, color="#e8590c", ls=":", lw=1)
        ax.set_title(f"{title}: {q[0]:,.0f} / {q[1]:,.0f} / {q[2]:,.0f}", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Sebaran hasil simulasi (garis merah = P15, P50, P85; oranye = open)", fontsize=9)
    fig.tight_layout()
    return _png_b64(fig)


def _chart_profile(fc: Forecast) -> str:
    prof = np.array(fc.diag["intraday"]["profile"])
    fig, ax = plt.subplots(figsize=(9, 2.8))
    hours = np.arange(24)
    colors = ["#e8590c" if v >= np.sort(prof)[-3] else "#adb5bd" for v in prof]
    ax.bar(hours, 100 * prof, color=colors)
    ax.axhline(100 / 24, color="#495057", ls="--", lw=1, label="Rata-rata (merata)")
    ax.set_xticks(hours)
    ax.set_xlabel("Jam New York (candle D1 dibuka jam 17)")
    ax.set_ylabel("% varians harian")
    ax.set_title("Profil volatilitas intraday (dari 2 tahun data per jam)")
    ax.legend(fontsize=8)
    return _png_b64(fig)


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------
def build_report(bundle: DataBundle, fc: Forecast, cfg: dict) -> tuple[str, str]:
    if "arrays" not in fc.diag:
        raise ValueError("Call forecast_day(..., explain=True) to build a report.")
    d, tz = fc.diag, cfg["email"]["display_tz"]
    k, rg, vol, dr, jp, it = d["kalman"], d["regime"], d["vol"], d["drift"], d["jumps"], d["intraday"]
    p0 = d["p0"]
    a = d["arrays"]

    sections: list[dict] = []

    # ---- 1. summary
    c15, c50, c85 = fc.close_q
    sections.append({
        "title": "1. Hasil akhir",
        "paras": [
            f"Target candle D1: {bundle.target_session.date()} (batas 17:00 New York). "
            f"Sumber data: {bundle.source_label}. Harga acuan {_usd(p0)}; "
            f"open {'tercatat ' + _usd(fc.open_price) if fc.open_known else 'belum ada (akhir pekan), memakai harga acuan'}."
        ],
        "table": (
            ["", "P15", "P50 (median)", "P85"],
            [["Close", _usd(c15), _usd(c50), _usd(c85)],
             ["High", _usd(fc.high_q[0]), _usd(fc.high_q[1]), _usd(fc.high_q[2])],
             ["Low", _usd(fc.low_q[0]), _usd(fc.low_q[1]), _usd(fc.low_q[2])]],
        ),
        "after": [f"Peluang candle ditutup di atas open: {fc.p_up:.1%} ({fc.bias}). "
                  "Kisaran P15 sampai P85 adalah interval prediksi 70%."],
    })

    # ---- 2. Kalman
    dollars_drift = fc.mu_d * p0
    sections.append({
        "title": "2. Pilar 1 - Kalman Filter: ekstrapolasi tren",
        "paras": [
            "Model <i>local linear trend</i>: level(t) = level(t-1) + kecepatan(t-1) + noise, "
            "kecepatan(t) = kecepatan(t-1) + noise. Rasio noise dipilih dengan maximum likelihood "
            f"(grid), hanya memakai estimasi <i>filtered</i> sehingga tidak ada kebocoran data masa depan.",
        ],
        "table": (
            ["Besaran", "Nilai"],
            [["Close terakhir", _usd(k["last_close"])],
             ["Level hasil filter", _usd(k["level"])],
             ["Kecepatan tren", f"{k['velocity']:+.2f} $/hari (galat baku {k['se']:.2f})"],
             ["z = kecepatan / galat baku", f"{k['z']:+.2f}"],
             ["Ekstrapolasi 1 hari (level + kecepatan)", _usd(k["extrapolated_next"])],
             ["Median Monte Carlo (hasil model)", _usd(c50)]],
        ),
        "after": [
            f"Arti: ekstrapolasi murni menebak {_usd(k['extrapolated_next'])}, sedangkan model menebak median "
            f"{_usd(c50)}. Selisihnya karena kecepatan tren dikecilkan (shrinkage) dan dibatasi. "
            f"Kontribusi drift akhir terhadap harga hanya sekitar {dollars_drift:+.2f} dolar. "
            + ("Karena |z| < 2, kecepatan ini tidak dapat dibedakan dari noise." if abs(k["z"]) < 2
               else "|z| >= 2: tren cukup jelas secara statistik."),
        ],
        "fig": ("trend", _chart_trend(fc, tz)),
    })

    # ---- 3. HMM
    rows = []
    for i, nm in enumerate(rg["names"]):
        rows.append([nm, f"{rg['probs_today'][i]:.1%}", f"{rg['probs_next'][i]:.1%}",
                     _pct(rg["mu"][i], 3), _pct(rg["sigma"][i], 3)])
    tm = np.array(rg["transmat"])
    order = [rg["names"].index(nm) for nm in ("Konsolidasi", "Trending", "Shock")]
    tm_rows = [[nm] + [f"{tm[i][j]:.1%}" for j in order] for nm, i in zip(("Konsolidasi", "Trending", "Shock"), order)]
    sections.append({
        "title": "3. Pilar 2 - Hidden Markov Model: rezim pasar",
        "paras": [
            f"3 rezim tersembunyi dilatih dengan EM pada {rg['n_obs']} hari, fitur: {', '.join(rg['features'])}. "
            "Rezim dinamai dari peringkat volatilitasnya (rendah = Konsolidasi, menengah = Trending, "
            "tinggi = Shock). Probabilitas rezim candle ini = probabilitas hari ini dikali matriks transisi."
        ],
        "table": (["Rezim", "Peluang hari ini", "Peluang candle ini", "Rata-rata return harian", "Volatilitas harian"], rows),
        "table2": (["Dari \\ ke", "Konsolidasi", "Trending", "Shock"], tm_rows),
        "after": [f"Rezim dominan: {fc.regime}. Log-likelihood model: {rg['loglik']:.1f}."],
        "fig": ("regimes", _chart_regimes(fc)),
    })

    # ---- 4. vol + drift
    sections.append({
        "title": "4. Volatilitas dan drift",
        "paras": [
            "Volatilitas harian = campuran volatilitas rezim dan EWMA (RiskMetrics), dibatasi pada pita "
            "kewajaran terhadap volatilitas 60 hari. Drift = (shrinkage x rata-rata rezim) + "
            "(shrinkage x kecepatan Kalman), dipotong pada +/-0,2 sigma."
        ],
        "table": (
            ["Besaran", "Nilai"],
            [["Volatilitas rezim (campuran)", _pct(vol["sigma_regime"], 3)],
             ["Volatilitas EWMA", _pct(vol["sigma_ewma"], 3)],
             [f"Campuran (bobot rezim {vol['w_regime']:.0%})", _pct(vol["sigma_raw"], 3)],
             ["Volatilitas harian dipakai (sigma_D)", f"{_pct(vol['sigma_d'], 3)} = sekitar {_usd(vol['sigma_d'] * p0, 0)}"],
             ["Drift rezim", _pct(dr["mu_regime"], 4)],
             ["Drift Kalman (kecepatan / harga)", _pct(dr["mu_kalman"], 4)],
             ["Drift gabungan sebelum batas", _pct(dr["mu_raw"], 4)],
             ["Batas +/-0,2 sigma", _pct(dr["cap"], 4)],
             ["Drift dipakai (mu_D)", f"{_pct(dr['mu_d'], 4)} = {dr['mu_d'] * p0:+.2f} dolar"]],
        ),
    })

    # ---- 5. jumps + intraday
    prof = np.array(it["profile"])
    top = np.argsort(prof)[-3:][::-1]
    end_ny = pd.Timestamp(it["end"]).tz_convert("America/New_York")
    def ny_hour_to_wib(h: int) -> str:
        t = pd.Timestamp(year=end_ny.year, month=end_ny.month, day=end_ny.day, hour=h, tz="America/New_York")
        return t.tz_convert(ZoneInfo(tz)).strftime("%H:%M")
    busiest = ", ".join(f"{h}:00 NY (= {ny_hour_to_wib(int(h))} WIB, {prof[h]:.1%})" for h in top)
    sections.append({
        "title": "5. Lompatan harga dan pola intraday",
        "paras": [
            "Lompatan (Merton): hari dengan |return| > 2,5 kali sigma robust (MAD) dianggap lompatan; "
            "frekuensi dan ukurannya dikalibrasi dari riwayat. Pada hari akhir pekan atau event, frekuensi dinaikkan."
        ],
        "table": (
            ["Besaran", "Nilai"],
            [["Frekuensi lompatan dasar (per hari)", f"{jp['lam0']:.3f}"],
             ["Ukuran lompatan (std)", _pct(jp["sig_j"], 3)],
             ["Pengali (akhir pekan/event)", f"x{jp['mult']:.2f}"],
             ["Frekuensi dipakai", f"{jp['lam']:.3f}"],
             ["Volatilitas difusi (di luar lompatan)", _pct(jp["sigma_diff"], 3)],
             ["Porsi varians hari yang tersisa", f"{it['remaining']:.1%}"],
             ["Jumlah segmen jam yang disimulasikan", str(it["n_segments"])]],
        ),
        "after": [f"Jam paling bergejolak: {busiest}."],
        "fig": ("profile", _chart_profile(fc)),
    })

    # ---- 6. Monte Carlo
    def stats(key: str) -> list[str]:
        v = a[key]
        return [f"{key.capitalize()}", _usd(v.mean()), _usd(v.std()), _usd(np.quantile(v, 0.15)),
                _usd(np.quantile(v, 0.5)), _usd(np.quantile(v, 0.85))]
    sections.append({
        "title": "6. Pilar 3 - Simulasi Monte Carlo jump-diffusion",
        "paras": [
            f"{fc.n_paths:,} jalur harga disimulasikan per jam dari harga acuan sampai penutupan candle. "
            "Tiap langkah: difusi normal + lompatan Poisson. High dan Low tiap jalur dihitung dengan koreksi "
            "Brownian bridge (agar tidak meremehkan puncak/lembah di antara jam), lalu digabung dengan "
            "High/Low yang sudah terjadi. Seed simulasi dari tanggal sehingga hasilnya bisa direproduksi."
        ],
        "table": (["", "Rata-rata", "Simpangan baku", "P15", "P50", "P85"], [stats("close"), stats("high"), stats("low")]),
        "fig": ("paths", _chart_paths(fc, tz)),
        "fig2": ("dist", _chart_dist(fc)),
    })

    # ---- 7. trust
    sections.append({
        "title": "7. Seberapa bisa dipercaya",
        "paras": [
            "Uji walk-forward pada 120 sesi emas asli (21 Nov 2026, tools/backtest.py; angka bisa berubah saat diuji ulang): kisaran 70% mencakup 63% (Close), 68% (High), "
            "66% (Low) hari sesungguhnya; kemampuan menebak arah tidak ada (Brier 0,249 vs 0,250); "
            "akurasi kisaran Close setara random walk dengan volatilitas EWMA. Nilai tambah model ada di High/Low dan di "
            "kisaran yang terkalibrasi, bukan di tebakan arah.",
            "Untuk menguji ekstrapolasi tren murni: python tools/backtest.py --set model.drift.k_kalman=1 "
            "--set model.drift.cap_sigma_mult=10 lalu bandingkan dengan hasil bawaan.",
        ],
    })

    # ---------------- render HTML ----------------
    def tbl_html(t):
        head = "".join(f"<th>{_html.escape(h)}</th>" for h in t[0])
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in t[1])
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    parts = []
    for sct in sections:
        parts.append(f"<h2>{_html.escape(sct['title'])}</h2>")
        parts += [f"<p>{p}</p>" for p in sct.get("paras", [])]
        for tk in ("table", "table2"):
            if tk in sct:
                parts.append(tbl_html(sct[tk]))
        parts += [f"<p>{_html.escape(p)}</p>" for p in sct.get("after", [])]
        for fk in ("fig", "fig2"):
            if fk in sct:
                parts.append(f'<img alt="{sct[fk][0]}" src="data:image/png;base64,{sct[fk][1]}">')
    css = (
        "body{font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;line-height:1.5;"
        "color:#111827;background:#ffffff}h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #e5e7eb;"
        "padding-bottom:4px}table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}"
        "th,td{border:1px solid #e5e7eb;padding:5px 8px;text-align:left}th{background:#f3f4f6}img{max-width:100%;margin:8px 0}"
        "p{font-size:14px}.note{color:#6b7280;font-size:12px}"
    )
    html_doc = (
        f"<!DOCTYPE html><html lang='id'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Au-tomatic - laporan komputasi</title><style>{css}</style></head><body>"
        f"<h1>Au-tomatic: laporan perhitungan komputasi</h1>"
        f"<p class='note'>Dibuat {bundle.now_utc.tz_convert(ZoneInfo(tz)).strftime('%d %b %Y %H:%M')} WIB. "
        "Estimasi statistik, bukan nasihat finansial.</p>" + "".join(parts) + "</body></html>"
    )

    # ---------------- render text ----------------
    def strip(s: str) -> str:
        return _html.unescape(s.replace("<i>", "").replace("</i>", ""))

    lines = ["AU-TOMATIC - LAPORAN PERHITUNGAN KOMPUTASI", ""]
    for sct in sections:
        lines += [sct["title"], "-" * len(sct["title"])]
        lines += [strip(p) for p in sct.get("paras", [])]
        for tk in ("table", "table2"):
            if tk in sct:
                head, body = sct[tk]
                lines.append(" | ".join(head))
                lines += [" | ".join(str(c) for c in r) for r in body]
        lines += sct.get("after", [])
        lines.append("")
    return html_doc, "\n".join(lines)
