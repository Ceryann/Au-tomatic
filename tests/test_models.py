"""Offline tests for the models (Sprint 2). Run: pytest -q   or   python tests/test_models.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_data_layer as T  # noqa: E402  (reuses synthetic data + fakes)
from src.data_fetcher import get_gold_bundle, session_end_utc  # noqa: E402
from src import email_dispatcher as em_mod  # noqa: E402
from src.email_dispatcher import build_forecast_email  # noqa: E402
from src.notes import INFO, WARN, Note, note_level, note_title  # noqa: E402
from src.hmm import GaussianHMM  # noqa: E402
from src.quant_engine import (  # noqa: E402
    calibrate_jumps,
    forecast_day,
    intraday_profile,
    segment_weights,
    simulate_day,
)
from src.signal_and_regime import build_features, kalman_trend, map_regimes  # noqa: E402


def fast_cfg() -> dict:
    cfg = T.load_cfg()
    cfg["model"]["hmm"]["n_inits"] = 2
    cfg["model"]["hmm"]["n_iter"] = 60
    return cfg


def bundle_at(now: pd.Timestamp, hourly: pd.DataFrame, cfg: dict):
    with T.FakeYahoo(lambda s, i, d: hourly[hourly.index <= now]):
        return get_gold_bundle(cfg, now_utc=now)


# ------------------------------------------------------------------ Kalman
def test_kalman_recovers_known_trend():
    rng = np.random.default_rng(1)
    y = 1000 + 4.0 * np.arange(400) + rng.normal(0, 3, 400)
    k = kalman_trend(pd.Series(y))
    assert abs(k.velocity - 4.0) < 0.5 and k.label == "naik"


def test_kalman_rarely_invents_trends_in_random_walks():
    rng = np.random.default_rng(5)
    false = sum(
        kalman_trend(pd.Series(3000 + np.cumsum(rng.normal(0, 30, 300)))).label != "datar"
        for _ in range(30)
    )
    assert false <= 4          # ~5% expected; allow noise


# --------------------------------------------------------------------- HMM
def _world(seed: int, T_: int = 600):
    rng = np.random.default_rng(seed)
    P = np.array([[0.96, 0.03, 0.01], [0.04, 0.94, 0.02], [0.10, 0.10, 0.80]])
    true = np.zeros(T_, int)
    for i in range(1, T_):
        true[i] = rng.choice(3, p=P[true[i - 1]])
    mu, sg = np.array([0.0, 0.0012, -0.0005]), np.array([0.005, 0.008, 0.020])
    ret = rng.normal(mu[true], sg[true])
    return true, pd.Series(ret, index=pd.bdate_range("2023-01-02", periods=T_))


def test_hmm_recovers_regimes_better_than_chance_and_names_by_volatility():
    code = {"Konsolidasi": 0, "Trending": 1, "Shock": 2}
    accs = []
    for seed in range(6):
        true, s = _world(100 + seed)
        feats, _ = build_features(s, None)
        X = feats.values
        Z = (X - X.mean(0)) / X.std(0)
        h = GaussianHMM(3, n_inits=3, n_iter=80, seed=7).fit(Z)
        names = map_regimes(h.means_)
        pred = np.array([code[names[k]] for k in np.argmax(h.gamma_, axis=1)])
        accs.append((pred == true[-len(pred):]).mean())
        # naming follows volatility rank
        order = np.argsort(h.means_[:, 1])
        assert names[order[0]] == "Konsolidasi" and names[order[2]] == "Shock"
    assert np.mean(accs) > 0.55, accs          # chance ~0.33-0.5
    # probabilities are valid
    assert np.allclose(h.gamma_.sum(axis=1), 1.0) and np.allclose(h.transmat_.sum(axis=1), 1.0)


def test_hmm_loglik_improves_over_single_state_baseline():
    _, s = _world(7)
    feats, _ = build_features(s, None)
    X = feats.values
    Z = (X - X.mean(0)) / X.std(0)
    h3 = GaussianHMM(3, n_inits=2, n_iter=80, seed=1).fit(Z)
    ll_single = float(np.sum(-0.5 * (np.log(2 * np.pi * Z.var(0)) + (Z - Z.mean(0)) ** 2 / Z.var(0))))
    assert h3.loglik_ > ll_single


# -------------------------------------------------------- simulation maths
def test_simulation_paths_are_internally_consistent():
    rng = np.random.default_rng(3)
    w = np.full(23, 1 / 23)
    sim = simulate_day(4000.0, w, 0.006, 0.0, 0.05, 0.0, 0.015, 20000, rng)
    assert (sim["high"] >= sim["close"] - 1e-9).all()
    assert (sim["low"] <= sim["close"] + 1e-9).all()
    assert (sim["high"] >= 4000.0 - 1e-9).all() and (sim["low"] <= 4000.0 + 1e-9).all()
    # terminal log-return std ~ sigma_diff plus jump variance
    lr = np.log(sim["close"] / 4000.0)
    expected = np.sqrt(0.006 ** 2 + 0.05 * 0.015 ** 2)
    assert abs(lr.std() - expected) / expected < 0.05


def test_zero_vol_is_deterministic_and_more_vol_widens_the_band():
    w = np.full(10, 0.1)
    flat = simulate_day(100.0, w, 0.0, 0.0, 0.0, 0.0, 0.0, 500, np.random.default_rng(1))
    assert np.allclose(flat["close"], 100.0) and np.allclose(flat["high"], 100.0)

    def band(sig):
        s = simulate_day(100.0, w, sig, 0.0, 0.0, 0.0, 0.0, 20000, np.random.default_rng(2))
        q = np.quantile(s["close"], [0.15, 0.85])
        return q[1] - q[0]

    assert band(0.01) > band(0.005) * 1.8


def test_simulation_expected_extremes_match_theory():
    # Brownian motion: E[max] = sigma*sqrt(2/pi) (T=1). Bridge correction should hit it closely.
    w = np.full(24, 1 / 24)
    sim = simulate_day(1.0, w, 0.01, 0.0, 0.0, 0.0, 0.0, 60000, np.random.default_rng(9))
    emax = np.log(sim["high"]).mean()
    assert abs(emax - 0.01 * np.sqrt(2 / np.pi)) / (0.01 * np.sqrt(2 / np.pi)) < 0.04


def test_segment_weights_cover_remaining_session():
    prof = np.full(24, 1 / 24)
    tz = "America/New_York"
    end = pd.Timestamp("2026-01-14 22:00", tz="UTC")
    full = segment_weights(pd.Timestamp("2026-01-13 22:00", tz="UTC"), end, prof, tz)
    assert abs(full.sum() - 1.0) < 1e-9 and len(full) == 24
    half = segment_weights(pd.Timestamp("2026-01-14 10:00", tz="UTC"), end, prof, tz)
    assert abs(half.sum() - 12 / 24) < 1e-9
    part = segment_weights(pd.Timestamp("2026-01-13 22:30", tz="UTC"), end, prof, tz)
    assert abs(part.sum() - 23.5 / 24) < 1e-9


def test_intraday_profile_sums_to_one_and_detects_busy_hours():
    hourly = T.make_hourly(T.FIRST_SUNDAY, 60, seed=2)
    # boost volatility during 13:00 NY bars
    hours = hourly.index.tz_convert("America/New_York").hour
    boost = np.where(hours == 13, 3.0, 1.0)
    lr = np.log(hourly["Close"]).diff().fillna(0) * boost
    hourly["Close"] = 2000 * np.exp(lr.cumsum())
    prof = intraday_profile(hourly, "America/New_York")
    assert abs(prof.sum() - 1) < 1e-9 and prof.argmax() == 13


def test_jump_calibration_finds_planted_jumps():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.005, 500)
    idx = rng.choice(500, 30, replace=False)
    r[idx] += rng.choice([-1, 1], 30) * 0.03
    lam, mu_j, sig_j, rs = calibrate_jumps(r, {"mad_threshold": 2.5, "lambda_min": 0.02, "lambda_max": 0.15})
    assert 0.04 < lam <= 0.15 and sig_j > 0.015 and abs(rs - 0.005) < 0.0015


# ------------------------------------------------------------ full forecast
def test_forecast_is_consistent_and_reproducible():
    cfg = fast_cfg()
    hourly = T.hourly_until(T.NOW, seed=11)
    b = bundle_at(T.NOW, hourly, cfg)
    f1 = forecast_day(b, None, cfg)
    f2 = forecast_day(b, None, cfg)
    for q in (f1.close_q, f1.high_q, f1.low_q):
        assert q[0] < q[1] < q[2]
    assert f1.low_q[1] < f1.close_q[1] < f1.high_q[1]
    assert f1.high_q[0] >= f1.open_price and f1.low_q[2] <= f1.open_price
    assert 0.0 < f1.p_up < 1.0 and f1.bias in {"Bullish", "Netral", "Bearish"}
    assert f1.close_q == f2.close_q and f1.p_up == f2.p_up      # seeded by date
    assert abs(f1.mu_d) <= cfg["model"]["drift"]["cap_sigma_mult"] * f1.sigma_d + 1e-12


def test_event_day_widens_the_range():
    cfg = fast_cfg()
    b = bundle_at(T.NOW, T.hourly_until(T.NOW, seed=11), cfg)
    base = forecast_day(b, None, cfg)
    event = forecast_day(b, None, cfg, event_today=True)
    assert (event.close_q[2] - event.close_q[0]) > (base.close_q[2] - base.close_q[0])
    assert event.diag["event"]["n"] == 1 and base.diag["event"]["n"] == 0


def test_sunday_gap_run_is_flagged_and_forecasts_monday():
    cfg = fast_cfg()
    sunday = T.utc("2026-01-11 22:30")
    hourly = T.hourly_until(sunday, seed=11)
    hourly = hourly[hourly.index < T.utc("2026-01-11 22:00")]          # no new-week bar yet
    b = bundle_at(sunday, hourly, cfg)
    fc = forecast_day(b, None, cfg)
    assert b.target_session == pd.Timestamp("2026-01-12") and not fc.open_known
    assert any("akhir pekan" in n for n in fc.notes)


def test_weekend_runs_forecast_monday_over_one_full_session():
    """Regression: a Saturday run used to simulate ~63 closed-market hours (variance x2.6)."""
    cfg = fast_cfg()
    widths = {}
    for label, now in (
        ("friday evening", T.utc("2026-01-09 22:30")),
        ("saturday", T.utc("2026-01-10 12:00")),
        ("sunday morning", T.utc("2026-01-11 10:00")),
    ):
        hourly = T.hourly_until(now, seed=11)                    # data ends at Friday's close
        b = bundle_at(now, hourly, cfg)
        assert b.quality == "nyclose_1h", label                   # hourly source, no silent downgrade
        assert b.target_session == pd.Timestamp("2026-01-12"), label
        fc = forecast_day(b, None, cfg)
        assert abs(fc.remaining_var_frac - 1.0) < 1e-9 and not fc.open_known, label
        widths[label] = fc.close_q[2] - fc.close_q[0]
    # same information, same horizon -> same width (not inflated by closed hours)
    assert max(widths.values()) / min(widths.values()) < 1.02, widths
    # and comparable to a normal weekday run
    normal = forecast_day(bundle_at(T.NOW, T.hourly_until(T.NOW, seed=11), cfg), None, cfg)
    ratio = widths["saturday"] / (normal.close_q[2] - normal.close_q[0])
    assert 0.6 < ratio < 1.8, ratio


def test_calibration_of_the_70_percent_band_on_synthetic_data():
    """Walk forward through ~50 sessions of a world with KNOWN constant volatility.

    The model never sees the truth; if the P15-P85 band is well calibrated, about 70% of
    realised Close / High / Low values should fall inside their bands.
    """
    cfg = fast_cfg()
    full = T.make_hourly(T.FIRST_SUNDAY, 111, seed=21, start_price=3000.0)
    tz, hour = T.TZ, T.HOUR
    from src.data_fetcher import build_ny_close_candles

    truth = build_ny_close_candles(full, tz, hour, T.utc("2026-02-01 00:00"))
    labels = truth.index[-70:-5]
    hit = {"close": 0, "high": 0, "low": 0}
    n = 0
    for label in labels[::1]:
        start = session_end_utc(label - pd.Timedelta(days=1), tz, hour)
        now = start + pd.Timedelta(minutes=30)
        b = bundle_at(now, full, cfg)
        fc = forecast_day(b, None, cfg)
        row = truth.loc[label]
        hit["close"] += fc.close_q[0] <= row["Close"] <= fc.close_q[2]
        hit["high"] += fc.high_q[0] <= row["High"] <= fc.high_q[2]
        hit["low"] += fc.low_q[0] <= row["Low"] <= fc.low_q[2]
        n += 1
    cov = {k: v / n for k, v in hit.items()}
    print("coverage", cov, "n =", n)
    for k, v in cov.items():
        assert 0.50 <= v <= 0.90, (k, cov)


def test_backtest_module_runs_and_reports_sane_numbers():
    from src.evaluation import run_backtest

    cfg = fast_cfg()
    full = T.make_hourly(T.FIRST_SUNDAY, 111, seed=51, start_price=3000.0)
    rep = run_backtest(full, cfg, n_sessions=15)
    assert rep.n >= 12
    for below, inside, above in rep.cover.values():
        assert abs(below + inside + above - 1.0) < 1e-9
    assert rep.pinball_baseline > 0 and rep.pinball_model > 0
    assert 0.0 <= rep.brier <= 1.0
    assert any("Coverage" in line for line in rep.lines())


def test_computation_report_shows_every_step_and_charts():
    from src.report import build_report

    cfg = fast_cfg()
    b = bundle_at(T.NOW, T.hourly_until(T.NOW, seed=11), cfg)
    fc = forecast_day(b, None, cfg, explain=True)
    html_doc, text = build_report(b, fc, cfg)
    for title in ("Kalman", "Hidden Markov", "Monte Carlo", "Volatilitas dan drift", "Lompatan"):
        assert title in html_doc and title in text, title
    assert html_doc.count("data:image/png;base64,") == 5          # trend, regimes, profile, paths, dist
    assert "Ekstrapolasi 1 hari" in text
    assert " nan" not in text.lower() and "None" not in text
    # consistency: report numbers equal the forecast numbers
    assert f"{fc.close_q[0]:,.2f}" in text and f"{fc.diag['kalman']['velocity']:+.2f}" in text
    # without explain=True a report cannot be built
    try:
        build_report(b, forecast_day(b, None, cfg), cfg)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_explain_mode_does_not_change_the_numbers():
    cfg = fast_cfg()
    b = bundle_at(T.NOW, T.hourly_until(T.NOW, seed=11), cfg)
    plain = forecast_day(b, None, cfg)
    explained = forecast_day(b, None, cfg, explain=True)
    assert plain.close_q == explained.close_q and plain.high_q == explained.high_q and plain.p_up == explained.p_up


def test_report_failure_never_replaces_the_forecast_email():
    """Regression: a missing matplotlib used to turn a good forecast into an error email."""
    import tempfile

    import main as main_mod

    cfg_hourly = T.hourly_until(T.NOW, seed=11)
    saved = sys.modules.get("src.report")
    sys.modules["src.report"] = None                      # importing it now raises ImportError
    try:
        with tempfile.TemporaryDirectory() as tmp, T.FakeYahoo(lambda s, i, d: cfg_hourly):
            rc = main_mod.main(["--dry-run", "--explain", "--no-macro", "--out", tmp,
                                "--now", "2026-01-13T22:30:00Z"])
            preview = (Path(tmp) / "preview.txt").read_text(encoding="utf-8")
    finally:
        if saved is None:
            sys.modules.pop("src.report", None)
        else:
            sys.modules["src.report"] = saved
    assert rc == 0
    assert "TIDAK TERSEDIA" not in preview and "Kisaran close" in preview


def test_ci_logs_contain_no_price_levels():
    """Actions logs of a public repo are public: they must not print price levels."""
    import logging
    import re
    import tempfile

    import main as main_mod

    messages: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler, root = Capture(), logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    prev = os.environ.get("GITHUB_ACTIONS")
    os.environ["GITHUB_ACTIONS"] = "true"
    hourly = T.hourly_until(T.NOW, seed=11)
    try:
        with tempfile.TemporaryDirectory() as tmp, T.FakeYahoo(lambda s, i, d: hourly):
            rc = main_mod.main(["--dry-run", "--no-macro", "--out", tmp, "--now", "2026-01-13T22:30:00Z"])
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        if prev is None:
            os.environ.pop("GITHUB_ACTIONS", None)
        else:
            os.environ["GITHUB_ACTIONS"] = prev
    joined = "\n".join(messages)
    assert rc == 0 and "Forecast:" in joined
    assert "Subject" not in joined
    assert not re.search(r"\b\d{4}\.\d{1,2}\b", joined), joined      # nothing like 3458.16


# -------------------------------------------------------------------- email
def test_forecast_email_renders_cleanly():
    cfg = fast_cfg()
    b = bundle_at(T.NOW, T.hourly_until(T.NOW, seed=11), cfg)
    fc = forecast_day(b, None, cfg)
    subject, html_body, text = build_forecast_email(b, fc, cfg)
    assert "Close $" in subject and "| H ~$" in subject and "| L ~$" in subject
    assert (fc.bias in subject) == (fc.bias != "Netral")          # bias word only when informative
    assert "Netral" not in subject
    assert fc.avg_range_5 > 0 and fc.avg_range_20 > 0
    for needle in ("Perkiraan High", "Perkiraan Low", "dari open", "bukan nasihat finansial", "Range High"):
        assert needle in html_body, needle
    for blob in (subject, html_body, text):
        assert "nan" not in blob.lower().replace("finansial", "")
        assert "None" not in blob


def test_futures_email_is_labelled():
    cfg = fast_cfg()
    hourly = T.hourly_until(T.NOW, seed=11)
    b = bundle_at(T.NOW, hourly, cfg)            # no key -> futures fallback
    assert b.is_futures
    fc = forecast_day(b, None, cfg)
    subject, html_body, text = build_forecast_email(b, fc, cfg)
    assert "Futures" in subject and "bukan spot" in html_body


def _email_at(hours_after_now: float = 0.0, seed: int = 11):
    """Synthetic run `hours_after_now` after the usual 22:30 UTC cron time (no network)."""
    cfg = fast_cfg()
    now = T.NOW + pd.Timedelta(hours=hours_after_now)
    b = bundle_at(now, T.hourly_until(now, seed=seed), cfg)
    fc = forecast_day(b, None, cfg)
    return b, fc, cfg, build_forecast_email(b, fc, cfg)


def test_run_timing_matches_the_session_boundaries():
    import dataclasses
    b, fc, cfg, _ = _email_at(0.0)
    t = em_mod.run_timing(b, fc, cfg)                    # NOW = 22:30 UTC = 17:30 NY, 30 min after the open
    assert abs(t["elapsed_h"] - 0.5) < 1e-9 and abs(t["left_h"] - 23.5) < 1e-9 and not t["late"], t
    assert abs(t["rem_frac"] - fc.remaining_var_frac) < 1e-12
    for hours, late in ((1.0, False), (1.49, False), (1.5, True), (7.0, True)):   # default threshold: 2 h after the open
        t = em_mod.run_timing(dataclasses.replace(b, now_utc=b.now_utc + pd.Timedelta(hours=hours)), fc, cfg)
        assert t["late"] is late, (hours, t)
        assert abs(t["elapsed_h"] - (0.5 + hours)) < 1e-9
    cfg2 = dict(cfg, email=dict(cfg["email"], late_run_hours=10))
    t = em_mod.run_timing(dataclasses.replace(b, now_utc=b.now_utc + pd.Timedelta(hours=7)), fc, cfg2)
    assert t["late"] is False, "late_run_hours from config.yaml must be honoured"
    assert em_mod.run_timing(b, fc, {"email": {}}) is None, "missing config keys must degrade to None, not raise"


def test_monday_candle_run_on_sunday_evening_counts_from_the_sunday_open():
    cfg = fast_cfg()
    sunday = T.utc("2026-01-11 22:30")
    hourly = T.hourly_until(sunday, seed=11)
    hourly = hourly[hourly.index < T.utc("2026-01-11 22:00")]
    b = bundle_at(sunday, hourly, cfg)
    fc = forecast_day(b, None, cfg)
    t = em_mod.run_timing(b, fc, cfg)
    assert abs(t["elapsed_h"] - 0.5) < 1e-9 and not t["late"], t        # Sunday 17:00 NY is the open, not Friday


def test_email_shows_the_late_run_banner_only_when_the_run_is_late():
    b0, fc0, cfg0, (s0, h0, t0) = _email_at(0.0)
    assert "Run terlambat" not in h0 and "Run terlambat" not in t0
    b7, fc7, cfg7, (s7, h7, t7) = _email_at(7.0)
    assert "Run terlambat" in h7 and "Run terlambat" in t7
    assert "7 jam 30 menit setelah candle dibuka" in h7 and "7 jam 30 menit setelah candle dibuka" in t7
    assert "bukan kisaran pagi" in h7
    assert "Run terlambat" not in s7, "the banner belongs in the body, not in the subject"
    cfg_strict = dict(cfg7, email=dict(cfg7["email"], late_run_hours=12))
    _, h_cfg, _ = build_forecast_email(b7, fc7, cfg_strict)
    assert "Run terlambat" not in h_cfg, "threshold comes from config.yaml"


def test_email_basis_line_states_time_price_and_remaining_variance():
    for hours in (0.0, 7.0):
        b, fc, cfg, (_, html_body, text) = _email_at(hours)
        price = f"${b.reference_price:,.2f}"
        pct = f"{fc.remaining_var_frac:.0%}"
        for blob in (html_body, text):
            assert f"harga terakhir {price}" in blob, (hours, price)
            assert f"kira-kira {pct} variansi harian masih tersisa" in blob, (hours, pct)
        assert "Dihitung 05:30 WIB" in text if hours == 0 else "Dihitung 12:30 WIB" in text


def test_range_bar_marks_the_range_the_open_and_the_latest_price():
    import re
    html_bar = em_mod._range_bar_html(90.0, 110.0, 100.0, 95.0, open_known=True)
    cells = re.findall(r'bgcolor="(#[0-9a-f]{6})"', html_bar)
    assert len(cells) == 50
    # axis 88..112 -> cell index = int((x - 88) / 24 * 50): lo 4, hi 45, open 25, latest 14
    assert cells.count(em_mod._BAR_RANGE) == 40 and cells.count(em_mod._BAR_TRACK) == 8
    assert cells.count(em_mod._BAR_OPEN) == 1 and cells.count(em_mod._BAR_REF) == 1
    assert cells.index(em_mod._BAR_REF) == 14 and cells.index(em_mod._BAR_OPEN) == 25
    assert cells[3] == em_mod._BAR_TRACK and cells[4] == em_mod._BAR_RANGE and cells[45] == em_mod._BAR_RANGE \
        and cells[46] == em_mod._BAR_TRACK
    assert "open $100.00" in html_bar and "harga terakhir $95.00" in html_bar
    far = em_mod._range_bar_html(90.0, 110.0, 100.0, 50.0, open_known=True)      # price far below the range
    assert len(re.findall(r'bgcolor="(#[0-9a-f]{6})"', far)) == 50
    no_open = em_mod._range_bar_html(90.0, 110.0, 100.0, 95.0, open_known=False)
    assert em_mod._BAR_OPEN not in no_open and "open $" not in no_open
    assert em_mod._range_bar_html(100.0, 100.0, 100.0, 100.0, open_known=True) == ""   # degenerate: nothing to draw


def test_notes_are_grouped_titled_escaped_and_plain_strings_still_work():
    assert isinstance(Note("Judul", "isi"), str) and Note("Judul", "isi") == "isi" and "isi" in [Note("J", "isi")]
    # regression: str.title is a built-in METHOD, so a plain string must not be mistaken for a title
    assert note_title("plain") == "" and note_level("plain") == WARN
    notes = [Note("Judul A", "isi <b>x</b> & lainnya", WARN), Note("Judul B", "info b", INFO),
             "catatan polos dari radar berita", Note("Judul A", "isi <b>x</b> & lainnya", WARN)]
    notes = em_mod._dedupe_notes(notes)
    assert len(notes) == 3
    out = em_mod._notes_html(notes)
    assert out.index("Perlu diperhatikan") < out.index("Judul A") < out.index("catatan polos") < out.index(">Info<")
    assert "<b>Judul A</b>" in out and "&lt;b&gt;x&lt;/b&gt; &amp; lainnya" in out and "<b>x</b>" not in out
    assert out.index(">Info<") < out.index("Judul B")
    plain = em_mod._notes_plain(notes)
    assert plain[0] == "PERLU DIPERHATIKAN:" and "- Judul A: isi <b>x</b> & lainnya" in plain
    assert "- catatan polos dari radar berita" in plain and "INFO:" in plain and "- Judul B: info b" in plain
    assert em_mod._notes_html([]) == "" and em_mod._notes_plain([]) == []


def test_forecast_email_lists_real_notes_under_the_right_group():
    cfg = fast_cfg()
    now = T.NOW
    cfg["data"]["holidays"][str(pd.Timestamp("2026-01-14").date())] = {"name": "Test Holiday"}   # target session
    b = bundle_at(now, T.hourly_until(now, seed=11), cfg)
    fc = forecast_day(b, None, cfg)
    _, html_body, text = build_forecast_email(b, fc, cfg)
    assert "Sesi libur AS (Test Holiday)" in html_body and "Sesi libur AS (Test Holiday)" in text
    assert html_body.index("Perlu diperhatikan") < html_body.index("Sesi libur AS (Test Holiday)")
    assert "CATATAN:" not in text and "PERLU DIPERHATIKAN:" in text
    for english in ("liquidity is usually thin", "Source is Futures", "Daily fallback"):
        assert english not in html_body, english


def test_headline_label_stays_and_states_what_it_is_based_on():
    import dataclasses
    for hours, late in ((0.0, False), (7.0, True)):
        b, fc, cfg, _ = _email_at(hours)
        prefix = cfg["email"]["subject_prefix"]
        for bias, p_up in (("Bearish", 0.30), ("Bullish", 0.70), ("Netral", 0.50)):
            subject, html_body, text = build_forecast_email(b, dataclasses.replace(fc, bias=bias, p_up=p_up), cfg)
            shown = f"{bias} \u00b7 tengah sesi" if late else bias
            assert f"({shown})" in html_body and f"({shown})" in text, (hours, bias)
            if bias == "Netral":
                assert "Netral" not in subject, "the Netral word stays out of the subject, as before"
            else:
                tag = " (tengah sesi)" if late else ""
                assert subject.startswith(f"{prefix} {bias}{tag} | Close"), subject
        delta = b.reference_price - fc.open_price
        side = "hampir sama dengan" if abs(delta) < 0.5 else f"${abs(delta):,.2f} {'di atas' if delta > 0 else 'di bawah'}"
        expected = f"harga terakhir ${b.reference_price:,.2f}, {side} open ${fc.open_price:,.2f}"
        for blob in (html_body, text):
            assert "Dasar label:" in blob and expected in blob, (hours, expected)
            assert f"sisa variansi harian {fc.remaining_var_frac:.0%}" in blob
            if late:
                assert "bukan prediksi arah dari awal sesi" in blob and "Backtest: arah harian" not in blob
            else:
                assert "Backtest: arah harian tidak bisa ditebak" in blob and "tengah sesi" not in blob


def test_headline_basis_before_the_candle_opens_says_so():
    cfg = fast_cfg()
    sunday = T.utc("2026-01-11 22:30")
    hourly = T.hourly_until(sunday, seed=11)
    hourly = hourly[hourly.index < T.utc("2026-01-11 22:00")]
    b = bundle_at(sunday, hourly, cfg)
    fc = forecast_day(b, None, cfg)
    assert not fc.open_known
    _, html_body, text = build_forecast_email(b, fc, cfg)
    assert "Dasar label: candle belum dibuka" in html_body and "Dasar label: candle belum dibuka" in text


def test_optional_blocks_never_break_the_main_email():
    b, fc, cfg, _ = _email_at(0.0)
    real_bar, real_timing = em_mod._range_bar_html, em_mod.run_timing
    try:
        def boom(*a, **k):
            raise RuntimeError("decoration failed")
        em_mod._range_bar_html = boom
        _, html_body, _ = build_forecast_email(b, fc, cfg)
        assert "Kisaran close D1 hari ini" in html_body and 'bgcolor="#f59e0b"' not in html_body
        em_mod._range_bar_html = real_bar
        em_mod.run_timing = lambda *a, **k: None
        _, html_body, text = build_forecast_email(b, fc, cfg)
        assert "Run terlambat" not in html_body and "sisa sesi" not in html_body and "Dihitung 05:30 WIB" in text
    finally:
        em_mod._range_bar_html, em_mod.run_timing = real_bar, real_timing


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
