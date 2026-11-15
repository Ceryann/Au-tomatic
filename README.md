# XAUUSD Quant Sentinel

Bot yang mengirim **satu email pagi (WIB)** berisi perkiraan kisaran **Close, High, dan Low** candle D1 XAUUSD hari ini.
Batas candle D1 = **17:00 New York** (sama dengan open candle D1 di MT5).

> Estimasi statistik, bukan nasihat finansial. Bukan sinyal entry/SL/TP.

**Status: Sprint 1 dari 5** — lapisan data + validasi + email tes. Prediksi (Kalman, HMM, Monte Carlo) hadir di Sprint 2.

## Apa yang dilakukan Sprint 1

1. Mengunduh bar 1 jam emas dari Yahoo (`XAUUSD=X`, cadangan `GC=F`, lalu data harian).
2. Merangkai candle D1 dengan batas 17:00 New York (DST ditangani otomatis).
3. Membuang candle yang belum selesai, lalu memvalidasi (tanggal terakhir, OHLC, bar cukup, feed tidak basi).
4. Mengirim email tes (bukan prediksi) berisi harga acuan, candle terakhir, sumber data, dan peringatan.
5. Jika data gagal/basi: mengirim email **"prediksi tidak tersedia"**, tidak pernah angka lama.

## Setup (sekali saja)

### 1. Buat repo Public di GitHub dan unggah folder ini

### 2. Siapkan Gmail App Password
Akun Google harus memakai verifikasi 2 langkah. Buka *Akun Google → Keamanan → Sandi aplikasi*, buat satu sandi (16 karakter).

### 3. Isi GitHub Secrets
Repo → *Settings → Secrets and variables → Actions → New repository secret*:

| Nama | Isi |
|---|---|
| `GMAIL_USER` | alamat Gmail pengirim |
| `GMAIL_APP_PASSWORD` | sandi aplikasi 16 karakter (tanpa spasi) |
| `EMAIL_TO` | alamat penerima (boleh beberapa, pisahkan koma) |

Jangan menulis alamat email atau sandi di kode, log, atau commit (repo ini Public).

### 4. Uji manual
*Actions → Daily XAUUSD Notifier → Run workflow*:
1. Jalankan dengan **dry_run = true**. Unduh artifact `email-preview` dan cek `preview.html`.
2. Jalankan dengan **dry_run = false**. Email tes harus masuk ke inbox.

Jadwal otomatis: **Minggu–Kamis 22:30 UTC** (= Senin–Jumat 05:30 WIB). GitHub bisa menunda cron beberapa menit.

## Menjalankan di komputer sendiri

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                   # tes offline, tanpa internet
python main.py --dry-run                              # hasil di out/preview.html
python main.py --dry-run --now 2026-11-13T22:30:00Z   # ulang momen tertentu
```

## Validasi terhadap chart MT5 Anda (disarankan sebelum Sprint 2)

1. MT5 → *View → Symbols → tab Bars* → pilih simbol emas Anda, timeframe **Daily (D1)**, rentang 60+ hari.
2. *Request* → *Export Bars* → simpan CSV.
3. Jalankan:

```bash
python tools/compare_mt5.py XAUUSD_Daily.csv --days 30
```

Yang dicek skrip: (a) **lag harus 0** (batas candle sama), (b) selisih level Close wajar (beberapa dolar), (c) selisih High/Low. High Yahoo cenderung sedikit di bawah High MT5 karena bar 1 jam bisa melewatkan lonjakan singkat.

## Struktur

```text
main.py                     # orkestrator
config.yaml                 # batas candle, ambang validasi, SMTP
src/data_fetcher.py         # unduh, candle NY-close, validasi, fallback
src/email_dispatcher.py     # email HTML + teks, Gmail SMTP
tests/test_data_layer.py    # tes offline (sesi, DST, fallback, stale, email)
tools/compare_mt5.py        # bandingkan dengan ekspor MT5
.github/workflows/          # daily_xauusd.yml (jadwal) dan ci.yml (tes)
```

## Keterbatasan yang diketahui (Sprint 1)

- Data Yahoo tidak resmi dan bisa terlambat, kosong, atau diblokir (429). Itu sebabnya ada retry, fallback, dan email error.
- Kode ini diuji offline dengan data sintetis. **Uji pertama terhadap Yahoo live adalah dry-run di langkah 4.**
- Fallback `GC=F` (Futures) memiliki basis harga berbeda dari spot dan belum dibersihkan dari lonjakan contract roll (Sprint 2).
- Cron GitHub di repo Public dinonaktifkan setelah 60 hari tanpa aktivitas repo. Commit ledger harian (Sprint 3) akan mencegahnya.

## Roadmap

1. **Sprint 1** — data + validasi + email tes (ini)
2. Sprint 2 — Kalman + HMM + Monte Carlo intraday (close, high, low) + email final
3. Sprint 3 — walk-forward + ledger + email error + tes tambahan
4. Sprint 4 — radar berita/kalender + README portofolio + lisensi
5. Sprint 5 (opsional) — Ridge/LightGBM + atribusi faktor
