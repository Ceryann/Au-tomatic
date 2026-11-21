# Au-tomatic

*Au = Aurum (emas). Notifikasi otomatis kisaran candle D1 XAUUSD.*

Bot yang mengirim **satu email pagi (WIB)** berisi perkiraan kisaran **Close, High, dan Low** candle D1 XAUUSD hari ini.
Batas candle D1 = **17:00 New York** (sama dengan open candle D1 di MT5). Semua kisaran adalah **interval prediksi 70% (P15 sampai P85)**.

> Estimasi statistik, bukan nasihat finansial. Bukan sinyal entry/SL/TP.

**Status: Sprint 2 dari 5.** Data spot + model + email prakiraan sudah ada. Radar berita/kalender (Sprint 4) belum aktif.

## Cara kerja

1. **Data:** harga spot XAU/USD per jam dari Twelve Data (± 2 tahun). Cadangan: futures `GC=F` dari Yahoo (levelnya bukan spot, dan email melabelinya).
2. **Candle D1** dibentuk dengan batas 17:00 New York, candle yang belum selesai dibuang, lalu data divalidasi.
3. **Kalman Filter** (local linear trend) mengestimasi kecepatan tren dalam $/hari.
4. **HMM 3 rezim** (Konsolidasi, Trending, Shock) dari return harian, volatilitas 10 hari, dan opsional DXY serta US10Y. Rezim diberi nama berdasarkan peringkat volatilitasnya, jadi "Trending" berarti "aktif/volatilitas menengah".
5. **Monte Carlo Merton jump-diffusion** (10.000 jalur, per jam) dari harga terbaru sampai penutupan sesi, dengan profil volatilitas intraday dan High/Low yang sudah terjadi dihitung. Hasilnya kisaran Close, High, Low, dan peluang candle naik.
6. **Email** menampilkan kisaran absolut dan **relatif terhadap open**, supaya bisa dipakai langsung di atas harga open MT5 Anda.

Yang model **tidak** lakukan: menebak arah dengan akurat. Untuk emas harian, peluang naik wajar berada di 45 sampai 58%. Label **Netral** muncul jika peluang di antara 44% dan 56%.

## Setup (sekali saja)

### 1. Buat repo Public di GitHub dan unggah folder ini

### 2. Gmail App Password
Akun Google harus memakai verifikasi 2 langkah. Buka *Akun Google → Keamanan → Sandi aplikasi*, buat satu sandi (16 karakter).

### 3. API key Twelve Data
Daftar gratis di twelvedata.com. Paket gratis cukup: bot memakai sekitar 4 sampai 5 credits per hari dari kuota 800. Baca syarat penggunaan paket gratis, dan jangan menaruh data harga mentah di repo Public.

### 4. GitHub Secrets
Repo → *Settings → Secrets and variables → Actions → New repository secret*:

| Nama | Isi |
|---|---|
| `TWELVEDATA_API_KEY` | API key Twelve Data |
| `GMAIL_USER` | alamat Gmail pengirim |
| `GMAIL_APP_PASSWORD` | sandi aplikasi 16 karakter (tanpa spasi) |
| `EMAIL_TO` | alamat penerima (boleh beberapa, pisahkan koma) |

Jangan menulis kunci atau alamat email di kode, log, atau commit (repo ini Public).

### 5. Uji manual
*Actions → Au-tomatic Daily → Run workflow*:
1. **dry_run = true**: unduh artifact `email-preview` dan cek `preview.html`.
2. **dry_run = false**: email sungguhan masuk ke inbox.

Jadwal otomatis: **Minggu–Kamis 22:30 UTC** (= Senin–Jumat 05:30 WIB). GitHub bisa menunda cron beberapa menit.

## Menjalankan di komputer sendiri (Windows PowerShell)

```powershell
cd "C:\Users\...\au-tomatic"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt        # gunakan "python -m pip", bukan "pip"
python -m pytest -q                              # tes offline, tanpa internet

$env:TWELVEDATA_API_KEY = "key-anda"             # hanya berlaku di jendela ini
python main.py --dry-run                         # hasil di out\preview.html
```

Opsi `main.py`:

| Opsi | Fungsi |
|---|---|
| `--dry-run` | tidak mengirim email, hanya menulis `out/preview.html` dan `preview.txt` |
| `--now 2026-11-13T22:30:00Z` | memutar ulang momen tertentu |
| `--data-check` | hanya email tes data (Sprint 1) |
| `--no-macro` | tanpa fitur DXY/US10Y |
| `--event` | anggap hari ini hari event high-impact (kisaran dilebarkan) |

## Uji kualitas model di data asli (WAJIB sebelum dipercaya)

```powershell
python tools/backtest.py --sessions 120
```

Memakai riwayat Twelve Data asli (3 sampai 4 credits, ± 1 menit). Model dijalankan seolah-olah pada pagi hari itu, hanya memakai data yang tersedia saat itu, lalu dibandingkan dengan kenyataan. Cara membaca:

- **inside ≈ 70%** untuk Close/High/Low. Untuk 120 sesi, 60 sampai 80% masih wajar.
- **Pinball gain vs baseline positif** berarti model mengalahkan random walk sederhana. Sekitar 0 berarti setara. Negatif berarti model lebih buruk, dan itu temuan penting.
- **Direction Brier mendekati 0,25** berarti tidak ada kemampuan menebak arah. Itu **normal** untuk emas harian.
- Tabel per sesi tersimpan di `out/backtest_table.csv`.

## Validasi terhadap chart MT5

```powershell
python tools/compare_mt5.py "C:\path\XAUUSD_D1.csv" --days 30
python tools/probe_twelvedata.py "C:\path\XAUUSD_D1.csv"     # uji sumber spot + kedalaman riwayat
python tools/probe_symbols.py "C:\path\XAUUSD_D1.csv"        # uji simbol Yahoo
```

Ekspor dari MT5: *View → Symbols → tab Bars* → simbol emas, timeframe D1, 60+ hari → *Request* → *Export Bars* (CSV).

## Struktur

```text
main.py                     # orkestrator
config.yaml                 # batas candle, parameter model, SMTP
src/data_fetcher.py         # Twelve Data + Yahoo, candle NY-close, validasi, fallback, makro
src/hmm.py                  # Gaussian HMM (numpy, tanpa hmmlearn)
src/signal_and_regime.py    # Kalman local linear trend, fitur, pemetaan rezim
src/quant_engine.py         # volatilitas, lompatan, Monte Carlo, bias
src/evaluation.py           # walk-forward: coverage, pinball, Brier
src/email_dispatcher.py     # email HTML + teks, Gmail SMTP
tests/                      # 33 tes offline
tools/                      # backtest, compare_mt5, probe_twelvedata, probe_symbols
.github/workflows/          # daily_xauusd.yml (jadwal) dan ci.yml (tes)
```

## Keterbatasan yang diketahui

- Data gratis tidak resmi dan bisa terlambat atau diblokir. Itu sebabnya ada retry, fallback bertingkat, dan email error (tidak pernah mengirim angka basi).
- Model dibuktikan terkalibrasi pada data **sintetis**. Pembuktian pada data asli adalah `tools/backtest.py`.
- Fallback `GC=F` memiliki basis harga berbeda dari spot. Pakai kisaran relatif terhadap open MT5 Anda.
- Radar berita dan kalender ekonomi belum aktif (Sprint 4). Hari FOMC/CPI/NFP, kisaran bisa lebih lebar dari yang ditampilkan.
- Cron GitHub di repo Public dinonaktifkan setelah 60 hari tanpa aktivitas repo. Commit ledger harian (Sprint 3) akan mencegahnya.

## Roadmap

1. Sprint 1: data + validasi + email tes (selesai)
2. **Sprint 2: Kalman + HMM + Monte Carlo + email prakiraan (ini)**
3. Sprint 3: ledger harian + coverage live di email + email error lengkap
4. Sprint 4: radar berita/kalender + README portofolio + lisensi
5. Sprint 5 (opsional): Ridge/LightGBM + atribusi faktor, hanya jika mengalahkan baseline
