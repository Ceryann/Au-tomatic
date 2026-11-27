# Metodologi Au-tomatic

Dokumen ini menjelaskan apa yang dihitung bot, asumsinya, apa yang sudah terukur, dan apa yang **belum** terbukti. Contoh keluaran memakai **data sintetis** (random walk), bukan harga pasar: lihat `docs/samples/` dan `docs/img/`. Semua parameter ada di `config.yaml`.

## 1. Pertanyaan yang dijawab

Bot memperkirakan **kisaran** Close, High, dan Low candle D1 XAUUSD hari ini sebagai **interval prediksi 70%** (kuantil P15 sampai P85). Candle D1 dibatasi pukul **17:00 New York**, sama dengan candle D1 broker MT5 yang dipakai penulis.

Yang **bukan** tujuannya: sinyal entry, SL, TP, atau tebakan arah. Hasil uji (bagian 6) menunjukkan arah harian tidak dapat diprediksi dari tren, dan bot tidak mengklaim sebaliknya.

## 2. Alur data

1. **Sumber:** Twelve Data XAU/USD spot per jam (paket gratis, riwayat sekitar 2,8 tahun). Cadangan: Yahoo `GC=F` (futures, diberi label karena basisnya berbeda dari spot).
2. **Candle D1:** bar per jam dikelompokkan ke sesi yang berakhir 17:00 New York. Candle yang belum selesai dibuang.
3. **Validasi:** tanggal candle terakhir harus sesuai sesi yang diharapkan, OHLC konsisten, jumlah bar per sesi cukup (`data.min_bars_per_session`), dan feed tidak basi. Jika gagal, yang dikirim adalah email "prediksi tidak tersedia", **tidak pernah angka basi**.
4. **Hari libur AS:** sesi yang terdaftar di `data.holidays` diperiksa dengan ambang bar lebih rendah dan, bila `absent_ok`, boleh kosong. Daftar dan ambangnya masih **sementara** sampai `tools/probe_holidays.py` dijalankan di data asli. Bila sesi target adalah hari libur `absent_ok` dan semua sumber data gagal, yang dikirim adalah email singkat "pasar libur", bukan email error.
5. **Akhir pekan:** run hari Senin memakai harga acuan close Jumat dan menambah risiko gap (`model.jumps.sunday_mult`).

## 3. Model

| Komponen | Peran | Catatan |
|---|---|---|
| Kalman local linear trend | kecepatan tren ($/hari) | hanya estimasi *filtered*, tanpa kebocoran data masa depan |
| HMM Gaussian 3 rezim | rezim volatilitas | ditulis sendiri dengan numpy (EM berskala); rezim dinamai dari peringkat volatilitas: Konsolidasi, Trending, Shock |
| Volatilitas | sigma harian | campuran volatilitas rezim dan EWMA RiskMetrics (`lambda` 0,94), dibatasi 0,5x sampai 2x volatilitas realisasi 60 hari |
| Drift | pergeseran median | rata-rata rezim dan kecepatan Kalman, masing-masing dikecilkan 0,5x, lalu dipotong pada ±0,2 sigma |
| Monte Carlo Merton | distribusi jalur harga | 10.000 jalur per jam sampai penutupan candle; lompatan Poisson dengan intensitas dikalibrasi dari hari lompatan historis |
| Brownian bridge | High dan Low | koreksi agar High/Low simulasi tidak bias rendah karena diskretisasi per jam |
| Profil intraday | bentuk volatilitas sepanjang hari | High/Low yang sudah terjadi hari ini digabung dengan simulasi sisa hari |

### Event USD berdampak tinggi

Pada jam rilis (CPI, NFP, FOMC, dan sejenisnya) ditambahkan varians lompatan: `model.jumps.event_extra_var = 0,5`, artinya hari itu kira-kira 1,5x lebih bervariasi, maksimal dua event per hari. Nilai ini adalah **prior**. Event study FOMC (22 hari) memberi rasio varians 1,51 dengan selang 0,61 sampai 2,69: konsisten dengan asumsi, tetapi selangnya lebar.

### Berita

Kalender ForexFactory (JSON mingguan) dan RSS Google News dibaca; kalender cadangan adalah jadwal FOMC statis. Sentimen memakai kamus heuristik khusus emas dan **hanya ditampilkan**, tidak mengubah angka, karena tidak ada arsip berita historis untuk memvalidasinya.

## 4. Ledger akurasi live

`data/predictions_log.csv` hanya menyimpan pecahan relatif terhadap harga dasar, probabilitas, rezim, dan penanda 0/1 (kena atau tidak). **Tanpa harga mentah**, sesuai syarat Twelve Data paket gratis dan karena repo ini publik. Ledger di-commit otomatis oleh workflow dan akurasi 30 hari tampil di email setelah 10 hari data.

## 5. Cara menguji

Semua tes berjalan **offline** (helper tes memblokir jaringan) dan memakai data sintetis yang jawabannya diketahui:

```bash
python -m pytest -q
# atau satu per satu:
python tests/test_data_layer.py && python tests/test_models.py && python tests/test_ledger.py && python tests/test_news.py
```

Alat evaluasi memakai data asli dan butuh kunci API di komputer Anda:

| Alat | Fungsi |
|---|---|
| `tools/backtest.py` | walk-forward: cakupan, pinball loss vs random walk, Brier arah |
| `tools/event_study.py` | rasio varians pada hari FOMC |
| `tools/compare_mt5.py` | bandingkan candle D1 hasil bot dengan MT5 |
| `tools/probe_*.py` | uji sumber data, simbol, berita, hari libur, dan SMTP |
| `tools/make_samples.py` | membuat contoh email dan laporan sintetis |

## 6. Hasil terukur

Walk-forward pada 120 sesi emas asli (21 November 2026):

| Ukuran | Hasil | Arti |
|---|---|---|
| Close di dalam kisaran 70% | 63,3% (±8,6%) | sedikit di bawah target |
| High di dalam kisaran | 68,3% (±8,3%) | cocok |
| Low di dalam kisaran | 65,8% (±8,5%) | cocok |
| Pinball loss vs random walk + EWMA | −0,5% | setara, model tidak lebih baik untuk Close |
| Direction Brier | 0,2484 (acak = 0,2500) | **tidak ada kemampuan menebak arah** |
| Ekstrapolasi tren penuh | Brier 0,2488; 2 panggilan arah, 1 benar | tidak membantu |

Kesimpulan: nilai bot ada pada **kisaran yang terkalibrasi dan perkiraan High/Low**. Pada run di awal sesi (kondisi backtest) label Bullish atau Bearish hampir tidak pernah muncul, dan itu disengaja: ambang (`model.bias`) tidak diturunkan demi menghasilkan sinyal. Pada run yang lebih siang label sering muncul karena peluang dihitung dari harga terakhir dan sisa sesi, sehingga ia hanya mencerminkan posisi harga terhadap open. Uji 26 Nov 2026 pada 150 hari sintetis tanpa kemampuan arah: label non-Netral muncul pada 56% run di jam 0,5 sejak open (data sintetis punya tren bawaan, jadi angka ini tidak mewakili emas) dan 95% pada jam 17, sedangkan kesesuaian label dengan arah penutupan naik dari 57% ke 87% tanpa ada keahlian. Label run siang tidak boleh dipakai sebagai bukti akurasi arah.

## 7. Batasan dan hal yang belum terverifikasi

- Sampel backtest kecil (120 sesi); selang kepercayaannya lebar. Akurasi live baru bermakna setelah ledger terkumpul beberapa minggu.
- `event_extra_var = 0,5` adalah prior yang konsisten dengan 22 hari FOMC, bukan hasil pasti.
- Kamus sentimen berita tidak dapat divalidasi.
- Feed gratis (data, kalender, berita) dapat berubah, terlambat, atau mati. Feed yang gagal dilewati dan email menyebutkannya.
- Perilaku bot pada hari libur AS dan fitur makro Yahoo (DXY, US10Y) belum diverifikasi pada data asli.
- Model terkalibrasi pada data sintetis; kecocokan pada data asli diukur hanya sejauh bagian 6.

## 8. Keamanan dan privasi

- Repo publik: log tidak mencetak level harga, alamat email, atau token; workflow tidak mengunggah file hasil.
- Email dikirim lewat **Gmail API (HTTPS)** dengan izin `gmail.send` saja. Login SMTP dari server GitHub terbukti ditutup oleh Google (`tools/probe_smtp.py`), jadi SMTP hanya cadangan untuk uji lokal.
- Secret yang dipakai workflow: `EMAIL_TO`, `GMAIL_OAUTH_CLIENT_ID`, `GMAIL_OAUTH_CLIENT_SECRET`, `GMAIL_OAUTH_REFRESH_TOKEN`, `TWELVEDATA_API_KEY`.
- Syarat Twelve Data paket gratis: penggunaan pribadi saja, jadi data mentah tidak disimpan atau dibagikan di repo ini.
