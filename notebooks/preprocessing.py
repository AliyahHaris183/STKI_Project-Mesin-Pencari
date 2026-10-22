"""
Modul 1 - Preprocessing Teks
Tahapan: pemuatan korpus, case folding, tokenisasi, penghapusan stopword, stemming.
Korpus: data/korpus.csv (30 dokumen berita berbahasa Indonesia).
"""
from pathlib import Path

import pandas as pd

# Folder data/ berada satu tingkat di atas folder notebooks/ tempat file ini berada
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)


# ---------------------------------------------------------------------------
# Langkah 1 - Memuat korpus
# ---------------------------------------------------------------------------
print("\n[Langkah 1] Memuat korpus")
korpus = pd.read_csv(DATA_DIR / "korpus.csv")
print("Jumlah dokumen:", len(korpus))
print(korpus[["doc_id", "judul"]].head().to_string(index=False))

# Judul dan isi digabung menjadi satu teks dokumen
korpus["teks"] = korpus["judul"] + ". " + korpus["isi"]
contoh = korpus.loc[16, "teks"]
print("\nTeks contoh (D17):")
print(contoh)


# ---------------------------------------------------------------------------
# Langkah 2 - Case folding
# Menyeragamkan huruf menjadi huruf kecil
# ---------------------------------------------------------------------------
print("\n[Langkah 2] Case folding")


def case_folding(teks):
    return teks.lower()


print(case_folding(contoh))


# ---------------------------------------------------------------------------
# Langkah 3 - Tokenisasi
# Setiap rangkaian huruf/angka menjadi satu token, tanda baca menjadi pemisah
# ---------------------------------------------------------------------------
print("\n[Langkah 3] Tokenisasi")
import re


def tokenize(teks):
    return re.findall(r"[a-z0-9]+", teks.lower())


token_contoh = tokenize(case_folding(contoh))
print(f"Jumlah token: {len(token_contoh)}")
print(token_contoh)

# Pembanding: split() tidak membuang tanda baca
print("Pembanding split():", case_folding(contoh).split()[:12])


# ---------------------------------------------------------------------------
# Langkah 4 - Penghapusan stopword
# Daftar stopword bahasa Indonesia dari Sastrawi
# ---------------------------------------------------------------------------
print("\n[Langkah 4] Penghapusan stopword")
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
print("Jumlah stopword:", len(STOPWORDS))
print("40 stopword pertama (urut abjad):", sorted(STOPWORDS)[:40])


def remove_stopwords(tokens, stopwords=STOPWORDS):
    return [t for t in tokens if t not in stopwords]


tanpa_sw = remove_stopwords(token_contoh)
print(f"Jumlah token: {len(token_contoh)} -> {len(tanpa_sw)}")
print(tanpa_sw)


# ---------------------------------------------------------------------------
# Langkah 5 - Stemming
# Mengubah kata berimbuhan menjadi kata dasar, dengan cache agar setiap
# kata unik hanya di-stem satu kali
# ---------------------------------------------------------------------------
print("\n[Langkah 5] Stemming")
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer = StemmerFactory().create_stemmer()
for kata in ["pembelajaran", "pertanian", "petani", "pencarian", "keamanan",
             "memantau", "berolahraga", "perekonomian", "mahasiswa"]:
    print(f"  {kata:<14} -> {stemmer.stem(kata)}")

_cache = {}


def stem(token):
    if token not in _cache:
        _cache[token] = stemmer.stem(token)
    return _cache[token]


print("Hasil stemming pada contoh:")
print(" ".join(stem(t) for t in tanpa_sw))


# ---------------------------------------------------------------------------
# Langkah 6 - Menggabungkan menjadi satu fungsi preprocess
# ---------------------------------------------------------------------------
print("\n[Langkah 6] Fungsi preprocess")


def preprocess(teks, use_stopword=True, use_stemming=True):
    tokens = tokenize(case_folding(teks))
    if use_stopword:
        tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


korpus["tokens"] = korpus["teks"].apply(preprocess)
ringkas = pd.DataFrame({
    "doc_id": korpus["doc_id"],
    "jumlah_token": korpus["tokens"].str.len(),
    "8 token pertama": korpus["tokens"].apply(lambda t: " ".join(t[:8])),
})
print(ringkas.head().to_string(index=False))


# ---------------------------------------------------------------------------
# Langkah 7 - Mengukur pengaruh tiap tahap
# Jumlah token dan ukuran kosakata untuk setiap kombinasi tahap
# ---------------------------------------------------------------------------
print("\n[Langkah 7] Pengaruh tiap tahap")
tahap = {
    "Tokenisasi saja": dict(use_stopword=False, use_stemming=False),
    "+ Stopword removal": dict(use_stopword=True, use_stemming=False),
    "+ Stemming (tanpa stopword removal)": dict(use_stopword=False, use_stemming=True),
    "+ Stopword removal + Stemming": dict(use_stopword=True, use_stemming=True),
}
baris = []
for nama, opsi in tahap.items():
    semua = [t for teks in korpus["teks"] for t in preprocess(teks, **opsi)]
    baris.append({"Tahap": nama, "Jumlah token": len(semua), "Ukuran kosakata": len(set(semua))})
print(pd.DataFrame(baris).to_string(index=False))

# Frekuensi kemunculan term setelah preprocessing lengkap
from collections import Counter

print("\nFrekuensi term teratas (setelah preprocessing lengkap):")
frekuensi = Counter(t for tokens in korpus["tokens"] for t in tokens)
print(pd.DataFrame(frekuensi.most_common(15), columns=["term", "frekuensi"]).to_string(index=False))


# ---------------------------------------------------------------------------
# Langkah 8 - Memastikan fungsi identik dengan stki_utils.py
# Fungsi yang sama tersimpan di stki_utils.py agar modul berikutnya cukup
# mengimpornya
# ---------------------------------------------------------------------------
print("\n[Langkah 8] Pencocokan dengan stki_utils.py")
import stki_utils

cocok = all(preprocess(t) == stki_utils.preprocess(t) for t in korpus["teks"])
print("Hasil identik dengan stki_utils.preprocess:", cocok)


# ===========================================================================
# TUGAS STEP 1
# ===========================================================================

# ---------------------------------------------------------------------------
# Tugas 1 - Mencari kandidat overstemming dan understemming
# Stem yang menaungi lebih dari satu kata asli ditampilkan sebagai kandidat
# ---------------------------------------------------------------------------
print("\n[Tugas 1] Kandidat overstemming")
from collections import defaultdict

kata_asli = {t for teks in korpus["teks"]
             for t in remove_stopwords(tokenize(case_folding(teks)))}
kelompok = defaultdict(set)
for kata in kata_asli:
    kelompok[stem(kata)].add(kata)

gabungan = {s: sorted(k) for s, k in kelompok.items() if len(k) > 1}
print(f"Jumlah stem yang menaungi lebih dari satu kata: {len(gabungan)}")
for s, daftar in sorted(gabungan.items()):
    print(f"  {s:<12} <- {', '.join(daftar)}")

# Pengecekan bentuk kata tertentu
print("\nPengecekan bentuk kata tertentu:")
for kata in ["melalui", "pembelajaran", "tokenisasi", "pemrosesan", "imbuhan", "berimbuhan"]:
    print(f"  {kata:<14} -> {stem(kata)}")

# Kemunculan bentuk mentah sebelum stemming, untuk dibandingkan dengan
# frekuensi stem pada Langkah 7
mentah = [t for teks in korpus["teks"] for t in tokenize(case_folding(teks))]
print("\nBentuk mentah 'melalui':", mentah.count("melalui"), "| 'lalu':", mentah.count("lalu"))


# ---------------------------------------------------------------------------
# Tugas 2 - Stopword khusus domain
# Ukuran kosakata dibandingkan sebelum dan sesudah penambahan stopword
# ---------------------------------------------------------------------------
print("\n[Tugas 2] Stopword khusus domain")
STOPWORDS_DOMAIN = STOPWORDS | {"dapat", "membantu"}


def ukuran_kosakata(stopwords):
    return len({stem(t) for teks in korpus["teks"]
                for t in remove_stopwords(tokenize(case_folding(teks)), stopwords)})


print("Kosakata sebelum:", ukuran_kosakata(STOPWORDS))
print("Kosakata sesudah:", ukuran_kosakata(STOPWORDS_DOMAIN))
print("Kemunculan bentuk mentah 'membantu':", mentah.count("membantu"))
print("Bentuk mentah lain yang mengandung 'bantu':",
      sorted({t for t in mentah if "bantu" in t and t != "membantu"}))


# ---------------------------------------------------------------------------
# Tugas 3 - Tokenizer yang menjaga kata berulang dengan tanda hubung
# Pola: rangkaian huruf/angka, boleh diikuti "-" dan rangkaian huruf/angka lagi
# ---------------------------------------------------------------------------
print("\n[Tugas 3] Tokenizer dengan tanda hubung")


def tokenize_hubung(teks):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", teks.lower())


kalimat_uji = "Anak-anak bermain di kebun sayur-mayur, bukan di jalan raya."
print("Kalimat uji        :", kalimat_uji)
print("Tokenizer baru     :", tokenize_hubung(case_folding(kalimat_uji)))
print("Tokenizer lama     :", tokenize(case_folding(kalimat_uji)))


# ---------------------------------------------------------------------------
# Tugas 4 - Analisis tahap yang paling banyak mengurangi kosakata
# Hasil perhitungan ditampilkan dari tabel Langkah 7
# ---------------------------------------------------------------------------
print("\n[Tugas 4] Pengurangan kosakata per tahap")
tabel = pd.DataFrame(baris).set_index("Tahap")
dasar = tabel.loc["Tokenisasi saja", "Ukuran kosakata"]
print(tabel.assign(Pengurangan=lambda d: dasar - d["Ukuran kosakata"]).to_string())


# ===========================================================================
# JAWABAN TUGAS STEP 1
# ===========================================================================
#
# Tugas 1 - Overstemming dan understemming
#   Overstemming (kata dengan makna berbeda menjadi stem yang sama):
#     - "melalui" menjadi "lalu". Bentuk mentah "melalui" muncul 8 kali dan
#       "lalu" muncul 1 kali, sehingga stem "lalu" berjumlah 9 pada tabel
#       frekuensi Langkah 7. Pada D01, "ditularkan melalui gigitan" menjadi
#       "tular lalu gigit". "Lalu" berarti waktu, sehingga kueri tentang waktu
#       dapat mengambil dokumen yang memuat "melalui" (berarti cara atau
#       perantara). Presisi turun.
#     - "meningkat" dan "tingkat" menjadi "tingkat". "Meningkat" berarti naik,
#       sedangkan "tingkat" berarti level atau taraf. Stem "tingkat" berjumlah 6
#       pada Langkah 7, sehingga kueri tentang kenaikan dapat mencocokkan dokumen
#       yang hanya memuat "tingkat" dalam arti level.
#     - "buatan" dan "membuat" menjadi "buat". Frasa "kecerdasan buatan" (artificial
#       intelligence) menjadi tidak spesifik, karena dokumen yang memuat
#       "membuat" (make) ikut cocok dengan kata "buatan" pada kueri.
#   Understemming (makna sama tetapi stem berbeda):
#     - "pemrosesan" tetap utuh (2 kemunculan), sehingga tidak tergabung dengan
#       "proses" atau "memproses". Kedua kata itu tidak ada di korpus, tetapi
#       pada kueri "proses" tidak akan menemukan dokumen yang memuat "pemrosesan".
#     - "tokenisasi" dan "stemming" tetap utuh, sehingga tidak terhubung dengan
#       "token" atau "stem". Kueri "token" tidak menemukan dokumen dengan bentuk
#       "tokenisasi".
#   Pasangan yang tergabung dengan benar: "pertanian" dan "petani" menjadi "tani",
#   "imbuhan" dan "berimbuhan" menjadi "imbuh", "berolahraga" dan "olahraga"
#   menjadi "olahraga", serta "kesehatan" dan "sehat" menjadi "sehat".
#
# Tugas 2 - Stopword domain
#   "dapat" sudah berada di daftar stopword Sastrawi (terlihat pada daftar
#   Langkah 4), sehingga penambahannya tidak mengubah ukuran kosakata.
#   "membantu" berbeda: stopword dicek sebelum stemming, sehingga bentuk mentah
#   "membantu" harus ada di daftar agar terhapus. Bentuk "membantu" muncul 8 kali
#   dan tidak ada bentuk mentah lain yang mengandung "bantu", sehingga semua 8
#   kemunculan stem "bantu" (tabel frekuensi Langkah 7) berasal dari "membantu".
#   Akibatnya kosakata berkurang 1 term, dari 472 menjadi 471, dan 8 token dihapus.
#   Stopword diperiksa pada token mentah, sehingga bentuk yang ditambahkan harus
#   sama dengan bentuk di korpus, bukan stem-nya.
#
# Tugas 3 - Tokenizer dengan tanda hubung
#   Pola "[a-z0-9]+(?:-[a-z0-9]+)*" menjaga "anak-anak" dan "sayur-mayur" tetap
#   satu token. Tokenizer lama memecahnya menjadi dua token. Pada korpus ini tidak
#   ada token yang mengandung tanda hubung, sehingga jumlah token tetap 1033 di
#   kedua tokenizer. Hasil stemming untuk token yang mengandung tanda hubung perlu
#   dicek secara terpisah, karena stemmer dirancang untuk kata tanpa tanda hubung.
#
# Tugas 4 - Tahap dengan pengurangan kosakata terbesar
#   Stemming paling banyak mengurangi ukuran kosakata. Dari 574 term (tokenisasi
#   saja), stemming menurunkan menjadi 501 (berkurang 73 atau 12,7%), sedangkan
#   stopword removal hanya menurunkan menjadi 541 (berkurang 33 atau 5,7%).
#   Gabungan keduanya menghasilkan 472 (berkurang 102 atau 17,8%).
#   Penyebabnya: stemming menyatukan banyak bentuk berimbuhan yang berbeda menjadi
#   satu kata dasar. Stopword removal menghapus banyak token (1033 menjadi 863,
#   berkurang 16,5%), tetapi kata stopword yang sama muncul berulang kali. Setiap
#   jenis kata hanya dihitung sekali dalam kosakata, sehingga penghapusan token
#   berulang tidak banyak mengurangi jumlah jenis kata.