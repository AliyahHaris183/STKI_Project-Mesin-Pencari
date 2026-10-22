"""
Step 4 - Vector Space Model dan TF-IDF (STKI)
Jalankan dari folder notebooks/ :  python step4_vsm_tfidf.py
Butuh: pandas, numpy, Sastrawi, scikit-learn (opsional), dan stki_utils.py (satu folder). Data: ../data/korpus.csv, ../data/queries.csv

Isi file:
  Bagian A : langkah praktikum 1-5 (TF, IDF, matriks TF-IDF, cosine similarity, TfidfModel)
  Bagian B : jawaban Tugas Step 4 no. 1-4 (di bagian bawah)
"""
import pandas as pd
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


print("\n" + "=" * 70 + "\nStep 4 — Vector Space Model dan Pembobotan TF-IDF\n" + "=" * 70)
# # Step 4 — Vector Space Model dan Pembobotan TF-IDF
#
# **Tujuan praktikum**
# 1. Mahasiswa memahami pembobotan *term frequency* (TF), *inverse document frequency* (IDF), dan TF-IDF.
# 2. Mahasiswa mampu merepresentasikan dokumen dan query sebagai vektor.
# 3. Mahasiswa mampu menghitung *cosine similarity* dan menghasilkan daftar dokumen terurut (*ranked retrieval*).
#
# **Rumus yang digunakan**
#
# $$w_{t,d} = (1 + \log_{10} tf_{t,d}) \times \log_{10}\frac{N}{df_t}$$
#
# $$\cos(\vec q, \vec d) = \frac{\vec q \cdot \vec d}{\lVert \vec q \rVert \, \lVert \vec d \rVert}$$

# Jalankan sekali jika pustaka belum terpasang:
# !pip install pandas numpy matplotlib Sastrawi
import pandas as pd
pd.set_option("display.max_colwidth", 120)

import math
import numpy as np
from collections import Counter
from stki_utils import corpus_texts, preprocess, build_inverted_index, load_corpus

judul = dict(zip(load_corpus().doc_id, load_corpus().judul))
docs = {d: preprocess(t) for d, t in corpus_texts().items()}
index = build_inverted_index(docs)
N = len(docs)
vocab = list(index)
print(N, "dokumen,", len(vocab), "term")

print("\n" + "=" * 70 + "\nLangkah 1 — Term frequency dan variasinya\n" + "=" * 70)
# ## Langkah 1 — Term frequency dan variasinya

tf_d10 = Counter(docs["D10"])
print(pd.DataFrame({
    "tf mentah": tf_d10,
    "tf log (1+log10 tf)": {t: 1 + math.log10(v) for t, v in tf_d10.items()},
}).sort_values("tf mentah", ascending=False).head(8))

print("\n" + "=" * 70 + "\nLangkah 2 — Document frequency dan IDF\n" + "=" * 70)
# ## Langkah 2 — Document frequency dan IDF
# Term yang muncul di banyak dokumen kurang mampu membedakan dokumen sehingga diberi bobot IDF kecil.

idf = {t: math.log10(N / len(p)) for t, p in index.items()}
tabel_idf = pd.DataFrame({"df": {t: len(index[t]) for t in vocab}, "idf": idf})
print(pd.concat([tabel_idf.sort_values("df", ascending=False).head(5),
           tabel_idf.sort_values("df").head(5)]).round(3))

print("\n" + "=" * 70 + "\nLangkah 3 — Matriks TF-IDF dokumen\n" + "=" * 70)
# ## Langkah 3 — Matriks TF-IDF dokumen

def tfidf_vector(tokens):
    return {t: (1 + math.log10(tf)) * idf[t] for t, tf in Counter(tokens).items() if t in idf}

M = pd.DataFrame(0.0, index=list(docs), columns=vocab)
for d, tokens in docs.items():
    for t, w in tfidf_vector(tokens).items():
        M.loc[d, t] = w
print("Ukuran matriks TF-IDF:", M.shape)
print(M.loc[["D07", "D08", "D09", "D10"], ["tani", "air", "pupuk", "sensor", "padi"]].round(3))

print("\n" + "=" * 70 + "\nLangkah 4 — Vektor query dan cosine similarity\n" + "=" * 70)
# ## Langkah 4 — Vektor query dan cosine similarity

def cosine(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu and nv else 0.0

query = "teknologi pertanian modern"
q_tokens = preprocess(query)
q_vec = pd.Series(0.0, index=vocab)
for t, w in tfidf_vector(q_tokens).items():
    q_vec[t] = w
print("Term query:", q_tokens, "| term di luar kosakata:", [t for t in q_tokens if t not in idf])

skor = {d: cosine(q_vec.values, M.loc[d].values) for d in docs}
hasil = sorted(skor.items(), key=lambda x: (-x[1], x[0]))[:5]
print(pd.DataFrame([(d, judul[d], round(s, 4)) for d, s in hasil], columns=["doc_id", "judul", "cosine"]))

print("\n" + "=" * 70 + "\nLangkah 5 — Implementasi efisien dengan inverted index\n" + "=" * 70)
# ## Langkah 5 — Implementasi efisien dengan inverted index
# Menghitung cosine terhadap semua dokumen boros. Dengan inverted index, cukup dokumen yang memuat minimal satu term query yang diberi skor (*term-at-a-time*). Implementasi ini tersedia sebagai `TfidfModel` di `stki_utils.py`.

from stki_utils import TfidfModel

model = TfidfModel(docs)

def cari(query, k=5):
    hasil = model.search(preprocess(query), k=k)
    return pd.DataFrame([(d, judul[d], round(s, 4)) for d, s in hasil], columns=["doc_id", "judul", "skor"])

cek = model.search(q_tokens, k=5)
print("Sama dengan perhitungan matriks:", [d for d, _ in cek] == [d for d, _ in hasil])
print(cari("pemasaran digital UMKM"))

print(cari("pembelajaran daring mahasiswa"))

print("\n" + "=" * 70 + "\nTugas Step 4\n" + "=" * 70)
# ## Tugas Step 4
# 1. Hitung secara manual (di kertas atau sel Markdown) bobot TF-IDF term `tani` pada dokumen D10, lalu cocokkan dengan hasil program.
# 2. Bandingkan tiga skema TF: mentah, logaritmik, dan biner. Untuk query `harga hasil panen petani`, bagaimana urutan 5 dokumen teratas berubah?
# 3. Query `pembelajaran daring mahasiswa` mengembalikan D14 (Kecerdasan Buatan). Jelaskan penyebabnya dari sisi preprocessing dan usulkan perbaikannya.
# 4. (Opsional) Bandingkan hasil Anda dengan `TfidfVectorizer` dari scikit-learn. Parameter apa yang harus diatur agar hasilnya mendekati implementasi manual?

# =====================================================================
# BAGIAN B - JAWABAN TUGAS
# =====================================================================

# ---
# # Jawaban Tugas Step 4
# Seluruh jawaban memakai data praktikum: **30 dokumen** (D01 sampai D30, 472 term unik setelah preprocessing) dan **8 query** (Q1 sampai Q8). Penilaian relevansi `data/qrels.csv` dipakai sebagai data pendukung untuk mengukur dampak perbedaan skema TF dan usulan perbaikan (P@5, MAP, dan nDCG@5 dari `stki_utils.evaluate`).

print("\n" + "=" * 70 + "\nContoh hasil pencarian: Q5 `pemasaran digital UMKM` dengan penilaian relevansi\n" + "=" * 70)
# ## Contoh hasil pencarian: Q5 `pemasaran digital UMKM` dengan penilaian relevansi
# Pelengkap Langkah 5: lima dokumen teratas `TfidfModel` untuk Q5 dibandingkan dengan qrels.

from stki_utils import load_queries, load_qrels
queries = load_queries()
qrels = load_qrels()

hasil_q5 = model.search(preprocess(queries["Q5"]), k=5)
print(pd.DataFrame([(i, d, judul[d], round(s, 4),
               f"relevan ({qrels['Q5'][d]})" if d in qrels["Q5"] else "tidak relevan")
              for i, (d, s) in enumerate(hasil_q5, start=1)],
             columns=["peringkat", "doc_id", "judul", "skor", "menurut qrels Q5"]))

# D19 (Digitalisasi UMKM) berada di peringkat 1 dengan skor jauh di atas yang lain, dan ketiga dokumen relevan menurut qrels (D19, D22, D21) berada di lima besar. D24 dan D27 ikut muncul karena memuat sebagian term query (`digital`, `pasar`) tetapi tidak dinilai relevan.

print("\n" + "=" * 70 + "\nTugas 4.1 — Perhitungan manual bobot TF-IDF term `tani` pada D10\n" + "=" * 70)
# ## Tugas 4.1 — Perhitungan manual bobot TF-IDF term `tani` pada D10
# **Perhitungan manual** (N = 30 dokumen):
#
# *Term `tani` pada D10:* tf = 3, df = 7
#
# 1. TF logaritmik: $1 + \log_{10} 3 = 1 + 0{,}4771 = 1{,}4771$
# 2. IDF: $\log_{10}(30/7) = 0{,}6320$
# 3. Bobot: $w = 1{,}4771 \times 0{,}6320 = 0{,}9336$
#
# Sebagai pembanding dihitung term `sensor` pada D10 (tf = 2, df = 2):
#
# 1. TF logaritmik: $1 + \log_{10} 2 = 1{,}3010$
# 2. IDF: $\log_{10}(30/2) = 1{,}1761$
# 3. Bobot: $w = 1{,}3010 \times 1{,}1761 = 1{,}5301$
#
# Walaupun `tani` lebih sering muncul di D10 (3 kali dibanding 2 kali), bobotnya lebih kecil karena `tani` ada di 7 dari 30 dokumen, sedangkan `sensor` hanya di 2 dokumen. Term yang jarang di koleksi lebih mampu membedakan dokumen.

def hitung_manual(term, doc):
    tf = Counter(docs[doc])[term]
    df = len(index[term])
    tf_log = 1 + math.log10(tf)
    idf_t = math.log10(N / df)
    return {"term": term, "dokumen": doc, "tf": tf, "df": df, "N": N,
            "1+log10(tf)": round(tf_log, 4), "log10(N/df)": round(idf_t, 4),
            "w manual": round(tf_log * idf_t, 4),
            "w program (matriks M)": round(M.loc[doc, term], 4)}

print(pd.DataFrame([hitung_manual("tani", "D10"), hitung_manual("sensor", "D10")]))

# Normalisasi panjang vektor (cosine): bobot pada TfidfModel adalah bobot di atas dibagi norma vektor dokumen
vec_d10 = M.loc["D10"]
norma = np.linalg.norm(vec_d10.values)
print("Jumlah term unik D10          :", int((vec_d10 > 0).sum()))
print("Norma vektor D10 (||d||)      :", round(norma, 4))
print("w ternormalisasi (manual)     :", round(M.loc["D10", "tani"] / norma, 4))
print("w pada TfidfModel (program)   :", round(model.doc_vectors["D10"]["tani"], 4))

# **Kesimpulan 4.1.** Nilai hitungan manual (0,9336 untuk `tani` dan 1,5301 untuk `sensor` pada D10) sama dengan matriks TF-IDF program (0,934 dan 1,53 pada Langkah 3). Setelah dibagi norma vektor D10 (6,2286, dengan 26 term unik), bobot `tani` menjadi 0,9336 / 6,2286 = 0,1499, sama dengan nilai yang disimpan `TfidfModel`. Jadi implementasi program konsisten dengan rumus $w_{t,d} = (1+\log_{10} tf_{t,d}) \times \log_{10}(N/df_t)$ yang diikuti normalisasi cosine.

print("\n" + "=" * 70 + "\nTugas 4.2 — Perbandingan tiga skema TF (mentah, logaritmik, biner)\n" + "=" * 70)
# ## Tugas 4.2 — Perbandingan tiga skema TF (mentah, logaritmik, biner)
# Skema TF diterapkan pada vektor dokumen **dan** query, dengan IDF dan normalisasi cosine yang sama:
#
# | Skema | Bobot TF |
# |---|---|
# | mentah | $tf$ |
# | logaritmik | $1 + \log_{10} tf$ |
# | biner | 1 jika term ada, 0 jika tidak |
#
# Query yang diuji adalah `harga hasil panen petani` (Q7), yang setelah preprocessing menjadi `harga`, `hasil`, `panen`, `tani`.

from stki_utils import load_queries, load_qrels, evaluate
queries = load_queries()
qrels = load_qrels()

def tfw(tf, skema):
    return {"mentah": tf, "log": 1 + math.log10(tf), "biner": 1.0}[skema]

def vektor(tokens, skema):
    return {t: tfw(c, skema) * idf[t] for t, c in Counter(tokens).items() if t in idf}

def l2(v):
    n = math.sqrt(sum(x * x for x in v.values()))
    return {t: x / n for t, x in v.items()} if n else {}

SKEMA = ["mentah", "log", "biner"]
doc_vec = {s: {d: l2(vektor(tk, s)) for d, tk in docs.items()} for s in SKEMA}

def cari_skema(query, skema, k=5):
    qv = l2(vektor(preprocess(query), skema))
    skor = {}
    for d, dv in doc_vec[skema].items():
        s = sum(w * dv.get(t, 0.0) for t, w in qv.items())
        if s > 0:
            skor[d] = s
    return sorted(skor.items(), key=lambda x: (-x[1], x[0]))[:k]

# verifikasi: skema log harus sama dengan TfidfModel
for qid, q in queries.items():
    assert [d for d, _ in cari_skema(q, "log")] == [d for d, _ in model.search(preprocess(q), k=5)]
print("Skema 'log' sama dengan TfidfModel pada seluruh query: OK")

def bandingkan(query, k=5):
    print(f"Query: {query!r}  ->  {preprocess(query)}")
    print("Term di luar kosakata:", [t for t in preprocess(query) if t not in idf])
    tabel = {s: [f"{d} ({sk:.3f})" for d, sk in cari_skema(query, s, k)] for s in SKEMA}
    return pd.DataFrame(tabel, index=range(1, k + 1)).rename_axis("peringkat")

hasil_42 = bandingkan(queries["Q7"])
import os
os.makedirs("output", exist_ok=True)
hasil_42.to_csv("../output/perbandingan_skema_tf.csv", encoding="utf-8")
print(hasil_42)

# Dokumen relevan Q7 menurut qrels dan rincian kontribusi tiap term query (bobot query x bobot dokumen)
print("Relevan Q7 (qrels):", qrels["Q7"])
print("tf term query pada dokumen:")
for t in preprocess(queries["Q7"]):
    print(f"  {t:6s} df={len(index.get(t, {}))}  tf per dokumen: {dict(index.get(t, {}))}")

def rincian(query, doc, skema):
    qv = l2(vektor(preprocess(query), skema))
    return {t: round(w * doc_vec[skema][doc].get(t, 0.0), 3) for t, w in qv.items() if doc_vec[skema][doc].get(t, 0.0) > 0}

baris = []
for d in ["D12", "D07", "D20", "D10", "D23", "D18"]:
    for s in SKEMA:
        baris.append({"dokumen": d, "skema": s, "kontribusi per term": rincian(queries["Q7"], d, s),
                      "skor": round(sum(rincian(queries["Q7"], d, s).values()), 3)})
print(pd.DataFrame(baris))

# Norma vektor dokumen (penyebut cosine) pada tiap skema
dok = ["D07", "D10", "D12", "D20", "D23"]
print(pd.DataFrame({s: {d: np.linalg.norm(list(vektor(docs[d], s).values())) for d in dok} for s in SKEMA}).round(3))

# Dampak pada seluruh query: gunakan qrels sebagai data pendukung
metrik = {}
per_query_ap = {}
for s in SKEMA:
    run = {qid: [d for d, _ in cari_skema(q, s, k=30)] for qid, q in queries.items()}
    ev = evaluate(run, qrels, k=5)
    metrik[s] = ev.loc["Rata-rata", ["P@5", "AP", "nDCG@5"]].rename({"AP": "MAP"})
    per_query_ap[s] = ev["AP"].drop("Rata-rata")
print(pd.DataFrame(metrik).T)
print(pd.DataFrame(per_query_ap).T.round(3))

# **Analisis (query Q7, `harga hasil panen petani`).**
# - **Peringkat 1 dan 2 tidak berubah** pada ketiga skema: D12 (Harga Gabah Saat Panen Raya) dan D07 (Budidaya Padi Sawah). Perubahan terjadi di peringkat 3 sampai 5: skema mentah menghasilkan D20, D10, D23, sedangkan skema logaritmik dan biner menghasilkan D23, D20, D18.
# - **Pelakunya D10 (Pertanian Cerdas Berbasis IoT).** D10 hanya memuat satu term query, yaitu `tani`, tetapi muncul 3 kali. Pada skema mentah, bobot `tani` di D10 tiga kali lipat sehingga kontribusinya 0,090 dan D10 naik ke peringkat 4, melampaui D23 (Ekspor Kopi Nusantara, skor 0,082) yang memuat dua term query (`harga` dan `tani`). Pada skema logaritmik, kontribusi D10 turun menjadi 0,049 sehingga D10 keluar dari lima besar, sementara D23 naik ke peringkat 3 dengan skor 0,110.
# - **Dampak pada kualitas:** menurut qrels, D12, D07, D23, dan D20 relevan untuk Q7 sedangkan D10 tidak. Skema mentah menyisipkan D10 di antara dokumen relevan sehingga AP Q7 turun menjadi 0,950, sementara skema logaritmik dan biner menghasilkan AP 1,000.
# - **Rata-rata pada 8 query:** MAP skema mentah 0,722, logaritmik 0,762, dan biner 0,766; nDCG@5 masing-masing 0,849, 0,862, dan 0,889. Selisih terbesar ada pada Q6 (AP 0,610 untuk mentah, 0,833 untuk logaritmik, 0,867 untuk biner). Pada korpus berisi dokumen pendek ini, skema logaritmik dan biner sama-sama lebih baik daripada mentah, dan selisih antara keduanya kecil.
#
# **Kesimpulan 4.2.**
# 1. **Mentah** paling sensitif terhadap pengulangan: dokumen yang mengulang satu term query bisa mengalahkan dokumen yang memuat beberapa term query yang berbeda (kasus D10 dan D23).
# 2. **Logaritmik** meredam efek pengulangan (kemunculan ketiga bernilai 1,48 kali, bukan 3 kali) sehingga lebih seimbang; inilah skema yang dipakai pada `TfidfModel`.
# 3. **Biner** mengabaikan frekuensi sama sekali dan hanya melihat kehadiran term dan IDF. Skema ini cukup untuk dokumen pendek, tetapi akan kehilangan informasi pada dokumen panjang yang term pentingnya sering muncul.

print("\n" + "=" * 70 + "\nTugas 4.3 — Query `pembelajaran daring mahasiswa` mengembalikan D14 (Kecerdasan Buatan)\n" + "=" * 70)
# ## Tugas 4.3 — Query `pembelajaran daring mahasiswa` mengembalikan D14 (Kecerdasan Buatan)
# Query ini (Q8) menghasilkan D25 (Pembelajaran Daring di Perguruan Tinggi) di peringkat 1, tetapi peringkat 5 diisi **D14**, dokumen kecerdasan buatan. Di peringkat 2, D29 (Perpustakaan Digital) juga tidak relevan. Tabel berikut menunjukkan term query yang menyebabkan masing-masing dokumen muncul.

import re
import stki_utils as su

q43 = queries["Q8"]
isi = dict(zip(load_corpus().doc_id, load_corpus().isi))
qt = set(preprocess(q43))
print("Token query :", preprocess(q43))
print({t: f"df={len(index[t])}, idf={idf[t]:.3f}" for t in preprocess(q43)})

rows = []
for d, s in model.search(preprocess(q43), k=5):
    teks = judul[d] + " " + isi[d]
    rows.append((d, judul[d], round(s, 4),
                 sorted(t for t in qt if d in index.get(t, {})),
                 [w for w in re.findall(r"[a-z0-9]+", teks.lower()) if su.stem(w) in qt],
                 "relevan" if d in qrels["Q8"] else "tidak relevan"))
print(pd.DataFrame(rows, columns=["doc_id", "judul", "skor", "term query yang cocok", "kata asli di dokumen", "menurut qrels Q8"]))

# Semua dokumen yang memuat term `ajar` dan kata asli penyebabnya
baris_ajar = []
for d in sorted(index["ajar"]):
    teks_d = judul[d] + " " + isi[d]
    baris_ajar.append((d, judul[d], [w for w in re.findall(r"[a-z0-9]+", teks_d.lower()) if su.stem(w) == "ajar"]))
print(pd.DataFrame(baris_ajar, columns=["doc_id", "judul", "kata asli yang di-stem menjadi 'ajar'"]))

def bangun(prep):
    return TfidfModel({d: prep(t) for d, t in corpus_texts().items()})

def tok0(teks):
    return su.remove_stopwords(su.tokenize(su.case_folding(teks)))

# Usulan: kamus pengecualian stemming dan kolokasi (frasa dijadikan satu token)
PROT = {"belajar", "pelajar", "mempelajari", "dipelajari", "pelajaran"}
KOLOKASI = {("pembelajaran", "mesin"): "pembelajaran_mesin"}

def gabung(toks):
    out, i = [], 0
    while i < len(toks):
        if i + 1 < len(toks) and (toks[i], toks[i + 1]) in KOLOKASI:
            out.append(KOLOKASI[(toks[i], toks[i + 1])]); i += 2
        else:
            out.append(toks[i]); i += 1
    return out

def stem_kecuali(toks, prot):
    return [w if (w in prot or "_" in w) else su.stem(w) for w in toks]

# (fungsi preprocessing, syarat minimal jumlah term query berbeda yang cocok)
VARIAN = {
    "A. Dasar (stemming Sastrawi)":                       (preprocess, 1),
    "B. Tanpa stemming":                                  (lambda t: preprocess(t, use_stemming=False), 1),
    "C. Stemming + pengecualian (termasuk pembelajaran)": (lambda t: stem_kecuali(tok0(t), PROT | {"pembelajaran"}), 1),
    "D. Dasar + minimal 2 term cocok":                    (preprocess, 2),
    "E. Stemming + pengecualian + kolokasi":              (lambda t: stem_kecuali(gabung(tok0(t)), PROT), 1),
}

def jalankan(prep, min_term=1):
    m_ = bangun(prep)
    hasil = {}
    for qid, qtext in queries.items():
        toks = prep(qtext)
        ts = set(toks)
        hasil[qid] = [d for d, _ in m_.search(toks, k=30)
                      if sum(d in m_.index.get(t, {}) for t in ts) >= min_term]
    return hasil

dasar = jalankan(preprocess)
baris = []
for nama, (prep, mt) in VARIAN.items():
    h = jalankan(prep, mt)
    ev = evaluate(h, qrels, k=5)
    baris.append({"Varian": nama, "Top-5 Q8": h["Q8"][:5], "AP Q8": ev.loc["Q8", "AP"],
                  "MAP (8 query)": ev.loc["Rata-rata", "AP"], "nDCG@5 (8 query)": ev.loc["Rata-rata", "nDCG@5"],
                  "Query yang top-5-nya berubah": [q for q in queries if h[q][:5] != dasar[q][:5]]})
pd.set_option("display.max_colwidth", 80)
print(pd.DataFrame(baris))

# **Penyebab (sisi preprocessing).**
# 1. **Stemming meleburkan kata dengan makna berbeda.** Sastrawi mengubah `pembelajaran` menjadi `ajar`, sama dengan `belajar` (D14: "komputer belajar dari data") dan `pelajar` (D27). Dengan demikian D14 cocok dengan query lewat satu term, yaitu `ajar` (muncul dua kali di D14: `belajar` dan `pembelajaran`). Contoh yang sama bahkan ditulis di D17: "kata pembelajaran menjadi ajar".
# 2. **Model bag-of-words tidak membedakan konteks.** D14 memuat kata `pembelajaran` secara harfiah, tetapi dalam frasa "pembelajaran mesin" (machine learning), bukan "pembelajaran daring". Setelah tokenisasi, kedua konteks itu menjadi token yang sama sehingga stemming bukan satu-satunya penyebab.
# 3. **Tidak ada syarat kecocokan minimum.** Cosine similarity memberi skor pada setiap dokumen yang memuat minimal satu term query. D29, D28, D26, dan D14 masing-masing hanya cocok pada satu dari tiga term (`daring`, `mahasiswa`, atau `ajar`), tetapi tetap tampil karena korpus kecil. Skor D14 (0,0587) hampir sama dengan D26 (0,0595) yang relevan, sehingga urutan keduanya rapuh. Tokenisasi dan stopword bukan penyebab karena `pembelajaran`, `daring`, dan `mahasiswa` bukan stopword.
#
# **Usulan perbaikan dan hasilnya (lihat tabel).**
#
# | Varian | Top-5 Q8 | AP Q8 | MAP | nDCG@5 | Catatan |
# |---|---|---|---|---|---|
# | A. Dasar | D25, D29, D28, D26, D14 | 0,806 | 0,762 | 0,862 | Kondisi awal. |
# | B. Tanpa stemming | D25, D29, D28, D26, D14 | 0,806 | 0,768 | 0,892 | D14 tetap muncul (kata `pembelajaran` cocok secara harfiah). Mengubah top-5 pada 5 dari 8 query (Q1, Q2, Q5, Q6, Q7). |
# | C. Stemming + pengecualian | D25, D29, D28, D26, D14 | 0,806 | 0,762 | 0,862 | Tidak mengubah hasil: D14 tetap cocok lewat `pembelajaran`. |
# | D. Dasar + minimal 2 term cocok | D25 | 0,333 | 0,507 | 0,737 | Semua salah ambil hilang, tetapi dokumen relevan D28 dan D26 ikut terbuang. |
# | E. Pengecualian + kolokasi `pembelajaran mesin` | D25, D29, D26, D28, D17 | 0,806 | 0,776 | 0,871 | D14 hilang dari Q8. Hanya Q4 dan Q8 yang berubah. |
#
# **Pembacaan hasil.**
# - **E** adalah perbaikan preprocessing yang paling tepat sasaran: `belajar` dan `pelajar` tidak lagi dilebur dengan `pembelajaran`, dan "pembelajaran mesin" menjadi satu token sehingga D14 tidak lagi cocok dengan query pendidikan. MAP naik dari 0,762 menjadi 0,776 karena pada Q4 D29 (relevan) kini berada di depan D14. Namun pada Q8 posisi D14 digantikan D17, dokumen tentang stemming yang secara harfiah menulis kata `pembelajaran`, sehingga masih ada satu dokumen salah ambil.
# - **D** menghilangkan seluruh salah ambil tetapi merusak recall (AP Q8 turun dari 0,806 menjadi 0,333; MAP dari 0,762 menjadi 0,507), jadi tidak direkomendasikan sebagai aturan umum.
# - **B** dan **C** tidak menyelesaikan masalah pada query ini, karena penyebab utamanya adalah kata `pembelajaran` yang bermakna ganda.
# - **Rekomendasi:** gunakan E (kamus pengecualian dan kolokasi domain) sebagai perbaikan preprocessing. Pembedaan makna yang menyeluruh membutuhkan informasi konteks, misalnya indeks frasa seperti pada Step 3 (`"pembelajaran daring"` hanya mengembalikan D25, tetapi D28 dan D26 yang masih relevan ikut terlewat) atau model berbasis makna kata pada Step 9 dan 10 (Word2Vec dan BERT).

print("\n" + "=" * 70 + "\nTugas 4.4 (opsional) — Perbandingan dengan `TfidfVectorizer` scikit-learn\n" + "=" * 70)
# ## Tugas 4.4 (opsional) — Perbandingan dengan `TfidfVectorizer` scikit-learn
# **Parameter agar mendekati implementasi manual:**
#
# | Parameter | Nilai | Alasan |
# |---|---|---|
# | `analyzer` | `preprocess` | Memakai tokenisasi, stopword, dan stemming yang sama. |
# | `sublinear_tf` | `True` | TF menjadi $1 + \log(tf)$, bukan tf mentah. |
# | `smooth_idf` | `False` | IDF tanpa penghalusan: $\ln(N/df)+1$, bukan $\ln\frac{1+N}{1+df}+1$. |
# | `use_idf` | `True` | Memakai IDF. |
# | `norm` | `'l2'` | Normalisasi cosine (bawaan). |
#
# **Perbedaan yang tidak bisa dihilangkan lewat parameter:** (1) sklearn memakai logaritma natural (ln), sedangkan implementasi manual memakai $\log_{10}$; (2) rumus IDF sklearn selalu mengandung **+1**, sedangkan implementasi manual $\log_{10}(N/df)$ tidak. Kedua hal ini membuat skor dan kadang urutan sedikit berbeda.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

ids = list(corpus_texts())
teks = [corpus_texts()[d] for d in ids]

def top5_sklearn(vec, k=5):
    X = vec.fit_transform(teks)
    hasil = {}
    for qid, qt_ in queries.items():
        s = (X @ vec.transform([qt_]).T).toarray().ravel()
        urut = sorted(range(len(ids)), key=lambda i: (-s[i], ids[i]))
        hasil[qid] = [ids[i] for i in urut[:k] if s[i] > 0]
    return hasil

acuan = {qid: [d for d, _ in model.search(preprocess(qt_), k=5)] for qid, qt_ in queries.items()}
konfigurasi = {
    "bawaan":                                dict(),
    "sublinear_tf=True":                     dict(sublinear_tf=True),
    "sublinear_tf=True + smooth_idf=False":  dict(sublinear_tf=True, smooth_idf=False),
}
baris = []
for nama, cfg in konfigurasi.items():
    hasil = top5_sklearn(TfidfVectorizer(analyzer=preprocess, **cfg))
    identik = sum(hasil[q] == acuan[q] for q in queries)
    irisan = sum(len(set(hasil[q]) & set(acuan[q])) / max(len(acuan[q]), 1) for q in queries) / len(queries)
    baris.append({"Konfigurasi sklearn": nama,
                  "Urutan top-5 identik (dari 8 query)": identik,
                  "Rata-rata irisan top-5": round(irisan, 3),
                  "Query yang berbeda": [q for q in queries if hasil[q] != acuan[q]]})
tabel_sk = pd.DataFrame(baris)
tabel_sk.to_csv("output/perbandingan_sklearn.csv", index=False, encoding="utf-8")
print(tabel_sk)

# Replikasi eksak: ambil hitungan term dari sklearn, tetapi TF dan IDF memakai rumus manual (log10)
cv = TfidfVectorizer(analyzer=preprocess, use_idf=False, norm=None)
X = cv.fit_transform(teks).tocsr().astype(float)
df_vec = np.asarray((X > 0).sum(axis=0)).ravel()
idf_vec = np.log10(len(ids) / df_vec)
X.data = 1 + np.log10(X.data)
X = normalize(X @ sp.diags(idf_vec))

def top5_replikasi(qt_, k=5):
    Q = cv.transform([qt_]).tocsr().astype(float)
    Q.data = 1 + np.log10(Q.data)
    Q = normalize(Q @ sp.diags(idf_vec))
    s = (X @ Q.T).toarray().ravel()
    urut = sorted(range(len(ids)), key=lambda i: (-s[i], ids[i]))
    return [ids[i] for i in urut[:k] if s[i] > 1e-12]

identik = sum(top5_replikasi(qt_) == acuan[qid] for qid, qt_ in queries.items())
print(f"Replikasi eksak: urutan top-5 identik pada {identik} dari {len(queries)} query")

# **Kesimpulan 4.4.** Dengan `analyzer=preprocess`, `sublinear_tf=True`, `smooth_idf=False`, dan `norm='l2'`, hasil `TfidfVectorizer` sudah **mendekati** implementasi manual (rata-rata irisan top-5 0,95), tetapi urutan top-5 hanya identik pada 3 dari 8 query (Q2, Q5, Q6, Q7, dan Q8 berbeda) akibat perbedaan basis logaritma dan konstanta +1 pada IDF. Hasil **identik pada seluruh query** baru tercapai bila hitungan term diambil dari sklearn sementara TF dan IDF dihitung dengan rumus manual. Ini menegaskan bahwa perbedaan skor berasal dari rumus pembobotan, bukan dari kesalahan implementasi.

print("\n" + "=" * 70 + "\nRingkasan Modul 4\n" + "=" * 70)
# ## Ringkasan Modul 4
# - Bobot TF-IDF dihitung dengan $w_{t,d} = (1+\log_{10} tf) \times \log_{10}(N/df)$; hitungan manual `tani` pada D10 (0,9336) sama dengan hasil program, dan setelah normalisasi cosine menjadi 0,1499.
# - Skema TF mentah memberi keuntungan pada dokumen yang mengulang satu term (D10 pada Q7); MAP mentah 0,722 dibanding 0,762 (logaritmik) dan 0,766 (biner).
# - Query `pembelajaran daring mahasiswa` memunculkan D14 karena `pembelajaran`, `belajar`, dan `pelajar` dilebur menjadi `ajar`, kata `pembelajaran` bermakna ganda pada model bag-of-words, dan tidak ada syarat kecocokan minimum. Kamus pengecualian dengan kolokasi menghilangkan D14 dan menaikkan MAP menjadi 0,776, sedangkan syarat minimal 2 term merusak recall.
# - `TfidfVectorizer` mendekati implementasi manual dengan `sublinear_tf=True` dan `smooth_idf=False`, tetapi tidak identik karena logaritma natural dan IDF yang selalu ditambah 1.

