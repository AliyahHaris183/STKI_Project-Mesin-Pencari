"""
Step 3 - Model Boolean (STKI)
Jalankan dari folder notebooks/ :  python step3_boolean.py
Butuh: pandas, numpy, Sastrawi, scikit-learn (opsional), dan stki_utils.py (satu folder). Data: ../data/korpus.csv, ../data/queries.csv

Isi file:
  Bagian A : langkah praktikum 1-5 (intersect, union/and_not, optimasi urutan term,
             parser Boolean, keterbatasan model Boolean)
  Bagian B : jawaban Tugas Step 3 no. 1-4 (di bagian bawah)
"""
import pandas as pd
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


print("\n" + "=" * 70 + "\nStep 3 — Model Boolean\n" + "=" * 70)
# # Step 3 — Model Boolean
#
# **Tujuan praktikum**
# 1. Mahasiswa memahami model Boolean dan operator AND, OR, NOT.
# 2. Mahasiswa mampu mengimplementasikan algoritma *merge* (intersection) dua posting list terurut.
# 3. Mahasiswa mampu mengoptimasi urutan pemrosesan query berdasarkan *document frequency*.
# 4. Mahasiswa memahami keterbatasan model Boolean (tidak ada peringkat).

# Jalankan sekali jika pustaka belum terpasang:
# !pip install pandas numpy matplotlib Sastrawi
import pandas as pd
pd.set_option("display.max_colwidth", 120)

from stki_utils import corpus_texts, preprocess, build_inverted_index

docs = {d: preprocess(t) for d, t in corpus_texts().items()}
index = build_inverted_index(docs)
SEMUA_DOC = sorted(docs)

def postings(term):
    t = preprocess(term, use_stopword=False)
    return sorted(index.get(t[0], {})) if t else []

print("tani   :", postings("tani"))
print("air    :", postings("air"))

print("\n" + "=" * 70 + "\nLangkah 1 — Algoritma intersection (AND) dengan dua pointer\n" + "=" * 70)
# ## Langkah 1 — Algoritma intersection (AND) dengan dua pointer
# Karena posting list terurut, irisan dapat dihitung dalam waktu O(x + y) dengan menggeser dua pointer.

def intersect(p1, p2):
    hasil, i, j, perbandingan = [], 0, 0, 0
    while i < len(p1) and j < len(p2):
        perbandingan += 1
        if p1[i] == p2[j]:
            hasil.append(p1[i]); i += 1; j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return hasil, perbandingan

hasil, n = intersect(postings("petani"), postings("air"))
print("petani AND air =", hasil, f"({n} perbandingan)")

print("\n" + "=" * 70 + "\nLangkah 2 — Union (OR) dan NOT\n" + "=" * 70)
# ## Langkah 2 — Union (OR) dan NOT

def union(p1, p2):
    hasil, i, j = [], 0, 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            hasil.append(p1[i]); i += 1; j += 1
        elif p1[i] < p2[j]:
            hasil.append(p1[i]); i += 1
        else:
            hasil.append(p2[j]); j += 1
    return hasil + p1[i:] + p2[j:]

def and_not(p1, p2):
    s2 = set(p2)
    return [d for d in p1 if d not in s2]

print("pupuk OR kompos     =", union(postings("pupuk"), postings("kompos")))
print("tani AND NOT hama   =", and_not(postings("tani"), postings("hama")))

print("\n" + "=" * 70 + "\nLangkah 3 — Optimasi query: proses term dengan df terkecil lebih dulu\n" + "=" * 70)
# ## Langkah 3 — Optimasi query: proses term dengan df terkecil lebih dulu
# Untuk query `t1 AND t2 AND ... AND tn`, hasil antara tidak mungkin lebih besar dari posting list terpendek. Mengurutkan term berdasarkan df menaikkan efisiensi.

def and_query(terms, optimasi=True):
    lists = [postings(t) for t in terms]
    if optimasi:
        lists.sort(key=len)
    hasil, total = lists[0], 0
    for p in lists[1:]:
        hasil, n = intersect(hasil, p)
        total += n
    return hasil, total

query = ["data", "aplikasi", "sensor"]
print("df:", {t: len(postings(t)) for t in query})
print("Tanpa optimasi :", and_query(query, optimasi=False))
print("Dengan optimasi:", and_query(query, optimasi=True))

print("\n" + "=" * 70 + "\nLangkah 4 — Parser query Boolean lengkap\n" + "=" * 70)
# ## Langkah 4 — Parser query Boolean lengkap
# `stki_utils.boolean_search` mem-parsing query dengan operator `AND`, `OR`, `NOT` (huruf kapital) dan tanda kurung, dengan prioritas NOT > AND > OR. Pelajari kodenya di `stki_utils.py`.

from stki_utils import boolean_search

contoh_query = [
    "petani AND pupuk",
    "pupuk OR kompos",
    "tani AND NOT hama",
    "(digital OR daring) AND internet",
    "gizi AND (anak OR balita) AND NOT sekolah",
]
for q in contoh_query:
    print(f"{q:45s} -> {boolean_search(q, index, SEMUA_DOC)}")

print("\n" + "=" * 70 + "\nLangkah 5 — Keterbatasan model Boolean\n" + "=" * 70)
# ## Langkah 5 — Keterbatasan model Boolean
# Perhatikan dua query berikut. Query AND terlalu ketat (sering kosong), sedangkan query OR terlalu longgar dan semua hasil dianggap sama relevannya karena tidak ada peringkat.

q = "keamanan data pribadi di internet"
terms = preprocess(q)
print("Term query:", terms)
print("AND :", boolean_search(" AND ".join(terms), index, SEMUA_DOC))
print("OR  :", boolean_search(" OR ".join(terms), index, SEMUA_DOC))

print("\n" + "=" * 70 + "\nTugas Step 3\n" + "=" * 70)
# ## Tugas Step 3
# 1. Implementasikan fungsi `or_not(p1, p2)` untuk `t1 OR NOT t2` menggunakan daftar semua dokumen. Mengapa operasi ini mahal pada koleksi besar?
# 2. Buat 5 query Boolean untuk mencari dokumen topik **ekonomi** tanpa mengambil dokumen topik lain. Laporkan hasil tiap query dan jelaskan strategi Anda.
# 3. Tambahkan dukungan *phrase query* (dalam tanda kutip) ke parser Boolean dengan memanfaatkan positional index dari Step 2.
# 4. Diskusikan: mengapa model Boolean masih digunakan pada pencarian dokumen hukum atau paten?

# =====================================================================
# BAGIAN B - JAWABAN TUGAS
# =====================================================================

# ---
# # Jawaban Tugas Step 3
# Seluruh jawaban memakai data praktikum: **30 dokumen** berbahasa Indonesia (`data/korpus.csv`, D01 sampai D30) dan **8 query** (`data/queries.csv`, Q1 sampai Q8).

from stki_utils import build_positional_index, load_corpus, load_queries
korpus = load_corpus()
judul = dict(zip(korpus.doc_id, korpus.judul))
print(len(korpus), "dokumen,", len(load_queries()), "query")

print("\n" + "=" * 70 + "\nTugas 3.1 — Fungsi `or_not(p1, p2)` untuk `t1 OR NOT t2`\n" + "=" * 70)
# ## Tugas 3.1 — Fungsi `or_not(p1, p2)` untuk `t1 OR NOT t2`
# `t1 OR NOT t2` = gabungan (union) posting list `t1` dengan **komplemen** posting list `t2` terhadap seluruh dokumen. Komplemen itu harus dibangun dari daftar semua dokumen (`SEMUA_DOC`).

def complement(p, semua=SEMUA_DOC):
    """Komplemen posting list terhadap seluruh dokumen (operasi NOT)."""
    s = set(p)
    return [d for d in semua if d not in s]

def or_not(p1, p2, semua=SEMUA_DOC):
    """t1 OR NOT t2 = union(p1, komplemen(p2))."""
    return union(p1, complement(p2, semua))

t1, t2 = "pupuk", "padi"
p1, p2 = postings(t1), postings(t2)
hasil = or_not(p1, p2)

print(f"df({t1}) = {len(p1)}, df({t2}) = {len(p2)}, N = {len(SEMUA_DOC)}")
print(f"{t1} OR NOT {t2}  -> {len(hasil)} dokumen dari {len(SEMUA_DOC)}")
print("Dokumen yang tidak ikut :", sorted(set(SEMUA_DOC) - set(hasil)))
print(f"{t1} AND NOT {t2} -> {and_not(p1, p2)}")

# verifikasi terhadap parser Boolean bawaan
assert hasil == boolean_search(f"{t1} OR NOT {t2}", index, SEMUA_DOC)
print("Hasil or_not() sama dengan boolean_search(): OK")

N = len(SEMUA_DOC)
biaya = pd.DataFrame({
    "t1 AND NOT t2": {"elemen yang diproses": len(p1) + len(p2),
                      "ukuran hasil": len(and_not(p1, p2))},
    "t1 OR NOT t2":  {"elemen yang diproses": N + len(p1) + (N - len(p2)),
                      "ukuran hasil": len(hasil)},
})
print(biaya)

# Tabel di atas menghitung jumlah elemen yang harus ditelusuri atau dibentuk dan ukuran hasil kedua operasi pada korpus ini.
#
# **Penjelasan.** Model Boolean hanya menyimpan daftar dokumen yang *memuat* sebuah term, bukan yang *tidak* memuatnya. Karena itu `NOT t2` harus dibangun dengan menelusuri seluruh daftar dokumen dan membuang anggota posting list `t2`. Pada contoh di atas `padi` hanya muncul di 1 dari 30 dokumen, sehingga komplemennya berisi 29 dokumen dan hasil `pupuk OR NOT padi` berisi 29 dokumen, hampir seluruh koleksi.
#
# **Mengapa mahal pada koleksi besar?**
# 1. **Biaya sebanding dengan N, bukan df.** Posting list term yang jarang sangat pendek, tetapi komplemennya hampir sebesar seluruh koleksi. Satu operasi NOT harus menelusuri dan menghasilkan sekitar N docID. Pada koleksi 100 juta dokumen, itu berarti daftar ratusan juta ID di memori.
# 2. **Hasil antara membengkak.** Operasi lanjutan (union, intersect) harus memproses daftar sebesar itu, dan optimasi "proses term dengan df terkecil lebih dulu" (Langkah 3) tidak berlaku: makin jarang `t2`, makin besar komplemennya.
# 3. **Hasilnya tidak selektif.** Hampir semua dokumen lolos, sehingga hasil kurang berguna bagi pengguna.
# 4. **Bandingkan dengan `AND NOT`.** `t1 AND NOT t2` cukup memfilter `p1` terhadap `p2`, dengan biaya O(|p1| + |p2|), dan hasilnya tidak mungkin lebih besar dari `p1`. Itulah sebabnya mesin pencari nyata umumnya hanya mengizinkan `NOT` bila ada minimal satu term positif.

print("\n" + "=" * 70 + "\nTugas 3.2 — Lima query Boolean untuk topik **ekonomi**\n" + "=" * 70)
# ## Tugas 3.2 — Lima query Boolean untuk topik **ekonomi**
# **Rumusan topik.** Korpus tersusun per topik (kesehatan D01-D06, pertanian D07-D12, teknologi D13-D18, ekonomi D19-D24, pendidikan D25-D30). Enam dokumen D19 sampai D24 dijadikan acuan topik ekonomi untuk menghitung presisi dan recall setiap query.

EKONOMI = ["D19", "D20", "D21", "D22", "D23", "D24"]
print(pd.DataFrame({"doc_id": EKONOMI, "judul": [judul[d] for d in EKONOMI]}))

K = {
 "K1": "inflasi OR kredit OR investasi OR saham OR ekspor OR umkm",
 "K2": "inflasi OR kredit OR investasi OR ekspor OR umkm OR qris",
 "K3": "harga OR bayar OR usaha OR bank OR pasar OR modal OR bisnis",
 "K4": "(harga OR bayar OR usaha OR bank OR pasar OR modal OR bisnis) AND NOT (panen OR komputasi OR data)",
 "K5": "bank AND (bunga OR kredit)",
}
rel = set(EKONOMI)
baris = []
for kid, q in K.items():
    r = set(boolean_search(q, index, SEMUA_DOC))
    tp = len(r & rel)
    baris.append({"Query": kid, "Ekspresi Boolean": q, "Jumlah hasil": len(r),
                  "Relevan (TP)": tp,
                  "Presisi": round(tp / len(r), 3) if r else 0.0,
                  "Recall": round(tp / len(rel), 3),
                  "Salah ambil (di luar topik)": sorted(r - rel)})
pd.set_option("display.max_colwidth", 90)
tabel_ekonomi = pd.DataFrame(baris)
import os
os.makedirs("output", exist_ok=True)
tabel_ekonomi.to_csv("../output/query_ekonomi.csv", index=False, encoding="utf-8")
print(tabel_ekonomi)

for kid, q in K.items():
    r = boolean_search(q, index, SEMUA_DOC)
    print(f"{kid}: {q}")
    for d in r:
        print(f"    {d}  {judul[d]}{'' if d in rel else '   <-- di luar topik ekonomi'}")
    print()

# **Strategi dan analisis.**
#
# | Query | Strategi | Hasil |
# |---|---|---|
# | K1 | **OR atas istilah khas ekonomi** (`inflasi`, `kredit`, `investasi`, `saham`, `ekspor`, `umkm`). Istilah ini hanya muncul di dokumen ekonomi, jadi tidak mengambil topik lain. | Presisi 1,00; recall 0,833 (5 dari 6). D21 (QRIS) terlewat karena memakai kosakata pembayaran. |
# | K2 | **Perluas OR** dengan istilah khas dari D21: `qris`; `saham` dilepas karena `investasi` sudah mewakili D24. | Presisi 1,00; recall 1,00 (6 dari 6). |
# | K3 | **OR atas istilah umum** (`harga`, `bayar`, `usaha`, `bank`, `pasar`, `modal`, `bisnis`). Istilah umum lebih mudah ditebak pengguna, tetapi tidak eksklusif. | Recall 1,00 tetapi presisi turun menjadi 0,667: D12 (Harga Gabah, `harga`), D16 (Komputasi Awan untuk Bisnis, `bayar`, `usaha`, `bisnis`), dan D18 (Analisis Big Data, `bisnis`) ikut terambil. |
# | K4 | **K3 + `AND NOT (panen OR komputasi OR data)`** untuk membuang dokumen salah ambil pada K3. | Presisi 1,00; recall 1,00. |
# | K5 | **AND** (`bank AND (bunga OR kredit)`) untuk mempersempit. | Presisi 1,00 tetapi recall hanya 0,333 (2 dari 6: D20 dan D22); terlalu ketat. |
#
# **Pelajaran.**
# - Untuk mendapat topik tertentu tanpa topik lain, pilih istilah yang eksklusif pada topik itu (K1, K2). Operator `OR` mendongkrak recall, sedangkan `AND` dan `NOT` mendongkrak presisi.
# - Istilah umum (K3) menimbulkan salah ambil lintas topik, misalnya `harga` yang juga muncul pada topik pertanian. `NOT` dapat membersihkannya (K4), tetapi hanya setelah penyusun query memeriksa hasil secara manual dan mengetahui dokumen pengganggunya.
# - Boolean bersifat semua-atau-tidak-sama-sekali: sebuah dokumen lolos hanya jika memenuhi ekspresi logis secara persis, sehingga K5 melewatkan sebagian besar dokumen ekonomi.
# - **Catatan jujur:** K2 dan K4 mencapai presisi dan recall 1,00 karena disusun dengan melihat kosakata korpus yang kecil (30 dokumen) dan dinilai dengan acuan topik per blok dokumen. Pada koleksi yang lebih besar, hasil sebaik ini jauh lebih sulit dicapai.

print("\n" + "=" * 70 + "\nTugas 3.3 — Dukungan *phrase query* pada parser Boolean\n" + "=" * 70)
# ## Tugas 3.3 — Dukungan *phrase query* pada parser Boolean
# **Rancangan.**
# 1. **Indeks posisi dibangun dari token tanpa penghapusan stopword** (`preprocess(..., use_stopword=False)`). Jika stopword dibuang lebih dulu (seperti pada indeks posisi Step 2), dua kata yang sebenarnya dipisahkan stopword menjadi tampak berdampingan dan menghasilkan frasa palsu. Contohnya ada di bawah.
# 2. **Frasa dinormalisasi dengan preprocessing yang sama** (case folding dan stemming), sehingga `"pembayaran digital"` cocok dengan variasi bentuk katanya.
# 3. **Algoritma pencocokan:** (a) ambil irisan dokumen yang memuat semua kata frasa, (b) di tiap dokumen, periksa apakah ada posisi awal *p* sehingga kata ke-*i* berada di posisi *p + i*.
# 4. **Integrasi ke parser:** frasa (diapit tanda kutip) diperlakukan sebagai satu operan seperti term biasa, sehingga bisa dikombinasikan dengan `AND`, `OR`, `NOT`, dan tanda kurung.

import re

# Indeks posisi tanpa membuang stopword agar adjacency kata pada frasa akurat
tokens_penuh = {d: preprocess(t, use_stopword=False) for d, t in corpus_texts().items()}
pos_index = build_positional_index(tokens_penuh)

def phrase_postings(frasa, pos_index=pos_index):
    """Dokumen yang memuat seluruh kata `frasa` secara berurutan (berdampingan)."""
    terms = preprocess(frasa, use_stopword=False)
    if not terms or any(t not in pos_index for t in terms):
        return []
    kandidat = set(pos_index[terms[0]])
    for t in terms[1:]:
        kandidat &= set(pos_index[t])
    hasil = []
    for d in sorted(kandidat):
        mulai = set(pos_index[terms[0]][d])                 # kemungkinan posisi awal frasa
        for i, t in enumerate(terms[1:], start=1):
            mulai &= {p - i for p in pos_index[t][d]}       # kata ke-i harus di posisi awal + i
        if mulai:
            hasil.append(d)
    return hasil

def boolean_search_phrase(query, pos_index, all_docs):
    """Parser Boolean (AND, OR, NOT, kurung) dengan dukungan frasa dalam tanda kutip."""
    if query.count('"') % 2:
        raise ValueError("Tanda kutip tidak seimbang")
    all_docs = set(all_docs)
    tokens = re.findall(r'"[^"]+"|\(|\)|[^\s()"]+', query)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def parse_or():
        result = parse_and()
        while peek() == "OR":
            take()
            result = result | parse_and()
        return result

    def parse_and():
        result = parse_not()
        while peek() == "AND":
            take()
            result = result & parse_not()
        return result

    def parse_not():
        if peek() == "NOT":
            take()
            return all_docs - parse_not()
        if peek() == "(":
            take()
            result = parse_or()
            if take() != ")":
                raise ValueError("Kurung tidak seimbang")
            return result
        tok = take()
        if tok.startswith('"'):                              # operan frasa
            return set(phrase_postings(tok[1:-1], pos_index))
        term = preprocess(tok, use_stopword=False)           # operan term biasa
        return set(pos_index.get(term[0], {})) if term else set()

    result = parse_or()
    if pos != len(tokens):
        raise ValueError(f"Query tidak valid di dekat '{tokens[pos]}'")
    return sorted(result)

uji = [
    '"gizi anak"',                             # frasa
    'gizi AND anak',                           # bandingkan: kata terpisah
    '"lahan petani"',
    'lahan AND petani',
    '"pembelajaran daring"',                   # dipakai lagi pada Step 4 (Tugas 4.3)
    '"daring pembelajaran"',                   # urutan kata penting
    '("kredit usaha rakyat" OR "pasar modal") AND NOT umkm',
    '"media sosial" AND NOT digital',          # frasa + NOT
    '"stunting balita"',                       # ada kata "pada" di antara keduanya
    '"stunting pada balita"',
]
for q in uji:
    print(f"{q:52s} -> {boolean_search_phrase(q, pos_index, SEMUA_DOC)}")

# Mengapa stopword tidak boleh dibuang dari indeks posisi? Contoh: "stunting pada balita" pada D02
tokens_tanpa_stop = {d: preprocess(t) for d, t in corpus_texts().items()}
pos_index_stop = build_positional_index(tokens_tanpa_stop)

print("Judul asli D02         :", judul["D02"])
print('"stunting balita" (stopword dibuang) ->', phrase_postings("stunting balita", pos_index_stop), " <- frasa palsu")
print('"stunting balita" (stopword dijaga)  ->', phrase_postings("stunting balita", pos_index))

# **Penjelasan hasil.**
# - `"gizi anak"` hanya cocok dengan **D30** (Gizi Anak Sekolah), sedangkan `gizi AND anak` juga mengambil **D02** (stunting pada balita) yang memuat kedua kata tetapi tidak berdampingan. Hal serupa terjadi pada `"lahan petani"` (hanya D08) dibandingkan `lahan AND petani` (D08, D09). Jadi frasa menaikkan **presisi** tanpa perlu menambah operator `NOT`.
# - Urutan kata penting: `"pembelajaran daring"` cocok dengan D25, tetapi `"daring pembelajaran"` kosong. Frasa dapat dikombinasikan dengan operator lain, misalnya `("kredit usaha rakyat" OR "pasar modal") AND NOT umkm` menghasilkan D24 saja karena D22 dibuang oleh `NOT umkm`.
# - `"pembelajaran daring"` hanya mengembalikan D25. Hasil ini akan dibandingkan dengan pencarian berperingkat pada Step 4 (Tugas 4.3), yang mengembalikan dokumen tidak relevan seperti D14. Frasa Boolean presisi tinggi, tetapi dokumen yang masih relevan seperti D28 (beasiswa untuk mahasiswa) dan D26 (Kurikulum Merdeka) tidak ikut terambil sehingga recall rendah.
# - Contoh `"stunting balita"` membuktikan perlunya indeks posisi yang menjaga stopword: judul D02 berbunyi "Pencegahan Stunting **pada** Balita". Pada indeks tanpa stopword, kedua kata terbaca berdampingan dan D02 muncul sebagai frasa palsu; pada indeks yang menjaga stopword hasilnya kosong, sedangkan `"stunting pada balita"` menghasilkan D02.
# - **Keterbatasan:** indeks posisi lebih besar dari indeks biasa (menyimpan setiap posisi kemunculan), dan frasa dicocokkan pada bentuk *stem*, sehingga frasa dengan stem sama tetapi makna berbeda tetap bisa cocok.

print("\n" + "=" * 70 + "\nTugas 3.4 — Mengapa model Boolean masih dipakai pada pencarian dokumen hukum atau paten?\n" + "=" * 70)
# ## Tugas 3.4 — Mengapa model Boolean masih dipakai pada pencarian dokumen hukum atau paten?
# Model Boolean memang tidak memberi peringkat, tetapi sifat itulah yang justru dibutuhkan pada dua domain tersebut:
#
# 1. **Kontrol dan presisi yang eksplisit.** Pencari profesional (peneliti hukum, pemeriksa paten) tahu persis istilah, pasal, atau klaim yang dicari. Operator AND, OR, NOT, tanda kurung, frasa, dan operator kedekatan (*proximity*) memungkinkan mereka menyatakan kebutuhan informasi secara tepat. Pada model peringkat, pengguna hanya menyerahkan kata kunci dan tidak dapat menjamin kondisi logis tertentu terpenuhi.
# 2. **Kebutuhan recall yang menyeluruh.** Pada pencarian *prior art* paten atau penelusuran preseden hukum, satu dokumen yang terlewat bisa berakibat serius (paten ditolak atau dibatalkan, perkara dikalahkan). Pengguna bersedia memeriksa seluruh himpunan hasil, sehingga urutan menjadi kurang penting dibanding kelengkapan.
# 3. **Hasil yang dapat dijelaskan, diulang, dan diaudit.** Sebuah dokumen muncul karena memenuhi ekspresi logis tertentu, dan ekspresi yang sama selalu memberi himpunan hasil yang sama. Ini penting bagi pertanggungjawaban profesional: string pencarian dapat dicatat, dibagikan, dan dibuktikan kelengkapannya. Skor TF-IDF atau BM25 bergantung pada statistik korpus sehingga sulit dijelaskan dengan cara yang sama.
# 4. **Terminologi baku dan pencocokan persis.** Dokumen hukum dan paten penuh istilah baku (nomor pasal, nomor klaim, nama badan hukum, kelas paten) yang harus dicocokkan secara eksak, bukan lewat kemiripan statistik.
#
# **Kelemahan yang tetap ada** (terlihat pada Langkah 5): hasil bisa kosong atau terlalu banyak, penyusunan query butuh keahlian, dan tanpa peringkat pengguna harus menyaring sendiri. Karena itu sistem modern sering memakai pendekatan **hibrida**: Boolean untuk menyaring himpunan kandidat yang pasti memenuhi syarat, lalu peringkat (TF-IDF/BM25) untuk mengurutkannya.

print("\n" + "=" * 70 + "\nRingkasan Modul 3\n" + "=" * 70)
# ## Ringkasan Modul 3
# - Posting list terurut memungkinkan AND (intersect), OR (union), dan AND NOT dihitung dalam waktu O(x + y); mengurutkan term menurut df (term jarang lebih dulu) mengurangi jumlah perbandingan (5 vs 8 pada query `data AND aplikasi AND sensor`).
# - `OR NOT` mahal karena komplemen posting list hampir sebesar seluruh koleksi; `AND NOT` jauh lebih murah.
# - Query topik ekonomi paling baik disusun dari istilah khas yang digabung `OR`; `NOT` dan `AND` dipakai untuk menaikkan presisi dengan konsekuensi recall turun.
# - Dukungan frasa memerlukan indeks posisi yang menjaga stopword, dan menaikkan presisi dibanding `AND` biasa (`"gizi anak"` vs `gizi AND anak`).
# - Boolean tidak berperingkat, tetapi tetap relevan pada domain yang membutuhkan kontrol, recall menyeluruh, dan hasil yang dapat diaudit.

