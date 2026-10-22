"""
Modul 2 - Indexing: Incidence Matrix dan Inverted Index
Tahapan: incidence matrix, inverted index, penyimpanan index, positional index
dan phrase query.
Korpus: data/korpus.csv (30 dokumen), diproses dengan preprocess() dari stki_utils.
"""
import pandas as pd

# Pustaka yang dibutuhkan: pandas, numpy, matplotlib, Sastrawi
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)


# ---------------------------------------------------------------------------
# Persiapan - memuat dokumen dan mengubahnya menjadi token
# ---------------------------------------------------------------------------
from collections import Counter, defaultdict
from stki_utils import corpus_texts, preprocess

teks = corpus_texts()                      # {doc_id: judul + isi}
docs = {d: preprocess(t) for d, t in teks.items()}
print("\n[Persiapan] Memuat dokumen")
print("Jumlah dokumen:", len(docs))
print("D07 setelah preprocess:", docs["D07"])


# ---------------------------------------------------------------------------
# Langkah 1 - Term-document incidence matrix
# Matriks biner |V| x N: bernilai 1 jika term muncul di dokumen
# ---------------------------------------------------------------------------
subset = ["D07", "D08", "D09", "D10", "D11", "D12"]
vocab_subset = sorted({t for d in subset for t in docs[d]})
incidence = pd.DataFrame([[int(t in docs[d]) for d in subset] for t in vocab_subset],
                         index=vocab_subset, columns=subset)
print("\n[Langkah 1] Term-document incidence matrix")
print("Ukuran matriks:", incidence.shape)
print(incidence.loc[["air", "hama", "padi", "pupuk", "sawah", "tani", "tanah"]].to_string())

# Kepadatan matriks pada seluruh korpus
vocab = sorted({t for tokens in docs.values() for t in tokens})
sel_total = len(vocab) * len(docs)
sel_isi = sum(len(set(tokens)) for tokens in docs.values())
print(f"\nKosakata: {len(vocab)} term, {len(docs)} dokumen")
print(f"Sel matriks penuh : {sel_total}")
print(f"Sel bernilai 1    : {sel_isi} ({sel_isi / sel_total:.1%})")


# ---------------------------------------------------------------------------
# Langkah 2 - Membangun inverted index
# Struktur: index[term] = {doc_id: tf}; panjang posting list = df
# ---------------------------------------------------------------------------
def build_inverted_index(docs_tokens):
    index = defaultdict(dict)
    for doc_id, tokens in docs_tokens.items():
        for term, tf in Counter(tokens).items():
            index[term][doc_id] = tf
    return dict(sorted(index.items()))


print("\n[Langkah 2] Inverted index")
index = build_inverted_index(docs)
print("Jumlah term di kamus:", len(index))
for term in ["tani", "digital", "gizi", "cari"]:
    print(f"{term:8s} df={len(index[term]):2d}  postings={index[term]}")

# Tabel df dan cf (total tf) untuk term paling sering muncul di dokumen
tabel = pd.DataFrame({
    "df": {t: len(p) for t, p in index.items()},
    "cf (total tf)": {t: sum(p.values()) for t, p in index.items()},
}).sort_values("df", ascending=False)
print("\n10 term dengan df tertinggi:")
print(tabel.head(10).to_string())


# ---------------------------------------------------------------------------
# Langkah 3 - Menyimpan dan memuat index
# ---------------------------------------------------------------------------
print("\n[Langkah 3] Menyimpan dan memuat index")
import json, os

os.makedirs("output", exist_ok=True)
with open("output/inverted_index.json", "w") as f:
    json.dump(index, f)

with open("output/inverted_index.json") as f:
    index_dimuat = json.load(f)
print("Index tersimpan dan dimuat ulang dengan benar:", index_dimuat == index)
print("Ukuran file:", os.path.getsize("output/inverted_index.json"), "byte")


# ---------------------------------------------------------------------------
# Langkah 4 - Positional index dan phrase query
# Posisi kemunculan term: index[term][doc_id] = [posisi, ...]
# ---------------------------------------------------------------------------
def build_positional_index(docs_tokens):
    index = defaultdict(lambda: defaultdict(list))
    for doc_id, tokens in docs_tokens.items():
        for pos, term in enumerate(tokens):
            index[term][doc_id].append(pos)
    return {t: dict(p) for t, p in sorted(index.items())}


print("\n[Langkah 4] Positional index dan phrase query")
pos_index = build_positional_index(docs)
print("cerdas :", pos_index["cerdas"])
print("buat   :", pos_index["buat"])


def phrase_search(frasa, pos_index):
    terms = preprocess(frasa)
    if not terms or any(t not in pos_index for t in terms):
        return []
    kandidat = set.intersection(*(set(pos_index[t]) for t in terms))
    hasil = []
    for d in sorted(kandidat):
        for p in pos_index[terms[0]][d]:
            if all(p + i in pos_index[t][d] for i, t in enumerate(terms[1:], start=1)):
                hasil.append(d)
                break
    return hasil


for frasa in ["kecerdasan buatan", "pertanian cerdas", "pupuk organik", "data pribadi"]:
    print(f"{frasa!r:22s} -> {phrase_search(frasa, pos_index)}")


# ===========================================================================
# TUGAS STEP 2
# ===========================================================================

# ---------------------------------------------------------------------------
# Tugas 1 - Perbandingan ukuran incidence matrix dan inverted index
# Matriks menyimpan semua sel |V| x N. Inverted index hanya menyimpan
# pasangan term-dokumen yang bernilai tf > 0 (posting) dan daftar kamus.
# ---------------------------------------------------------------------------
entri_matriks = len(index) * len(docs)
entri_posting = sum(len(p) for p in index.values())
entri_kamus = len(index)
print("\n[Tugas 1] Perbandingan ukuran")
print("Entri matriks penuh :", entri_matriks)
print("Entri posting       :", entri_posting)
print("Entri kamus         :", entri_kamus)
print(f"Rasio (posting saja): {entri_matriks / entri_posting:.2f} kali")
print(f"Rasio (+ kamus)     : {entri_matriks / (entri_posting + entri_kamus):.2f} kali")


# ---------------------------------------------------------------------------
# Tugas 2 - Term dengan df = 1 (hapax)
# ---------------------------------------------------------------------------
hapax = [t for t, p in index.items() if len(p) == 1]
print("\n[Tugas 2] Hapax")
print("Jumlah hapax:", len(hapax))
for t in hapax[:10]:
    d, tf = next(iter(index[t].items()))
    print(f"{t:15s} dokumen={d} tf={tf}")


# ---------------------------------------------------------------------------
# Tugas 3 - Posisi dihitung sebelum stopword dihapus
# Stopword tetap menempati posisi agar jarak antarkata sesuai teks asli
# ---------------------------------------------------------------------------
print("\n[Tugas 3] Posisi sebelum stopword dihapus")
docs_raw = {d: preprocess(t, use_stopword=False) for d, t in teks.items()}
pos_index_raw = build_positional_index(docs_raw)


def phrase_search_raw(frasa, pos_index):
    terms = preprocess(frasa, use_stopword=False)
    if not terms or any(t not in pos_index for t in terms):
        return []
    kandidat = set.intersection(*(set(pos_index[t]) for t in terms))
    hasil = []
    for d in sorted(kandidat):
        for p in pos_index[terms[0]][d]:
            if all(p + i in pos_index[t][d] for i, t in enumerate(terms[1:], start=1)):
                hasil.append(d)
                break
    return hasil


for frasa in ["sistem temu kembali informasi", "gerakan 3M", "air pupuk"]:
    print(f"{frasa!r:34s} setelah stopword : {phrase_search(frasa, pos_index)}")
    print(f"{frasa!r:34s} sebelum stopword : {phrase_search_raw(frasa, pos_index_raw)}")


# ---------------------------------------------------------------------------
# Tugas 4 - Proximity search
# Mengembalikan dokumen yang memuat t1 dan t2 dengan jarak posisi terdekat
# tidak lebih dari k. Jika t1 dan t2 sama, posisi yang sama tidak dihitung.
# ---------------------------------------------------------------------------
def proximity_search(t1, t2, k, pos_index):
    a = preprocess(t1, use_stopword=False)
    b = preprocess(t2, use_stopword=False)
    if not a or not b or a[0] not in pos_index or b[0] not in pos_index:
        return []
    a, b = a[0], b[0]
    kandidat = set(pos_index[a]) & set(pos_index[b])
    hasil = []
    for d in sorted(kandidat):
        jarak = min((abs(p1 - p2) for p1 in pos_index[a][d] for p2 in pos_index[b][d]
                     if not (a == b and p1 == p2)), default=None)
        if jarak is not None and jarak <= k:
            hasil.append(d)
    return hasil


print("\n[Tugas 4] Proximity search")
print("cerdas, buat, k=3   ->", proximity_search("cerdas", "buat", 3, pos_index_raw))
print("cerdas, buat, k=1   ->", proximity_search("cerdas", "buat", 1, pos_index_raw))
print("padi, hama, k=5     ->", proximity_search("padi", "hama", 5, pos_index_raw))


# ===========================================================================
# JAWABAN TUGAS STEP 2
# ===========================================================================
#
# Tugas 1 - Perbandingan ukuran
#   Matriks penuh memiliki 472 term x 30 dokumen = 14.160 sel. Inverted index
#   menyimpan 702 posting, yaitu jumlah sel bernilai 1 pada matriks (dihitung
#   pada Langkah 1). Rasio posting terhadap sel matriks sekitar 20,2 kali.
#   Jika 472 entri kamus ikut dihitung, total entri index menjadi 1.174 dan
#   rasionya sekitar 12,1 kali. Inverted index lebih hemat karena sel bernilai
#   0 (95% dari matriks) tidak disimpan. Pada koleksi yang lebih besar, rasio
#   ini semakin besar karena matriks tumbuh dengan |V| x N, sedangkan posting
#   tumbuh sesuai jumlah kemunculan term.
#
# Tugas 2 - Hapax
#   Hapax adalah term yang hanya muncul di satu dokumen (df = 1). Pada korpus ini
#   terdapat 346 hapax, dan 10 contohnya dicetak oleh kode di atas (150, 2, 3m,
#   acu, aedes, aegypti, ahli, aju, akibat, aktif).
#   Hapax berguna untuk pencarian ketika kueri memuatnya: term yang hanya ada
#   di sedikit dokumen memiliki nilai IDF tinggi, sehingga kecocokan padanya
#   sangat membedakan dokumen. Namun hapax tidak membantu pada kueri umum yang
#   term-nya sudah tersebar di banyak dokumen. Sebagian hapax juga dapat berupa
#   salah ketik, bentuk stemming yang tidak sempurna, atau kata yang kurang
#   penting, sehingga tidak semuanya layak dipertahankan. Contoh yang jelas
#   adalah angka "150" (D03), angka "2" (D04, dari "tipe 2") serta kode "3m"
#   (D01, dari gerakan 3M). Token seperti ini tidak diubah oleh
#   stemming, sehingga tetap menjadi hapax, dan hanya relevan untuk kueri yang
#   memuat angka atau kode tersebut. Kesimpulannya, hapax berguna untuk kueri
#   yang spesifik, tetapi perlu dinilai satu per satu.
#
# Tugas 3 - Posisi sebelum atau sesudah stopword
#   Pada index yang dibangun dari teks setelah stopword dihapus, kata yang
#   dipisahkan oleh stopword menjadi berurutan. Contohnya D10 memuat "air dan
#   pupuk". Setelah "dan" dihapus, index mencatat "air" dan "pupuk" pada posisi
#   berurutan, sehingga phrase query "air pupuk" cocok dengan D10 padahal di teks
#   asli kedua kata itu tidak berurutan. Kode di atas mencetak hasil untuk
#   "air pupuk" dengan kedua versi index; versi sebelum stopword dihapus tidak
#   mengembalikan D10.
#   Dua frasa pada tugas ini ("sistem temu kembali informasi" dan "gerakan 3M")
#   tidak terpengaruh di korpus ini, karena pada teks aslinya kedua frasa sudah
#   berurutan tanpa stopword di antaranya (D29 dan D01). Hasil kedua versi index
#   sama untuk keduanya. Versi kode yang dimodifikasi mempertahankan posisi asli
#   dengan memanggil preprocess(use_stopword=False), sehingga selisih posisi
#   sesuai urutan kata dalam teks.
#
# Tugas 4 - Proximity search
#   Fungsi proximity_search mengambil selisih posisi terkecil antara term t1
#   dan term t2 di setiap dokumen, lalu membandingkannya dengan k. Dokumen
#   dikembalikan jika selisih itu tidak lebih dari k. Fungsi ini menerima
#   pos_index sebagai argumen keempat agar dapat dipakai dengan index mana pun.
#   Pada pengujian: "cerdas" dan "buat" dengan k=3 maupun k=1 sama-sama
#   menghasilkan ['D14'], karena di D14 kedua kata berjarak 1 posisi.
#   "padi" dan "hama" dengan k=5 tidak menghasilkan dokumen, karena kedua kata
#   tidak berdekatan di dokumen mana pun dalam jarak tersebut.