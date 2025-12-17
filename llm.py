# assistant_keuangan_improved.py
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from typing import Dict, Any
import dateparser
import pymysql
from decimal import Decimal
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import json 
from rapidfuzz import fuzz, process
import gradio as gr
import pandas as pd
from typing import Optional
from typing import Tuple, Optional

# ==================== KONFIGURASI ====================
from db_connection import get_connection

def some_function():
    conn = get_connection()
    cur = conn.cursor()

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "data_finance"
QDRANT_DICT = "dict_user"   # collection kamus semantik
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
QDRANT_INTENT_COLLECTION = "data_intent"

# Semantic thresholds (adjustable)
SEMANTIC_SCORE_THRESHOLD = 0.50   # untuk semantic_lookup (dict_user)
LEXICAL_BOOST_THRESHOLD = 85     # untuk lexical boost
FUZZY_LOCAL_THRESHOLD = 80       # fallback fuzzy threshold untuk kamus lokal

# Intent confidence threshold (vector-first)
INTENT_CONFIDENCE_THRESHOLD = 0.7

# Inisialisasi model & qdrant client (pakai qdrant_client konsisten)
model = SentenceTransformer(EMBED_MODEL_NAME)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ==================== EMBEDDING & INTENT DETECTION ====================

def embed_text(text: str):
    """
    Ubah teks menjadi embedding vektor (list float) agar bisa dikirim ke Qdrant.
    """
    if not text or not isinstance(text, str):
        return [0.0] * model.get_sentence_embedding_dimension()
    return model.encode(text).tolist()

# ==================== INTENT CLASSIFIER (REVISI) ====================

# Asumsikan Anda telah mendefinisikan: QDRANT_INTENT_COLLECTION dan SEMANTIC_SCORE_THRESHOLD
# serta menginisialisasi model dan qdrant_client

# ==================== INTENT CLASSIFIER (REVISI FINAL) ====================

def detect_intent(text: str) -> str:
    """
    Mengklasifikasikan intent pengguna berdasarkan semantic search di Qdrant.
    """
    # GANTI: Menghilangkan panggilan correct_typo untuk menghindari error TypeError
    t = (text or "").lower().strip() 
    
    try:
        # Encode teks pengguna menggunakan helper
        q_vec = embed_text(t) 
        
        # Cari di koleksi intent (data_intent)
        hits = qdrant_client.search(
            collection_name=QDRANT_INTENT_COLLECTION,
            query_vector=q_vec,
            limit=1,
            with_payload=True 
        )
    except Exception as e:
        print(f"❌ Error Qdrant Search Intent: {e}")
        return "unknown"

    if hits:
        best = hits[0]
        score = getattr(best, "score", 0.0)
        
        # 1. Pastikan skor di atas ambang batas (High Confidence)
        if score >= SEMANTIC_SCORE_THRESHOLD:
            
            # 2. Ambil intent_name dari payload (KUNCI KRITIS!)
            intent_name = best.payload.get("intent_name", "unknown_from_payload")
            
            print(f"[DEBUG] Qdrant Intent Hit: Score={score:.3f} -> Intent='{intent_name}'")
            return intent_name 
            
        else:
            print(f"[DEBUG] Qdrant hit found, but score {score:.3f} below threshold ({SEMANTIC_SCORE_THRESHOLD})")
            
    return "unknown"
# ==================== MUAT KANDIDAT LOKAL DARI QDRANT ====================

_local_keyword_list = None
_local_dict_map = {} 

def _load_local_keyword_list() -> list:
    """Ambil semua keyword dan sinonim dari tabel dict_user di MySQL."""
    global _local_keyword_list, _local_dict_map
    if _local_keyword_list is not None:
        return _local_keyword_list

    conn = get_connection()
    if not conn:
        print("⚠️ Gagal konek ke DB, menggunakan fallback local keywords.")
        fallback = ["bbm", "listrik", "gaji", "makan", "utang", "investasi", "transport"]
        _local_keyword_list = fallback
        return _local_keyword_list

    cur = conn.cursor()
    try:
        # Ambil keyword utama dan kolom sinonim (JSON)
        sql = "SELECT keyword, jenis_akun, sub_kategori, sinonim FROM dict_user;"
        cur.execute(sql)
        rows = cur.fetchall()
        
        terms = set()
        for row in rows:
            # 1. Tambahkan keyword utama
            keyword = (row['keyword'] or "").lower()
            if keyword:
                terms.add(keyword)
                _local_dict_map[keyword] = {
                    "jenis_akun": row['jenis_akun'], 
                    "sub_kategori": row['sub_kategori']
                }

            # 2. Tambahkan semua sinonim
            try:
                # Asumsi kolom sinonim di MySQL adalah string JSON array
                sinonim_list = json.loads(row['sinonim'])
                for s in sinonim_list:
                    s = (s or "").lower()
                    if s:
                        terms.add(s)
                        _local_dict_map[s] = {
                            "jenis_akun": row['jenis_akun'], 
                            "sub_kategori": row['sub_kategori']
                        }
            except (TypeError, json.JSONDecodeError):
                pass 
        
        _local_keyword_list = sorted(list(terms))
        
        if _local_keyword_list:
            print(f"[DEBUG] Muat {len(_local_keyword_list)} keyword/sinonim dari MySQL untuk fallback lokal.")
        
        cur.close()
        conn.close()
        return _local_keyword_list

    except Exception as e:
        print("⚠️ Gagal load keyword list dari MySQL:", e)
        # ... (handling fallback)
        return []

# ==================== SEMANTIC & TYPO HANDLER (REVISI) ====================

# ... (Pastikan fungsi correct_typo sudah ada)

# ==================== SEMANTIC LOOKUP (REVISI FINAL) ====================

# Pastikan Anda sudah mengimpor fuzzywuzzy.fuzz dan fuzzywuzzy.process
# dan mendefinisikan konstanta FUZZY_LOCAL_THRESHOLD (misalnya 85)
# dan SEMANTIC_SCORE_THRESHOLD (misalnya 0.50)

def semantic_lookup(text: str) -> Dict[str, Optional[str]]:
    """
    Cari kecocokan semantik/lexical dari Qdrant dict_user atau map lokal.
    Prioritas: Exact Match Lokal -> Fuzzy Match Lokal -> Semantic Search Qdrant.
    """
    t = (text or "").lower().strip()
    if not t:
        return {"jenis_akun": None, "sub_kategori": None}

    # 1️⃣ CEK LOCAL KEYWORDS (Exact & Fuzzy Match dari MySQL Map)
    # Asumsi: _load_local_keyword_list() sudah dipanggil di __main__
    local_terms = _load_local_keyword_list()
    
    # --- Exact Match ---
    if t in _local_dict_map:
        result = _local_dict_map[t]
        print(f"[DEBUG] Exact match local: '{t}' -> {result['sub_kategori']} ({result['jenis_akun']})")
        return result

    # --- Fuzzy/Partial Match ---
    # Asumsi fuzzywuzzy sudah diimpor: from fuzzywuzzy import fuzz, process
    fuzzy_match = process.extractOne(t, local_terms, scorer=fuzz.partial_ratio)
    if fuzzy_match:
        best_term, score, _ = fuzzy_match
        
        # Ganti angka 85 dengan variabel FUZZY_LOCAL_THRESHOLD Anda
        if score >= 85: # Gunakan ambang batas lokal Anda (atau konstanta FUZZY_LOCAL_THRESHOLD)
            result = _local_dict_map.get(best_term)
            if result:
                 print(f"[DEBUG] Fuzzy local accept: '{t}' matches '{best_term}' ({score}) -> {result['sub_kategori']} ({result['jenis_akun']})")
                 return result

    # 2️⃣ Lanjut ke Qdrant semantic search
    try:
        # PERUBAHAN KRITIS: Menggunakan embed_text() helper untuk konsistensi
        q_vec = embed_text(t) 
        
        hits = qdrant_client.search(
            collection_name=QDRANT_DICT, # Pastikan ini nama koleksi 'dict_user' Anda
            query_vector=q_vec,
            limit=6,
            with_payload=True 
        )
    except Exception as e:
        print("⚠️ Gagal mencari di Qdrant dict_user:", e)
        hits = []

    if hits:
        best = hits[0]
        best_score = getattr(best, "score", 0) or 0.0
        
        # Pengecekan Skor Qdrant
        if best_score >= 0.50: # Gunakan SEMANTIC_SCORE_THRESHOLD
            pl = best.payload or {}
            # Mencari kunci dengan huruf kecil ('jenis_akun') atau kapital ('Jenis_Akun') untuk keamanan
            jenis_akun = pl.get("jenis_akun") or pl.get("Jenis_Akun")
            sub_kategori = pl.get("sub_kategori") or pl.get("Sub_Kategori")
            
            if jenis_akun and sub_kategori:
                 print(f"[DEBUG] Qdrant semantic accept: score={best_score:.3f} -> {sub_kategori} ({jenis_akun})")
                 return {"jenis_akun": jenis_akun, "sub_kategori": sub_kategori}
            
    print(f"[DEBUG] No semantic match for: '{text}'")
    return {"jenis_akun": None, "sub_kategori": None}
# ==================== DATABASE HELPER ====================

def get_latest_year_from_db() -> str:
    conn = get_connection()
    if not conn:
        return str(datetime.now().year)
    cur = conn.cursor()
    cur.execute("SELECT MAX(LEFT(Periode, 4)) AS latest_year FROM data_transaksi")
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row["latest_year"] if row and row["latest_year"] else str(datetime.now().year)

# ==================== SEMANTIC & TYPO HANDLER ====================
# ==================== SEMANTIC & TYPO HANDLER (REVISI) ====================

def correct_typo(input_text: str, candidate_list: list[str], threshold: int = FUZZY_LOCAL_THRESHOLD) -> str:
    """Perbaiki typo berdasarkan kemiripan string (rapidfuzz)."""
    if not candidate_list:
        return input_text
    best = process.extractOne(input_text, candidate_list, scorer=fuzz.partial_ratio)
    if not best:
        return input_text
    best_match, score, _ = best
    return best_match if score >= threshold else input_text

def semantic_lookup(text: str) -> Dict[str, Optional[str]]:
    """
    Cari kecocokan semantik/lexical dari Qdrant dict_user atau map lokal.
    Prioritas: Exact Match Lokal -> Fuzzy Match Lokal -> Semantic Search Qdrant.
    """
    t = (text or "").lower().strip()
    if not t:
        return {"jenis_akun": None, "sub_kategori": None}

    # 🟢 1️⃣ CEK LOCAL KEYWORDS (Exact & Fuzzy Match dari MySQL Map)
    local_terms = _load_local_keyword_list()
    
    # --- Exact Match ---
    if t in _local_dict_map:
        result = _local_dict_map[t]
        print(f"[DEBUG] Exact match local: '{t}' -> {result['sub_kategori']} ({result['jenis_akun']})")
        return result

    # --- Fuzzy/Partial Match ---
    fuzzy_match = process.extractOne(t, local_terms, scorer=fuzz.partial_ratio)
    if fuzzy_match:
        best_term, score, _ = fuzzy_match
        if score >= FUZZY_LOCAL_THRESHOLD:
            result = _local_dict_map.get(best_term)
            if result:
                 print(f"[DEBUG] Fuzzy local accept: '{t}' matches '{best_term}' ({score}) -> {result['sub_kategori']} ({result['jenis_akun']})")
                 return result

    # 🟡 2️⃣ Lanjut ke Qdrant semantic search
    try:
        q_vec = model.encode(t).tolist()
        hits = qdrant_client.search(
            collection_name=QDRANT_DICT,
            query_vector=q_vec,
            limit=6,
            with_payload=True # Asumsi payload HANYA berisi jenis_akun & sub_kategori
        )
    except Exception as e:
        print("⚠️ Gagal mencari di Qdrant:", e)
        hits = []

    if hits:
        best = hits[0]
        best_score = getattr(best, "score", 0) or 0.0
        
        # 💡 Terima jika skor semantik tinggi
        if best_score >= SEMANTIC_SCORE_THRESHOLD:
            pl = best.payload or {}
            jenis_akun = pl.get("jenis_akun") or pl.get("Jenis_Akun")
            sub_kategori = pl.get("sub_kategori") or pl.get("Sub_Kategori")
            
            if jenis_akun and sub_kategori:
                 print(f"[DEBUG] Qdrant semantic accept: score={best_score:.3f} -> {sub_kategori} ({jenis_akun})")
                 return {"jenis_akun": jenis_akun, "sub_kategori": sub_kategori}
            
    print(f"[DEBUG] No semantic match for: '{text}'")
    return {"jenis_akun": None, "sub_kategori": None}

# ==================== VALIDASI & PARSING INPUT ====================

def parse_amount(text: str) -> int:
    t = text.lower().replace(".", "").replace(",", "").replace("rp", "").strip()
    if m := re.search(r"(\d+)\s*juta", t): return int(m.group(1)) * 1_000_000
    if m := re.search(r"(\d+)\s*(ribu|rb|k)", t): return int(m.group(1)) * 1_000
    if nums := re.findall(r"\d+", t): return int(nums[-1])
    return 0

def extract_description(text: str) -> str:
    t = text.lower()
    bulan = ["januari","februari","maret","april","mei","juni","juli","agustus",
             "september","oktober","november","desember"]
    for b in bulan:
        t = re.sub(rf"\b{b}\b", "", t)
    t = re.sub(r"(catat|tolong|saya|aku|mohon)\s*", "", t)
    t = re.sub(r"\b\d[\d\.,]*\b", "", t)
    t = re.sub(r"(ribu|rb|juta|jt|rp)", "", t)
    t = re.sub(r"^[,.;:\s]+", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    return re.sub(r"\s+", " ", t).strip().capitalize()

MONTH_MAP = {
    'januari': 1, 'februari': 2, 'maret': 3, 'april': 4, 'mei': 5, 'juni': 6,
    'juli': 7, 'agustus': 8, 'september': 9, 'oktober': 10, 'november': 11, 'desember': 12}

def extract_date_from_text(text):
    """
    Ekstrak tanggal dari teks bahasa alami seperti:
    '01 oktober 2023' → 2023-10-01
    """
    text = text.lower().strip()
    # format 01/10/2023 atau 1-10-2023
    match = re.search(r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})', text)
    if match:
        d, m, y = match.groups()
        return datetime(int(y), int(m), int(d)).date()
    # format 01 oktober 2023 (bahasa indonesia)
    match = re.search(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})', text)
    if match:
        d, month_name, y = match.groups()
        m = MONTH_MAP.get(month_name.lower())
        if m:
            return datetime(int(y), m, int(d)).date()
    return datetime.now().date()

# ==================== VALIDATOR & PARSER TRANSAKSI (REVISI FINAL) ====================

# ==================== VALIDATOR & PARSER TRANSAKSI (REVISI FINAL) ====================

def validate_and_parse_catat(user_text: str) -> Tuple[bool, str, dict]:
    amt = parse_amount(user_text) # Asumsi: Anda memiliki fungsi ini
    if amt <= 0:
        return False, "Nominal transaksi tidak terdeteksi.", None
        
    desc = extract_description(user_text) # Asumsi: Anda memiliki fungsi ini
    if len(desc) < 3:
        return False, "Deskripsi transaksi kurang jelas.", None

    # 🚨 PERUBAHAN KRITIS: Menggunakan retrieve_context untuk mendapatkan tanggal
    period_type, date_val = retrieve_context(user_text)
    
    # Kita hanya menggunakan date_val jika context-nya adalah "daily"
    # Karena retrieve_context sudah difallback ke "daily" jika tidak ada period_type lain
    tanggal = date_val

    print(f"[DEBUG] Parsed date: {tanggal}")
    
    return True, "", {
        "Deskripsi": desc, 
        "amount": amt, 
        "Tanggal": tanggal # YYYY-MM-DD
    }
    
# ==================== DETEKSI AKUN (VECTOR-FIRST, FALLBACK RULE) ====================

def detect_jenis_akun(text: str) -> str:
    """
    Deteksi jenis akun (Aset/Beban/Pendapatan/Kewajiban).
    Prioritas: Qdrant dict_user -> fallback rule-based.
    """
    t = (text or "").lower().strip()
    try:
        vec = model.encode(t).tolist()
        results = qdrant_client.search(collection_name=QDRANT_DICT, query_vector=vec, limit=1, with_payload=True)
        if isinstance(results, tuple):
            results = results[0]
        if results and len(results) > 0:
            top = results[0]
            score = getattr(top, "score", 0.0) or 0.0
            payload = top.payload or {}
            jenis = payload.get("jenis_akun") or payload.get("Jenis_Akun")
            if score >= 0.7 and jenis:
                print(f"[DEBUG] Qdrant detect_jenis_akun -> {jenis} (score={score:.3f})")
                return jenis
    except Exception as e:
        print("⚠️ Qdrant detect_jenis_akun error:", e)

    # fallback rule-based (lama)
    if any(k in t for k in ["aset", "tanah", "peralatan", "saham", "komputer", "laptop"]): return "Aset"
    if any(k in t for k in ["gaji", "makan", "transport", "bbm", "listrik", "belanja", "futsal"]): return "Beban"
    if any(k in t for k in ["jual", "pendapatan", "bonus", "bunga", "jasa"]): return "Pendapatan"
    if any(k in t for k in ["utang", "kewajiban", "pajak", "bpjs"]): return "Kewajiban"
    return "Beban"

def detect_sub_kategori(text: str) -> str:
    """
    Deteksi sub-kategori menggunakan dict_user (Qdrant) dulu, lalu fallback rule-based.
    """
    t = (text or "").lower().strip()
    try:
        vec = model.encode(t).tolist()
        results = qdrant_client.search(collection_name=QDRANT_DICT, query_vector=vec, limit=1, with_payload=True)
        if isinstance(results, tuple):
            results = results[0]
        if results and len(results) > 0:
            top = results[0]
            score = getattr(top, "score", 0.0) or 0.0
            payload = top.payload or {}
            sub = payload.get("sub_kategori") or payload.get("Sub_Kategori") or payload.get("subkategori")
            if score >= 0.7 and sub:
                print(f"[DEBUG] Qdrant detect_sub_kategori -> {sub} (score={score:.3f})")
                return sub
    except Exception as e:
        print("⚠️ Qdrant detect_sub_kategori error:", e)

    # fallback rule-based (lama)
    if "kas" in t: return "Kas"
    if "tanah" in t or "lahan" in t: return "Tanah"
    if "saham" in t or "investasi" in t: return "Saham"
    if any(k in t for k in ["komputer", "laptop", "mesin", "alat"]): return "Peralatan"
    if any(k in t for k in ["gaji", "tunjangan", "bonus"]): return "Gaji"
    if any(k in t for k in ["transport", "ojek", "taksi", "bbm", "bensin", "gojek", "grab"]): return "Transportasi"
    if any(k in t for k in ["makan", "minum", "ngopi", "snack", "jajan"]): return "Makan/Minum"
    if "listrik" in t or "pln" in t or "token" in t: return "Listrik"
    if any(k in t for k in ["pendapatan", "penjualan", "hasil", "komisi", "bonus", "honor"]): return "Penjualan"
    if "bunga" in t: return "Bunga Bank"
    if "jasa" in t: return "Jasa"
    if any(k in t for k in ["utang dagang", "supplier"]): return "Utang Dagang"
    if "utang gaji" in t or "tunggakan" in t: return "Utang Gaji"
    return "Lainnya"

# ==================== QDRANT TRANSAKSI (UPsert/Search) ====================

def upsert_transaction_qdrant(data: dict):
    try:
        vec = model.encode(data["Deskripsi"]).tolist()
        payload = {
            "deskripsi": data["Deskripsi"],
            "debit": data.get("Debit", 0),
            "kredit": data.get("Kredit", 0),
            "jenis_akun": data.get("Jenis_Akun", detect_jenis_akun(data["Deskripsi"])),
            "sub_kategori": data.get("Sub_Kategori", detect_sub_kategori(data["Deskripsi"])),
            "tanggal": datetime.now().strftime("%Y-%m-%d")
        }
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[qm.PointStruct(id=int(datetime.now().timestamp()), vector=vec, payload=payload)]
        )
        print("🧠 Sinkron ke Qdrant berhasil.")
    except Exception as e:
        print("⚠️ Gagal upsert ke Qdrant:", e)

def search_transactions(user_query: str, k: int = 5):
    try:
        q_vec = model.encode(user_query).tolist()
        hits = qdrant_client.search(collection_name=QDRANT_COLLECTION, query_vector=q_vec, limit=k, with_payload=True)
        return [h.payload for h in hits]
    except Exception as e:
        print("⚠️ Gagal mencari di Qdrant:", e)
        return []

# ==================== MYSQL SAVE ====================

training_examples = [
    "apakah pengeluaran bulan ini lebih banyak dari bulan sebelumnya",
    "bandingkan pengeluaran bulan september dengan bulan agustus",
    "apakah saya lebih boros bulan ini",
    "bulan apa pengeluaran saya tertinggi",]

def _to_float(val: Any) -> float:
    """Konversi aman ke float. Menerima None, Decimal, int, float, str."""
    if val is None:
        return 0.0
    if isinstance(val, float):
        return val
    if isinstance(val, Decimal):
        return float(val)
    try:
        return float(val)
    except Exception:
        return 0.0

def evaluate_spending_behavior() -> str:
    """
    Menilai apakah pengeluaran bulan ini (jenis_akun='Beban') lebih besar dari rata-rata 3 bulan sebelumnya.
    Mengembalikan string feedback.
    """
    conn = get_connection()
    if not conn:
        return "❌ Tidak bisa mengakses database untuk evaluasi pengeluaran."

    cur = conn.cursor()
    try:
        # Ambil total pengeluaran per bulan, urut dari bulan terbaru
        cur.execute("""
            SELECT
                DATE_FORMAT(Tanggal, '%Y-%m') AS bulan,
                COALESCE(SUM(Debit), 0) AS total_debit
            FROM data_transaksi
            WHERE Jenis_Akun = 'Beban'
            GROUP BY bulan
            ORDER BY bulan DESC
            LIMIT 4;
        """)
        rows = cur.fetchall()

    except Exception as e:
        print("⚠ Error saat query evaluate_spending_behavior:", e)
        cur.close()
        conn.close()
        return "Gagal mengambil data untuk evaluasi pengeluaran."

    cur.close()
    conn.close()

    if not rows or len(rows) < 2:
        return "⚠ Data transaksi beban kurang untuk menilai pola pengeluaran."

    # Konversi aman ke float
    def _to_float(x):
        try:
            return float(x) if x is not None else 0.0
        except:
            return 0.0

    # Hasil fetchall dari DictCursor → akses pakai key
    amounts = [_to_float(r["total_debit"]) for r in rows]

    # Urutan: index 0 = bulan terbaru
    this_month = amounts[0]
    last_others = amounts[1:]  # 1..n

    if len(last_others) == 0:
        return "⚠ Data bulan sebelumnya tidak cukup untuk perbandingan."

    avg_last = sum(last_others) / len(last_others)

    if avg_last == 0:
        if this_month > 0:
            feedback = "💸 Pengeluaran bulan ini meningkat (sebelumnya tidak ada pengeluaran)."
        else:
            feedback = "✅ Tidak ada pengeluaran tercatat baik bulan ini maupun sebelumnya."
    else:
        if this_month > 1.2 * avg_last:
            feedback = "💸 Pengeluaran bulan ini meningkat lebih dari 20% dibanding rata-rata 3 bulan terakhir."
        elif this_month < 0.8 * avg_last:
            feedback = "💰 Pengeluaran bulan ini turun lebih dari 20% dibanding rata-rata 3 bulan terakhir — bagus!"
        else:
            feedback = "✅ Pengeluaran bulan ini masih dalam kisaran normal."

    feedback += f" (Bulan ini: Rp {int(this_month):,} — Rata-rata sebelumnya: Rp {int(avg_last):,})"
    return feedback

# ==================== SAVE TRANSACTION (REVISI FINAL) ====================
def save_transaction_to_mysql(data: Dict[str, Any]):
    conn = get_connection()
    if not conn:
        print("❌ Gagal konek ke DB untuk menyimpan transaksi.")
        return
    
    cursor = conn.cursor()
    
    # 🚨 PERUBAHAN KRITIS: Ambil Tanggal dari data dan hitung Periode darinya
    try:
        transaction_date_str = data.get("Tanggal", datetime.now().strftime("%Y-%m-%d")) # Format: YYYY-MM-DD
        
        # Konversi Tanggal kembali ke objek datetime untuk mendapatkan Periode
        transaction_dt = datetime.strptime(transaction_date_str, "%Y-%m-%d")
        periode = transaction_dt.strftime("%Y-%m") # Format: YYYY-MM

    except (ValueError, TypeError) as e:
        print(f"❌ Error parsing Tanggal ({transaction_date_str}): {e}. Menggunakan tanggal hari ini.")
        transaction_date_str = datetime.now().strftime("%Y-%m-%d")
        periode = datetime.now().strftime("%Y-%m")
    
    sql = """
        INSERT INTO data_transaksi 
        (Tanggal, Periode, Deskripsi, Debit, Kredit, Jenis_Akun, Sub_Kategori)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    # Data yang disiapkan di process_user_input sudah memiliki Jenis_Akun/Sub_Kategori
    deskripsi = data.get("Deskripsi", "-")
    jenis_akun = data.get("Jenis_Akun") 
    sub_kategori = data.get("Sub_Kategori")
    
    if not jenis_akun or not sub_kategori:
        # Fallback jika somehow semantic_lookup gagal (tetapi harusnya tidak terjadi)
        jenis_akun = detect_jenis_akun(deskripsi)
        sub_kategori = detect_sub_kategori(deskripsi)
        
    cursor.execute(sql, (
        transaction_date_str, # YYYY-MM-DD
        periode, # YYYY-MM
        deskripsi,
        data.get("Debit", 0),
        data.get("Kredit", 0),
        jenis_akun,
        sub_kategori
    ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✅ Transaksi disimpan ({jenis_akun} → {sub_kategori}, Tanggal: {transaction_date_str})")
    
    # ... (Panggilan ke upsert_transaction_qdrant dan evaluate_spending_behavior)
# ==================== QUERY MYSQL HELPERS ====================

def format_idr(n: int) -> str:
    try:
        n = int(n)
    except:
        n = 0
    return f"Rp {n:,}".replace(",", ".")
from datetime import datetime, timedelta
# ... (pastikan ini diimport di bagian atas) ...

# PASTIKAN ANDA SUDAH MENGIMPORT re DI BAGIAN ATAS FILE
import re 
# ... (import datetime, dateparser, dll.) ...

def clean_input_for_description(user_input: str) -> str:
    """
    Membersihkan input dari kata kunci temporal dan pemicu intent 
    agar deskripsi transaksi menjadi bersih.
    """
    t = user_input.lower()
    
    # 1. Daftar Kata Kunci Temporal dan Intent Trigger
    keywords_to_remove = [
        # Intent Triggers
        r'\bcatat\b', r'\bproses\b', r'\btambah\b', r'\btransaksi\b',
        # Daily Relative
        r'\bkemarin\b', r'\bhari\s+ini\b', r'\blusa\b', r'\bsekarang\b',
        # Weekly/Monthly/Yearly Relative
        r'\bminggu\s+lalu\b', r'\bbulan\s+lalu\b', r'\btahun\s+lalu\b',
        r'\bminggu\s+depan\b', r'\bbulan\s+depan\b', r'\btahun\s+depan\b',
        r'\blalu\b', r'\bdepan\b', r'\b\d{4}\b', # Menghapus tahun (e.g., 2023)
        # Kata kunci yang sering muncul
        r'\bpada\b'
    ]
    
    # 2. Hapus Nama Bulan
    months = ["januari", "februari", "maret", "april", "mei", "juni", "juli", 
              "agustus", "september", "oktober", "november", "desember"]
    for month in months:
        keywords_to_remove.append(month)

    # 3. Proses Pembersihan Awal (Menghapus kata kunci)
    for pattern in keywords_to_remove:
        # Menggunakan regex untuk menghapus kata kunci dengan batasan kata (\b)
        t = re.sub(pattern, ' ', t)
        
    # 4. Hapus Nominal (karena nominal seharusnya sudah diekstrak)
    # Menghapus angka dengan format ribuan/juta/rb/k
    t = re.sub(r'(\d{1,3}(\.\d{3})*(,\d+)?|\d+)\s*(ribu|rb|ratus|juta|jt|k)?', ' ', t)
    
    # 5. Finalisasi
    t = " ".join(t.split()).strip()
    
    return t

def resolve_period_to_date(period_type: str, period_val: str) -> str:
    """
    Mengubah periode yang diidentifikasi ('month', 'year') menjadi tanggal spesifik 
    (akhir periode) untuk dimasukkan ke DB.
    """
    if period_type == 'daily':
        # Sudah YYYY-MM-DD, siap digunakan
        return period_val
        
    elif period_type == 'month' and period_val:
        # period_val adalah 'YYYY-MM' (misal: '2025-11')
        dt = datetime.strptime(period_val, '%Y-%m')
        
        # Cari hari terakhir di bulan tersebut
        # Trik: Pindah ke hari pertama bulan berikutnya, lalu mundur 1 hari
        if dt.month == 12:
            next_month = dt.replace(year=dt.year + 1, month=1, day=1)
        else:
            next_month = dt.replace(month=dt.month + 1, day=1)
            
        last_day = next_month - timedelta(days=1)
        return last_day.strftime('%Y-%m-%d')
        
    elif period_type == 'year' and period_val:
        # Resolusi ke hari terakhir tahun tersebut
        return f"{period_val}-12-31" 
        
    # Fallback ke hari ini
    return datetime.now().strftime('%Y-%m-%d')

# total_by_jenis DENGAN KOREKSI LOGIKA ASET
def total_by_jenis(jenis: str, period_type: Optional[str] = None, period_val: Optional[str] = None) -> int:
    conn = get_connection()
    if not conn: return 0
    cur = conn.cursor()
    
    # 🚨 PERBAIKAN: Pastikan Aset dan Beban menggunakan Debit - Kredit (menghitung saldo atau nilai pertambahan)
    
    jenis_lower = jenis.lower()
    
    # Untuk ASET dan BEBAN: Saldo = Debit - Kredit (Positif di Debit)
    if jenis_lower in ["aset", "beban"]:
        expr = "COALESCE(SUM(Debit),0) - COALESCE(SUM(Kredit),0)" 
    # Untuk PENDAPATAN dan KEWAJIBAN: Saldo = Kredit - Debit (Positif di Kredit)
    elif jenis_lower in ["pendapatan", "kewajiban"]:
        expr = "COALESCE(SUM(Kredit),0) - COALESCE(SUM(Debit),0)"
    else:
        # Default aman
        expr = "COALESCE(SUM(Debit),0)"

    # ... (Logika SQL tetap sama, pastikan %s match dengan parameter `jenis` yang sudah Title Case) ...
    # Pastikan Anda menggunakan `jenis` (Title Case) di parameter SQL
    if period_type == "month" and period_val:
        sql = f"SELECT {expr} AS total FROM data_transaksi WHERE Jenis_Akun=%s AND Periode LIKE %s"
        cur.execute(sql, (jenis, period_val + "%")) # Menggunakan LIKE %s untuk fleksibilitas
    elif period_type == "year" and period_val:
        sql = f"SELECT {expr} AS total FROM data_transaksi WHERE Jenis_Akun=%s AND LEFT(Periode,4)=%s"
        cur.execute(sql, (jenis, period_val))
    else:
        sql = f"SELECT {expr} AS total FROM data_transaksi WHERE Jenis_Akun=%s"
        cur.execute(sql, (jenis,))
        
    row = cur.fetchone()
    cur.close(); conn.close()
    
    # ... (Logika pengambilan hasil row tetap sama) ...
    if not row:
        return 0
    if isinstance(row, tuple):
        return int(row[0] or 0)
    return int(row.get("total", 0) or 0)

# total_by_subkategori DENGAN KOREKSI PARSING HASIL
def total_by_subkategori(subkategori: str, period_type: Optional[str] = None, period_val: Optional[str] = None) -> int:
    conn = get_connection()
    if not conn: return 0
    cur = conn.cursor()
    
    # Logika Akuntansi (Diasumsikan subkategori ini adalah Pendapatan atau Kewajiban)
    expr = "COALESCE(SUM(Debit),0) - COALESCE(SUM(Kredit),0)" # Default untuk Beban/Aset
    if any(k in subkategori.lower() for k in ["pendapatan", "penjualan", "bunga", "jasa"]):
        expr = "COALESCE(SUM(Kredit),0) - COALESCE(SUM(Debit),0)" # Untuk Pendapatan
    
    # ... (Logika SQL tetap sama) ...
    if period_type == "month" and period_val:
        sql = f"SELECT {expr} AS total FROM data_transaksi WHERE Sub_Kategori=%s AND Periode LIKE %s"
        cur.execute(sql, (subkategori, period_val + "%"))
    elif period_type == "year" and period_val:
        sql = f"SELECT {expr} AS total FROM data_transaksi WHERE Sub_Kategori=%s AND LEFT(Periode,4)=%s"
        cur.execute(sql, (subkategori, period_val))
    else:
        sql = f"SELECT {expr} AS total FROM data_transaksi WHERE Sub_Kategori=%s"
        cur.execute(sql, (subkategori,))
        
    row = cur.fetchone()
    cur.close(); conn.close()
    
    # 🚨 PERBAIKAN PARSING HASIL: Pastikan pengambilan hasil konsisten
    if not row:
        return 0
    
    # Jika fetchone mengembalikan tuple (karena tidak menggunakan dictionary cursor)
    if isinstance(row, tuple):
        return int(row[0] or 0)
    
    # Jika fetchone mengembalikan dict (karena menggunakan dictionary cursor)
    return int(row.get("total", 0) or 0)

# ==================== QUERY MYSQL HELPERS (TAMBAHAN) ====================

def calculate_net_saldo() -> int:
    """
    Menghitung saldo bersih: (Total Pemasukan/Aset) - (Total Pengeluaran/Beban).
    Ini adalah saldo akuntansi.
    """
    conn = get_connection()
    if not conn: return 0
    cur = conn.cursor()

    # Hitung total penambahan/pemasukan
    sql_pemasukan = """
        SELECT COALESCE(SUM(CASE
            WHEN Jenis_Akun IN ('Pendapatan', 'Kewajiban') THEN Kredit
            WHEN Jenis_Akun = 'Aset' THEN Debit
            ELSE 0
        END), 0) AS total_pemasukan
        FROM data_transaksi;
    """
    cur.execute(sql_pemasukan)
    pemasukan = int(cur.fetchone()["total_pemasukan"] or 0)

    # Hitung total pengurangan/pengeluaran
    sql_pengeluaran = """
        SELECT COALESCE(SUM(CASE
            WHEN Jenis_Akun = 'Beban' THEN Debit
            WHEN Jenis_Akun = 'Kewajiban' THEN Debit 
            WHEN Jenis_Akun = 'Aset' THEN Kredit 
            ELSE 0
        END), 0) AS total_pengeluaran
        FROM data_transaksi;
    """
    cur.execute(sql_pengeluaran)
    pengeluaran = int(cur.fetchone()["total_pengeluaran"] or 0)

    cur.close(); conn.close()
    return pemasukan - pengeluaran

# ==================== MAIN PIPELINE HELPERS ====================
# ... (asumsi import lain sudah ada)

def get_transactions_by_jenis(jenis_akun: str, period_type: Optional[str] = None, period_val: Optional[str] = None):
    """
    Ambil daftar transaksi berdasarkan Jenis_Akun.
    Hasil: DataFrame berisi Tanggal, Deskripsi, Debit, Kredit.
    """
    conn = get_connection()
    if not conn:
        return pd.DataFrame()

    cur = conn.cursor()
    params = [jenis_akun]
    query = """
        SELECT 
            Tanggal, 
            Deskripsi, 
            COALESCE(Debit, 0) AS Debit, 
            COALESCE(Kredit, 0) AS Kredit
        FROM data_transaksi
        WHERE Jenis_Akun = %s
    """
    # ... (Logika penambahan filter waktu tetap sama) ...
    if period_type == "month" and period_val:
        query += " AND Periode = %s"
        params.append(period_val)
    elif period_type == "year" and period_val:
        query += " AND LEFT(Periode,4) = %s"
        params.append(period_val)

    print(f"[DEBUG] Query (by jenis): {query} | params={params}")
    cur.execute(query, params)
    
    # 1. Ambil nama kolom yang dikembalikan oleh MySQL (bisa huruf kecil/campur)
    column_names = [i[0] for i in cur.description]
    rows = cur.fetchall()
    
    cur.close()
    conn.close() # Pindahkan penutup koneksi ke sini, setelah fetchall()

    if not rows:
        print("[DEBUG] No rows found.")
        return pd.DataFrame()

    # 2. Buat DataFrame dengan nama kolom asli
    df = pd.DataFrame(rows, columns=column_names)

    # 3. PERBAIKAN KEYERROR: Konversi semua nama kolom menjadi Title Case
    # Ini memastikan kita bisa mengakses 'Debit' dan 'Kredit' dengan huruf D dan K besar.
    df.columns = [col.title() for col in df.columns]

    # 4. Konversi tipe data (Sekarang tidak akan ada KeyError)
    df["Debit"] = pd.to_numeric(df["Debit"], errors="coerce").fillna(0)
    df["Kredit"] = pd.to_numeric(df["Kredit"], errors="coerce").fillna(0)
    
    # Pastikan Tanggal tetap format Date/DateTime jika diperlukan untuk pemrosesan lebih lanjut
    if "Tanggal" in df.columns:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"])

    print("[DEBUG] Sample data (by jenis):")
    print(df.head())
    
    return df

import pandas as pd
# Asumsikan get_connection() dan import lainnya sudah ada

def get_transactions_by_subkategori(sub_kategori: str, period_type: Optional[str] = None, period_val: Optional[str] = None):
    conn = get_connection()
    if not conn:
        return pd.DataFrame()

    # Pastikan Anda mengimpor cursor
    # Jika cursor adalah default cursor (bukan DictCursor), kita perlu column names
    cur = conn.cursor() 
    params = [sub_kategori]

    query = """
        SELECT 
            Tanggal, 
            Deskripsi, 
            COALESCE(Debit, 0) AS Debit, 
            COALESCE(Kredit, 0) AS Kredit
        FROM data_transaksi
        WHERE Sub_Kategori = %s
    """

    if period_type == "month" and period_val:
        query += " AND Periode = %s"
        params.append(period_val)
    elif period_type == "year" and period_val:
        query += " AND LEFT(Periode,4) = %s"
        params.append(period_val)

    print("[DEBUG] Query:", query, "| params=", params)
    cur.execute(query, params)
    
    # 1. Ambil nama kolom yang dikembalikan oleh MySQL
    column_names = [i[0] for i in cur.description] 
    rows = cur.fetchall()
    
    cur.close()
    conn.close() 

    if not rows:
        print("[DEBUG] No rows found.")
        return pd.DataFrame()

    # 2. Buat DataFrame dengan nama kolom asli
    df = pd.DataFrame(rows, columns=column_names)

    # 3. PERBAIKAN KEY ERROR: Konversi semua nama kolom menjadi Title Case (e.g., 'debit' menjadi 'Debit')
    # Ini memastikan kita bisa mengakses df["Debit"] tanpa error.
    df.columns = [col.title() for col in df.columns]

    # 4. Konversi tipe data (Sekarang tidak akan ada KeyError)
    df["Debit"] = pd.to_numeric(df["Debit"], errors="coerce").fillna(0)
    df["Kredit"] = pd.to_numeric(df["Kredit"], errors="coerce").fillna(0)
    
    # Konversi Tanggal ke tipe datetime agar lebih mudah diproses
    if "Tanggal" in df.columns:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"])

    print("[DEBUG] Sample data from MySQL (after conversion):")
    print(df.head())

    return df


# ==================== DATE CONTEXT RETRIEVER (REVISI FINAL) ====================

def retrieve_context(text: str) -> Tuple[str, str]:
    """
    Ambil konteks waktu dari teks user.
    Mengembalikan (period_type, period_val):
      - ("daily", "YYYY-MM-DD") -> Untuk transaksi (Kemarin, Lusa, 10/12/2025)
      - ("month", "YYYY-MM") -> Untuk query (Bulan ini, Januari 2024)
      - ("year", "YYYY") -> Untuk query (Tahun lalu, 2023)
      - Fallback: ("daily", Hari Ini)
    """
    now = datetime.now()
    t = (text or "").lower()
    
    # Hapus Nominal, Kata Kunci Transaksi (seperti yang sudah dilakukan)
    t_clean = re.sub(r'(\d{1,3}(\.\d{3})*(,\d+)?|\d+)\s*(ribu|rb|ratus|juta|jt|k)?', ' ', t)
    t_clean = re.sub(r'\b(catat|proses|tambah|transaksi)\b', ' ', t_clean)

    # 🚨 BARU: Hapus Sub Kategori yang bisa membingungkan dateparser
    # Panggil fungsi sub-kategori untuk mendapatkan list kategori yang harus di-clean.
    # ASUMSI Anda punya daftar kategori:
    categories = ["makan", "minum", "transport", "gaji", "tagihan"] # Contoh
    for cat in categories:
         t_clean = re.sub(r'\b' + cat + r'\b', ' ', t_clean)
         
    # Bersihkan spasi ganda
    t_clean = " ".join(t_clean.split()).strip()

    # 🚨 PENGATURAN dateparser: RELATIVE BASE
    settings = {'RELATIVE_BASE': now, 'DATE_ORDER': 'DMY'}
    parsed_date = dateparser.parse(t, settings=settings)

    # 1. HANDLE TANGGAL SPESIFIK/RELATIF (Untuk pencatatan transaksi)
    if parsed_date and parsed_date.date().year >= 2020:
        # Jika dateparser menemukan tanggal (misalnya "kemarin", "10/12/2025")
        # dan tanggal itu BUKAN HARI INI (untuk menghindari false positive 'bulan ini')
        return "daily", parsed_date.strftime("%Y-%m-%d")

    # 2. HANDLE KONTEKS QUERY TAHUNAN
    m_year = re.search(r"tahun (\d{4})", t)
    if m_year:
        return "year", m_year.group(1)
    if "tahun ini" in t:
        return "year", str(now.year)
    if "tahun lalu" in t:
        return "year", str(now.year - 1)

    # 3. HANDLE KONTEKS QUERY BULANAN
    # explicit month-year name (e.g., "september 2025")
    m_month_year = re.search(r"([a-zA-Z]+)\s+(\d{4})", t)
    if m_month_year:
        month_name = m_month_year.group(1)
        year = m_month_year.group(2)
        try:
            # Menggunakan dateparser lagi untuk konversi nama bulan yang lebih baik
            parsed_month = dateparser.parse(month_name)
            if parsed_month:
                month_num = parsed_month.month
                return "month", f"{year}-{month_num:02d}"
        except Exception:
            pass
            
    # numeric month-year like "2025-09" or "09-2025" (ISO)
    m_iso = re.search(r"(\d{4})[-/](\d{1,2})|(\d{1,2})[-/](\d{4})", t)
    if m_iso:
        # Kompleksitas regex di sini, sebaiknya gunakan dateparser untuk format "12/2025"
        pass
        
    if "bulan ini" in t:
        return "month", now.strftime("%Y-%m")
    if "bulan lalu" in t:
        prev = now.replace(day=1) - timedelta(days=1)
        return "month", prev.strftime("%Y-%m")

    # 4. FALLBACK: Tanggal hari ini
    return "daily", now.strftime("%Y-%m-%d")

# ==================== HANDLE QUERY (REVISI UNTUK 3 INTENT QUERY) ====================

import re
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime
# Asumsikan semua fungsi helper (retrieve_context, semantic_lookup, format_idr,
# total_by_jenis, total_by_subkategori, get_transactions_by_jenis, dll.) sudah ada.
import re

def extract_nominal_from_input(text: str) -> int:
    """
    Mengekstrak nominal angka dari teks input.
    Mendukung format: 20 ribu, 500rb, 1.500, 100000.
    Mengembalikan nilai integer (dalam Rupiah).
    """
    t = text.lower().replace('.', '').replace(',', '') # Hapus pemisah ribuan
    
    # Regex untuk mencari angka yang diikuti kata kunci nominal
    nominal_patterns = re.findall(r'(\d+)\s*(ribu|rb|k|juta|jt)?', t)
    
    if nominal_patterns:
        number = int(nominal_patterns[0][0])
        unit = nominal_patterns[0][1]
        
        if unit in ["ribu", "rb", "k"]:
            return number * 1000
        elif unit in ["juta", "jt"]:
            return number * 1000000
        else:
            # Jika hanya angka tanpa unit (misal: 100000)
            return number
            
    # Fallback: Cari angka murni (jika tidak ada unit "ribu" dll.)
    pure_number = re.search(r'\d+', t)
    if pure_number:
        return int(pure_number.group(0))

    raise ValueError("Nominal transaksi tidak ditemukan dalam input.")

def handle_query(user_input: str, intent: str) -> str:
    t = user_input.lower()
    period_type, period_val = retrieve_context(user_input)
    print(f"[DEBUG] Detected period: {period_type}, {period_val}")

    # 1. Parameter Extraction: Ambil dari Semantic (Qdrant)
    semantic = semantic_lookup(user_input)
    jenis_akun = semantic.get("jenis_akun")
    sub_kategori = semantic.get("sub_kategori")
    
    # Tambahkan ini di awal def handle_query, setelah baris 4 (Parameter Extraction)

    # === Intent: catat_transaksi (Asumsi ID 99) ===
    # Kita harus memproses ini sebelum Intent Rincian/Total
    if intent == "catat_transaksi" or t.startswith("catat"): 
        print("[DEBUG] Processing catat_transaksi intent...")
        
        # 1. Resolusi Tanggal Final
        tanggal_transaksi = resolve_period_to_date(period_type, period_val)
        
        # 2. Pembersihan Input
        cleaned_input = clean_input_for_description(user_input)
        
        # 3. Ekstraksi Kritis (Sesuaikan dengan fungsi Anda)
        try:
            # Asumsi: Anda memiliki fungsi untuk ini
            nominal = extract_nominal_from_input(user_input) 
            
            # Sub Kategori dari semantic/fallback yang sudah Anda lakukan di atas
            if not sub_kategori:
                sub_kategori = detect_sub_kategori(cleaned_input) # Coba lagi dari input bersih
            
            # Tentukan Jenis Akun default jika belum ada
            if not jenis_akun:
                jenis_akun = "Beban" # Asumsi default pengeluaran

            # 4. Final Deskripsi: Hilangkan Sub Kategori dari string bersih
            deskripsi = cleaned_input.replace(sub_kategori.lower(), "").strip()
            
        except Exception as e:
            return f"⚠ Gagal memproses transaksi: Nominal tidak terdeteksi atau error: {e}"

        # 5. Output Konfirmasi
        # Kita menggunakan variabel yang sudah kita definisikan dan hitung
        return f"""
            ⚠ Apakah maksud Anda ingin mencatat transaksi berikut?
            - Deskripsi: {deskripsi.capitalize()}
            - Nominal: {format_idr(nominal)}
            - Jenis Akun: {jenis_akun}
            - Sub Kategori: {sub_kategori}
            - Tanggal: {tanggal_transaksi}
            Ketik 'ya' untuk konfirmasi.
            """

    # ... (Sisa kode handle_query Anda untuk lihat_rincian, tanya_total_kategori, dll. tetap di bawah ini) ...

    # 2. Fallback Lokal (Jika Qdrant tidak memberikan hasil)
    if not jenis_akun:
        jenis_akun = detect_jenis_akun(user_input)
    if not sub_kategori:
        sub_kategori = detect_sub_kategori(user_input)

    # 3. KOREKSI & PRIORITASISASI EKSPLISIT (Memastikan Aset/Beban menang)
    jenis_akun_eksplisit = None
    if "aset" in t or "aktiva" in t:
        jenis_akun_eksplisit = "Aset"
    elif "beban" in t or "pengeluaran" in t or "biaya" in t:
        jenis_akun_eksplisit = "Beban"
    elif "pendapatan" in t or "pemasukan" in t:
        jenis_akun_eksplisit = "Pendapatan"
    elif "kewajiban" in t or "hutang" in t:
        jenis_akun_eksplisit = "Kewajiban"
        
    if jenis_akun_eksplisit:
        # TIMPA Jenis Akun dengan yang eksplisit
        jenis_akun = jenis_akun_eksplisit
        
        # JIKA Jenis Akun di-override, HAPUS Sub Kategori yang salah/ambigu dari Qdrant (misal: "Penjualan")
        sub_kategori = None
        
    # 4. FINAL ADJUSTMENT (Menggunakan Sub Kategori untuk Menentukan Jenis Akun jika Jenis Akun Kosong)
    # Ini adalah satu-satunya logika yang tersisa untuk Sub Kategori.
    elif sub_kategori and sub_kategori != "Lainnya" and not jenis_akun:
        jenis_akun_from_sub = semantic.get("jenis_akun") or detect_jenis_akun(sub_kategori)
        if jenis_akun_from_sub:
             jenis_akun = jenis_akun_from_sub
    
    # Tentukan query_target dan period_text untuk output yang konsisten
    period_text = f"pada {period_val}" if period_val else "keseluruhan"

    # === Intent: lihat_rincian (Intent ID 4) ===
    # === Intent: lihat_rincian (Intent ID 4) ===
    if intent == "lihat_rincian":
        print("[DEBUG] Processing lihat_rincian intent...")
        
        # ... (Logika fetching df dan query_target tetap sama) ...
        if sub_kategori and sub_kategori != "Lainnya":
            df = get_transactions_by_subkategori(sub_kategori, period_type, period_val)
            query_target = f"sub-kategori {sub_kategori}"
        elif jenis_akun:
            df = get_transactions_by_jenis(jenis_akun, period_type, period_val)
            query_target = f"jenis akun {jenis_akun}"
        else:
            return "⚠ Rincian tidak dapat ditampilkan. Harap sebutkan jenis akun atau kategori spesifik."

        if df.empty:
            return f"📄 Tidak ada rincian transaksi {query_target} {period_text}."
        
        # 🚨 PERBAIKAN 1: Logika Penentuan Kolom Nilai (Aset bertambah di DEBIT)
        if jenis_akun in ["Aset", "Beban"]:
            sum_col = "Debit"
        elif jenis_akun in ["Pendapatan", "Kewajiban"]:
            sum_col = "Kredit"
        else:
            sum_col = "Debit" # Default aman
        
        # 1. Buat kolom 'Total' untuk tampilan
        df['Total'] = df[sum_col]
        
        # 2. Urutkan berdasarkan Tanggal terbaru ke terlama
        df_sorted = df.sort_values(by='Tanggal', ascending=False)
        
        # 🚨 PERBAIKAN 2: Batasi tampilan (misalnya 20), tapi hitung total dari semua.
        MAX_DISPLAY = 20
        df_display = df_sorted.head(MAX_DISPLAY)
        
        # 3. Hitung total keseluruhan
        total_keseluruhan = df['Total'].sum()
        
        # 4. Format output
        hasil = "\n".join([
            f"- {r['Tanggal'].strftime('%d %b')} | {r['Deskripsi']} | {format_idr(r['Total'])}"
            for r in df_display.to_dict('records')
        ])

        # Tentukan pesan footer
        footer = ""
        if len(df_sorted) > MAX_DISPLAY:
            footer = f"\n*Hanya menampilkan {MAX_DISPLAY} transaksi terbaru dari total {len(df_sorted)} transaksi.*"
        
        return f"""
        📋 Rincian **{query_target}** {period_text} (Total: {format_idr(total_keseluruhan)}):

{hasil}
{footer}

        """

    # === Intent: tanya_total_kategori (Intent ID 7) ===
    if intent == "tanya_total_kategori":
        print("[DEBUG] Processing tanya_total_kategori intent...")
        
        if sub_kategori and sub_kategori != "Lainnya":
            # Asumsi total_by_subkategori sudah menghitung total dari Debit/Kredit yang relevan
            total = total_by_subkategori(sub_kategori, period_type, period_val) 
            label = f"sub-kategori {sub_kategori}"
        elif jenis_akun:
            total = total_by_jenis(jenis_akun, period_type, period_val) 
            label = f"jenis akun {jenis_akun}"
        else:
            return "⚠ Total tidak dapat dihitung. Sub-kategori atau Jenis Akun tidak terdeteksi."
            
        return f"📊 Total {label} {period_text} adalah: {format_idr(total)}"

    # === Intent: tanya_total_akun (Intent ID 6) ===
    if intent == "tanya_total_akun":
        print("[DEBUG] Processing tanya_total_akun intent...")
        
        if not jenis_akun:
            return "⚠ Jenis Akun (Aset/Beban/Pendapatan/Kewajiban) tidak terdeteksi."
            
        # Asumsi total_by_jenis sudah menghitung total dari Debit/Kredit yang relevan
        total = total_by_jenis(jenis_akun, period_type, period_val)
        label = f"jenis akun {jenis_akun}"
        
        return f"📊 Total {label} {period_text} adalah: {format_idr(total)}"
        
    return "❓ Saya belum mengerti permintaan total/rincian Anda."# ==================== MAIN PIPELINE ====================

pending_transaction: dict = {}
# ==================== MAIN PIPELINE (REVISI UNTUK 7 INTENT BARU) ====================

# ... (Definisi pending_transaction dan fungsi-fungsi lainnya)

def process_user_input(user_input: str) -> str:
    global pending_transaction
    text = (user_input or "").lower().strip()

    # 1️⃣ Konfirmasi transaksi (Logika Konfirmasi)
    if text in ["ya", "y", "iya"] and pending_transaction:
        data = pending_transaction
        
        is_pemasukan = data["Jenis_Akun"] in ("Pendapatan", "Kewajiban") 
        
        save_transaction_to_mysql({
            "Deskripsi": data["Deskripsi"],
            "Debit": data["amount"] if not is_pemasukan else 0,
            "Kredit": data["amount"] if is_pemasukan else 0,
            "Jenis_Akun": data["Jenis_Akun"], 
            "Sub_Kategori": data["Sub_Kategori"]
        })
        pending_transaction = {}
        return "✅ Transaksi berhasil disimpan!"
    
    # 🚨 PENTING: Pengecekan Intent CATAT harus dilakukan pertama kali!
    if text.startswith("catat"):
        # Kita panggil langsung logika pencatatan tanpa melalui Qdrant
        intent = "catat_transaksi" # Tetapkan Intent secara eksplisit
    else:
        # 2️⃣ Deteksi intent Normal (Jika tidak dimulai dengan 'catat')
        intent = detect_intent(user_input)
        print(f"[DEBUG] Detected intent: {intent}")

    # ✳ Intent: catat_transaksi (Intent ID 3)
    if intent == "catat_transaksi" or text.startswith("catat"):
        ok, msg, data = validate_and_parse_catat(user_input)
        if not ok:
            return f"⚠ {msg}"
    # === START: Integrasi Logika Tanggal & Deskripsi Baru ===
        
        # A. Resolve Tanggal
        period_type, period_val = retrieve_context(user_input)
        # Gunakan fungsi helper baru Anda untuk menyelesaikan tanggal
        tanggal_transaksi = resolve_period_to_date(period_type, period_val)
        
        # B. Clean Deskripsi
        # Gunakan fungsi helper baru Anda untuk membersihkan input dari kata waktu/nominal
        cleaned_input = clean_input_for_description(user_input)
        
        # C. Ekstraksi Akun/Kategori (Kita sekarang menggunakan cleaned_input)
        # Ambil Sub Kategori dari semantic/fallback yang Anda punya, tapi pakai input yang bersih
        semantic = semantic_lookup(cleaned_input)
        jenis_akun = semantic.get("jenis_akun") or detect_jenis_akun(cleaned_input)
        sub_kategori = semantic.get("sub_kategori") or detect_sub_kategori(cleaned_input)
        
        # D. Tentukan Deskripsi Final (Anggap sisa dari cleaned_input adalah deskripsi)
        deskripsi_final = cleaned_input.replace(sub_kategori.lower(), "").strip().capitalize()
        if not deskripsi_final:
             deskripsi_final = sub_kategori # Fallback jika hanya ada kategori
        
        # === END: Integrasi Logika Tanggal & Deskripsi Baru ===
        
        # 4. Simpan Data ke pending_transaction (TIMPA dengan hasil proses kita)
        data["Jenis_Akun"] = jenis_akun
        data["Sub_Kategori"] = sub_kategori
        data["Tanggal"] = tanggal_transaksi  # 🚨 REVISI 1: Tanggal yang sudah di-resolve
        data["Deskripsi"] = deskripsi_final # 🚨 REVISI 2: Deskripsi yang sudah bersih

        pending_transaction = data

        # 5. Output Konfirmasi (Menggunakan data yang sudah direvisi)
        return f"""
        ⚠ Apakah maksud Anda ingin mencatat transaksi berikut?
        - Deskripsi: {data['Deskripsi']}  
        - Nominal: {format_idr(data['amount'])}
        - Jenis Akun: {jenis_akun}
        - Sub Kategori: {sub_kategori}
        - Tanggal: {data['Tanggal']}  
        Ketik 'ya' untuk konfirmasi.
        """

    # ✳ Intent: tanya_saldo (Intent ID 5)
    if intent == "tanya_saldo":
        saldo = calculate_net_saldo() 
        return f"💰 Total saldo bersih (Aset Netto) Anda saat ini adalah: {format_idr(saldo)}"
    
    # ✳ Intent: Tanya Total Akun / Tanya Total Kategori / Lihat Rincian (Diteruskan ke handle_query)
    if intent in ["tanya_total_akun", "tanya_total_kategori", "lihat_rincian"]:
        return handle_query(user_input, intent=intent)
    
    # ✳ Intent: greeting (Intent ID 1)
    if intent == "greeting":
        return "👋 Halo! Saya Asisten Keuangan Anda. Ada yang bisa saya bantu hari ini?"
    
    # ✳ Intent: help (Intent ID 2)
    if intent == "help":
        return "Saya bisa membantu Anda mencatat transaksi ('Catat beli pulsa 50 ribu'), menanyakan total keuangan ('Berapa total beban bulan ini?'), saldo ('Saldo saya berapa?'), atau rincian ('Lihat rincian aset')."

    # ✳ Fallback
    return "❓ Saya belum mengerti maksud Anda. Coba tanyakan dengan lebih spesifik, atau ketik 'help' untuk melihat fitur."# ==================== GRADIO ====================

def chat_interface(message, history):
    response = process_user_input(message)
    return response

iface = gr.ChatInterface(
    fn=chat_interface,
    title="💬 Asisten Keuangan Pintar",
    description="Tanyakan tentang aset, beban, pendapatan, atau catat transaksi seperti 'beli laptop 15 juta'.",
    theme="soft",
    examples=[
        ["Total aset saya berapa rupiah?"],
        ["Total pengeluaran listrik tahun ini?"],
        ["Catat beli printer 600 ribu"],
        ["Total pendapatan tahun ini"]
    ]
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)

