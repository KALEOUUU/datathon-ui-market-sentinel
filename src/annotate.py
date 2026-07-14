#!/usr/bin/env python3
"""
AI batch annotator for the propaganda/clickbait/chaos/market-relevance CSV.

USAGE:
    export OPENROUTER_API_KEY="sk-or-..."
    python3 -m src.annotate --input annotation_template.csv --annotator A --out annotator_A.csv

- Kalau koneksi putus / proses ke-kill, tinggal jalanin command yang sama lagi:
  script otomatis skip artikel yang statusnya udah "done" di file output, lanjut dari yang "pending".
- Batch default 10 artikel/request. Progress disimpan (atomic write) setelah SETIAP batch,
  jadi kalau tiba-tiba disconnect, paling banter rugi 1 batch (10 artikel), gak corrupt filenya.
"""

import argparse
import csv
import json
import os
import sys
import time
import random
import tempfile
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4-flash"

LABELS = [
    "clickbait_framing",
    "chaos_prone_emotion",
    "propaganda_pattern",
    "market_relevance",
]

VALID_VALUES = {"yes", "no", "uncertain"}

FIELDNAMES = [
    "article_id", "source", "published_at", "title", "content_clean",
    "clickbait_framing_label",
    "chaos_prone_emotion_label",
    "propaganda_pattern_label",
    "market_relevance_label",
    "annotator_id", "annotation_status", "rationale",
]

RUBRIC = """Kamu adalah annotator berita Indonesia yang TELITI dan analitis, terlatih dalam analisis framing
media (media framing analysis). Tugasmu mendeteksi teknik framing yang OBSERVABLE dari teks — baik yang
terang-terangan (eksplisit) MAUPUN yang halus/implisit — tapi SELALU berbasis bukti tekstual konkret,
BUKAN menebak niat atau motif penulis yang tidak bisa ditunjuk buktinya di teks.

PRINSIP UTAMA: "implisit" bukan berarti "menebak-nebak". Implisit di sini artinya TEKNIK penyajiannya halus
(tidak pakai kata-kata provokatif terang-terangan), TAPI teknik itu sendiri harus tetap bisa kamu TUNJUK
buktinya secara konkret dari teks (kutipan/frasa/struktur artikel) — bukan spekulasi soal "maksud tersembunyi"
penulis yang gak ada jejaknya di teks. Kalau kamu gak bisa nunjuk bukti konkret, jawabannya "no", bukan "yes".

Untuk SETIAP artikel, isi 4 label berikut:

1. clickbait_framing (yes/no/uncertain):
   yes jika judul sensasional/menyesatkan untuk memancing klik — baik terang-terangan (judul bombastis,
   clickbait klasik) MAUPUN halus (curiosity gap: judul sengaja menahan info kunci yang ada di isi berita
   agar orang penasaran dan klik; judul pakai framing tanya/misteri padahal isinya faktual biasa).
   Judul yang dramatis tapi proporsional & akurat terhadap isi = "no".

2. chaos_prone_emotion (yes/no/uncertain):
   yes jika CARA PENYAJIAN mengamplifikasi panik/amarah/permusuhan/eskalasi — baik lewat diksi provokatif
   terang-terangan, MAUPUN teknik halus yang observable: pengulangan angka/kata yang menakutkan tanpa
   konteks pembanding, penempatan detail paling mengerikan di judul/paragraf pembuka padahal itu bukan inti
   berita, atau pemilihan kutipan paling emosional dari banyak kutipan yang tersedia. Berita berat yang
   disajikan proporsional & faktual (tanpa amplifikasi semacam itu) = "no".

3. propaganda_pattern (yes/no/uncertain):
   yes jika ada pola manipulatif yang bisa ditunjuk buktinya, baik eksplisit (dehumanisasi, framing
   "kami vs mereka" terang-terangan, bahasa menghasut) MAUPUN teknik framing implisit berikut — SEBUTKAN
   nama tekniknya di rationale kalau kamu pilih yes/uncertain karena salah satu dari ini:
     a. One-sidedness: hanya menampilkan satu sisi/narasumber untuk isu yang jelas punya sisi lain yang
        relevan, sisi itu diabaikan sepenuhnya tanpa disebut sama sekali.
     b. Presuposisi: kalimat menyisipkan asumsi seolah fakta yang tidak dipertanyakan, padahal itu klaim
        yang belum terbukti/kontroversial (mis. "konflik yang dipicu kelompok X" padahal penyebabnya
        belum jelas/masih tuduhan).
     c. Pemilihan sumber timpang: kutipan/narasumber yang dimuat didominasi satu pihak berkepentingan,
        tanpa ada upaya konfirmasi/pembanding dari pihak lain yang relevan dan mudah dijangkau.
     d. Eufemisme/disfemisme selektif: pemilihan kata yang secara sistematis melunakkan tindakan satu
        pihak (mis. "mengamankan" utk tindakan represif) sambil mengeraskan tindakan pihak lain untuk
        peristiwa yang setara (mis. "menyerang" utk tindakan defensif serupa).
     e. Penempatan & pengulangan: informasi yang menguntungkan satu narasi ditaruh di judul/paragraf awal
        dan diulang, sementara info pembanding/klarifikasi (jika ada disebut sama sekali) ditaruh di
        paragraf akhir atau hanya disebut sekilas satu kali.
   PENTING: pilih "yes" HANYA kalau kamu bisa sebut salah satu teknik (a-e) DAN kutip/tunjuk bagian teks
   yang jadi buktinya. Kalau cuma "berasa manipulatif" tapi gak bisa ditunjuk teknik+buktinya = "no".
   Kalau ada indikasi teknik tapi buktinya lemah/ambigu = "uncertain" (sebut teknik yang dicurigai).
   Ini BUKAN penilaian benar/salah (hoaks) — jangan nilai kebenaran faktual artikel, hanya CARA
   PENYAJIANNYA.

4. market_relevance (yes/no/uncertain):
   yes HANYA jika SUBSTANSI artikel relevan langsung dengan ekonomi/pasar Indonesia (kebijakan fiskal/
   moneter, emiten, sektor industri, makroekonomi, regulasi bisnis, anggaran publik dengan dampak fiskal
   nyata, dsb). Ini BUKAN prediksi arah IHSG dan BUKAN sekadar menyebut angka/rupiah sekilas tanpa
   substansi ekonomi.

KALIBRASI "uncertain" — PENTING: "uncertain" bukan tombol default kalau ragu sedikit. Pakai "uncertain"
HANYA kalau ada indikasi konkret (bisa ditunjuk buktinya) tapi levelnya ambigu/lemah untuk disebut penuh
"yes". Kalau memang gak ada indikasi apa pun di teks, jawabannya "no", BUKAN "uncertain". Jangan sistematis
condong ke "no" untuk menghindari tanggung jawab — kalau bukti teknik framing implisit (a-e di atas) ADA
dan jelas, jawab "yes" dengan percaya diri, jangan diturunkan ke "uncertain" hanya karena terasa "kurang
eksplisit". Sebagian artikel media Indonesia yang profesional memang genuinely netral — itu valid jadi "no"
di semua label, jangan dipaksakan cari-cari pola yang gak ada.

ATURAN FORMAT (WAJIB, PALING PENTING):
- Nilai label HARUS PERSIS salah satu dari tiga string ini (huruf kecil semua): "yes", "no", "uncertain".
- DILARANG KERAS memakai true/false, "Yes"/"No" (kapital), 1/0, ya/tidak, atau bentuk lain apa pun.
- DILARANG KERAS memakai tanda kutip dua (") di DALAM isi rationale, walau untuk mengutip kalimat dari
  artikel — ini akan merusak format JSON. Kalau perlu mengutip kata/frasa persis dari artikel, pakai
  tanda kutip satu (') saja, contoh: propaganda: yes, eufemisme 'mengamankan' utk tindakan represif.
  JANGAN PERNAH tulis karakter " di tengah kalimat rationale.
- Hanya ada SATU kolom "rationale" per artikel (bukan 4 rationale terpisah). Isi dengan 4 alasan singkat
  digabung dalam satu string, format persis: "clickbait: <alasan>; chaos: <alasan>; propaganda: <alasan>; market: <alasan>"
  Tiap <alasan> 4-12 kata, WAJIB menyebut sesuatu yang KONKRET dari artikel (nama, angka, frasa singkat
  dari teks, atau nama teknik framing a-e jika relevan untuk propaganda_pattern) — DILARANG kalimat
  generik yang bisa dipasang ke artikel manapun.
  Contoh BENAR (ada teknik implisit terdeteksi): "clickbait: no, judul sesuai isi; chaos: no, angka AQI
  155 disajikan faktual; propaganda: yes, one-sidedness hanya kutip BMKG tanpa pembanding; market: no,
  soal cuaca bukan ekonomi"
  Contoh BENAR (semua no, genuinely netral): "clickbait: no, judul cocok isi soal KIM Plus; chaos: no,
  kutipan Dasco netral; propaganda: no, semua pihak dikutip proporsional; market: no, hanya koalisi pilkada"
  Contoh SALAH (generik tanpa detail): "clickbait: no, judul sesuai isi; chaos: no, netral; propaganda:
  no, tidak ada pola; market: no, bukan ekonomi"

Balas HANYA dalam format JSON array, tanpa teks lain, tanpa markdown fence. CONTOH OUTPUT PERSIS (ikuti formatnya, jangan tambah/kurangi field):
[
  {
    "article_id": "abc-123",
    "clickbait_framing_label": "no",
    "chaos_prone_emotion_label": "no",
    "propaganda_pattern_label": "yes",
    "market_relevance_label": "no",
    "rationale": "clickbait: no, judul sesuai isi soal cuaca Jakarta; chaos: no, angka AQI disajikan faktual; propaganda: yes, one-sidedness hanya kutip data BMKG/IQAir tanpa pembanding pemerintah; market: no, topik lingkungan bukan ekonomi"
  }
]
Array harus punya JUMLAH ELEMEN YANG SAMA PERSIS dan article_id YANG SAMA PERSIS dengan artikel yang diberikan, urutan bebas tapi semua id wajib ada."""



def build_user_prompt(rows):
    articles = []
    for r in rows:
        articles.append({
            "article_id": r["article_id"],
            "title": r["title"],
            "content": r["content_clean"][:4000],  # safety cap per article
        })
    return "Berikut artikel yang perlu dianotasi (JSON array input):\n" + json.dumps(articles, ensure_ascii=False)


def repair_truncated_json_array(content):
    """
    Best-effort repair for a JSON array that got cut off mid-string (e.g. hit max_tokens)
    or has a stray unescaped quote breaking one object. Strategy: walk backwards from the
    end, drop the last (broken/incomplete) object, and close the array at the last known-good
    '},' boundary. Returns a parsed list, or raises if nothing usable is found.
    """
    # try trimming from the end at each '},' boundary until it parses as a valid JSON array
    idx = content.rfind("},")
    while idx != -1:
        candidate = content[:idx + 1] + "]"
        try:
            result = json.loads(candidate)
            if isinstance(result, list) and result:
                return result
        except Exception:
            pass
        idx = content.rfind("},", 0, idx)
    raise ValueError("could not repair truncated JSON array")


def call_openrouter(api_key, rows, max_retries=5, debug=False):
    payload = {
        "model": MODEL,
        "temperature": 0.1,
        "max_tokens": 8000,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": build_user_prompt(rows)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = min(60, (2 ** attempt) + random.random())
                print(f"  [rate limited] retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            # some providers return HTTP 200 but embed an error object instead of choices
            if "error" in data:
                raise RuntimeError(f"API returned error object: {data['error']}")
            if "choices" not in data or not data["choices"]:
                raise RuntimeError(f"Respons tidak punya 'choices' sama sekali. Raw keys: {list(data.keys())}, raw: {json.dumps(data, ensure_ascii=False)[:1500]}")

            choice = data["choices"][0]
            message = choice.get("message", {}) or {}
            content = message.get("content")
            finish_reason = choice.get("finish_reason")

            if content is None:
                # model may have refused, hit a filter, or provider used a different field
                refusal = message.get("refusal")
                raise RuntimeError(
                    f"content kosong (None) dari API. finish_reason={finish_reason}, "
                    f"refusal={refusal!r}, raw message keys={list(message.keys())}"
                )
            content = content.strip()
            if finish_reason == "length":
                print("  [WARNING] respons kepotong karena max_tokens habis (finish_reason=length) — "
                      "akan dicoba repair, tapi pertimbangkan turunkan --batch-size kalau sering terjadi.")
            if debug:
                print("\n===== RAW MODEL OUTPUT (before parsing) =====")
                print(content[:3000])
                print("===== END RAW MODEL OUTPUT =====\n")
            # strip accidental markdown fences
            if content.startswith("```"):
                content = content.strip("`")
                content = content.split("\n", 1)[1] if "\n" in content else content
                if content.lower().startswith("json"):
                    content = content[4:]
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as je:
                print(f"  [json truncated/malformed: {je}] mencoba repair (buang objek terakhir yang rusak)...")
                parsed = repair_truncated_json_array(content)
                print(f"  [repair OK] dapat {len(parsed)} artikel dari batch ini (mungkin < jumlah diminta, sisanya tetap pending).")
            if debug:
                print("===== PARSED FIRST ITEM (before normalize) =====")
                print(json.dumps(parsed[0], ensure_ascii=False, indent=2) if parsed else "EMPTY")
                print("===== END PARSED FIRST ITEM =====\n")
            return parsed
        except Exception as e:
            last_err = e
            wait = min(60, (2 ** attempt) + random.random())
            print(f"  [error: {e}] retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}")


LABEL_NORMALIZE = {
    "true": "yes", "false": "no",
    "y": "yes", "n": "no",
    "ya": "yes", "tidak": "no", "tidak yakin": "uncertain",
    "1": "yes", "0": "no",
}
MAX_RATIONALE_WORDS = 60  # combined rationale for all 4 labels, safety cap only


def normalize_item(item):
    """Mutates item in place: lowercase/normalize label values, truncate overly long rationale."""
    for lbl in LABELS:
        lk = f"{lbl}_label"
        val = item.get(lk)
        if isinstance(val, bool):
            val = "yes" if val else "no"
        elif val is not None:
            val = str(val).strip().lower()
            val = LABEL_NORMALIZE.get(val, val)
        item[lk] = val

    rat = str(item.get("rationale", "")).strip()
    words = rat.split()
    if len(words) > MAX_RATIONALE_WORDS:
        rat = " ".join(words[:MAX_RATIONALE_WORDS])
    item["rationale"] = rat
    return item


def validate_result(item, expected_id):
    if item.get("article_id") != expected_id:
        return False
    for lbl in LABELS:
        if item.get(f"{lbl}_label") not in VALID_VALUES:
            return False
    if not str(item.get("rationale", "")).strip():
        return False
    return True


def atomic_write_csv(path, fieldnames, rows):
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix=".tmp_annot_")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_or_init_state(input_path, out_path, annotator_id):
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"Resuming from existing {out_path} ({len(rows)} rows).")
        return rows
    with open(input_path, newline="", encoding="utf-8") as f:
        src_rows = list(csv.DictReader(f))
    rows = []
    for r in src_rows:
        row = {k: r.get(k, "") for k in ["article_id", "source", "published_at", "title", "content_clean"]}
        for lbl in LABELS:
            row[f"{lbl}_label"] = ""
        row["rationale"] = ""
        row["annotator_id"] = annotator_id
        row["annotation_status"] = "pending"
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to source template csv")
    ap.add_argument("--out", required=True, help="path to this annotator's output csv (also resume file)")
    ap.add_argument("--annotator", required=True, help="e.g. A or B")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--sleep-between-batches", type=float, default=1.0)
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: set OPENROUTER_API_KEY env var atau pakai --api-key", file=sys.stderr)
        sys.exit(1)

    rows = load_or_init_state(args.input, args.out, args.annotator)
    by_id = {r["article_id"]: r for r in rows}

    pending = [r for r in rows if r["annotation_status"] != "done"]
    print(f"Total artikel: {len(rows)} | belum selesai: {len(pending)}")

    batches = [pending[i:i + args.batch_size] for i in range(0, len(pending), args.batch_size)]

    for bi, batch in enumerate(batches, 1):
        print(f"[Annotator {args.annotator}] Batch {bi}/{len(batches)} ({len(batch)} artikel)...")
        try:
            result = call_openrouter(args.api_key, batch)
        except Exception as e:
            print(f"  BATCH GAGAL total, skip batch ini untuk sekarang (masih 'pending'): {e}")
            continue

        result_by_id = {item.get("article_id"): item for item in result if isinstance(item, dict)}

        for r in batch:
            aid = r["article_id"]
            item = result_by_id.get(aid)
            if item is not None:
                item = normalize_item(item)
            if item is None or not validate_result(item, aid):
                print(f"  WARNING: hasil invalid/hilang untuk article_id={aid}, tetap 'pending' (akan dicoba lagi run berikutnya).")
                continue
            for lbl in LABELS:
                by_id[aid][f"{lbl}_label"] = item[f"{lbl}_label"]
            by_id[aid]["rationale"] = item["rationale"]
            by_id[aid]["annotation_status"] = "done"

        # save after EVERY batch -> crash-safe
        atomic_write_csv(args.out, FIELDNAMES, rows)
        done_count = sum(1 for r in rows if r["annotation_status"] == "done")
        print(f"  saved. progress: {done_count}/{len(rows)} done.")
        time.sleep(args.sleep_between_batches)

    done_count = sum(1 for r in rows if r["annotation_status"] == "done")
    print(f"\nSelesai untuk annotator {args.annotator}: {done_count}/{len(rows)} artikel done.")
    if done_count < len(rows):
        print("Masih ada yang pending/gagal — jalankan ulang command yang sama untuk lanjut/retry.")


if __name__ == "__main__":
    main()
