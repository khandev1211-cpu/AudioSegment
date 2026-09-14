"""
Build the complete Quran dataset from the enriched quran_arabic_words.json.

Outputs (in <workspace>/dataset):
    README.md                  dataset description
    manifest_meta.json         dataset-level metadata + column docs
    verses.jsonl / verses.csv  6236 rows  (one per ayah)
    words.jsonl  / words.csv   77433 rows (one per word)
    splits/train.csv valid.csv test.csv   deterministic 80/10/10 verse split

Verse row: verse_key, surah_id, surah_name_ar, surah_name_en, revelation_type,
           verse_id, global_verse_number, text, words[], ruku, juz, manzil,
           page, hizb_quarter, sajda, word_count, audio_file
Word row:  word, verse_key, surah_id, verse_id, global_verse_number,
           word_position (0-based), ruku, juz, page, audio_file
"""
import csv
import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "quran_arabic_words.json")
OUT = os.path.join(BASE, "dataset")
SPLITS = os.path.join(OUT, "splits")


def split_of(verse_key: str) -> str:
    """Deterministic 80/10/10 split (hash-based, reproducible)."""
    h = int(hashlib.sha256(verse_key.encode("utf-8")).hexdigest(), 16)
    r = h % 100
    if r < 80:
        return "train"
    if r < 90:
        return "valid"
    return "test"


def main() -> None:
    if not os.path.exists(SRC):
        sys.exit(f"enriched dataset not found: {SRC}")
    data = json.load(open(SRC, encoding="utf-8"))
    os.makedirs(SPLITS, exist_ok=True)

    verse_rows = []
    word_rows = []
    split_rows = {"train": [], "valid": [], "test": []}

    for s in data:
        for v in s["verses"]:
            row = {
                "verse_key": v["verse_key"],
                "surah_id": s["id"],
                "surah_name_ar": s["name"],
                "surah_name_en": s.get("english_name"),
                "revelation_type": s.get("revelation_type"),
                "verse_id": v["id"],
                "global_verse_number": v["global_verse_number"],
                "text": v["text"],
                "words": v["words"],
                "ruku": v["ruku"],
                "juz": v["juz"],
                "manzil": v["manzil"],
                "page": v["page"],
                "hizb_quarter": v["hizb_quarter"],
                "sajda": v["sajda"],
                "word_count": len(v["words"]),
                "audio_file": v["audio_file"],
            }
            verse_rows.append(row)

            for pos, w in enumerate(v["words"]):
                word_rows.append({
                    "word": w,
                    "verse_key": v["verse_key"],
                    "surah_id": s["id"],
                    "verse_id": v["id"],
                    "global_verse_number": v["global_verse_number"],
                    "word_position": pos,
                    "ruku": v["ruku"],
                    "juz": v["juz"],
                    "page": v["page"],
                    "audio_file": v["audio_file"],
                })

            sp = split_of(v["verse_key"])
            split_rows[sp].append({
                "verse_key": v["verse_key"],
                "global_verse_number": v["global_verse_number"],
                "surah_id": s["id"],
                "verse_id": v["id"],
                "text": v["text"],
                "words": v["words"],
                "audio_file": v["audio_file"],
                "split": sp,
            })

    # ---- write verse level ------------------------------------------------ #
    with open(os.path.join(OUT, "verses.jsonl"), "w", encoding="utf-8") as fh:
        for r in verse_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "verses.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        cols = [k for k in verse_rows[0] if k != "words"] + ["words_joined"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in verse_rows:
            out = {k: r[k] for k in r if k in cols}
            out["words_joined"] = " ".join(r["words"])
            w.writerow(out)

    # ---- write word level ------------------------------------------------ #
    with open(os.path.join(OUT, "words.jsonl"), "w", encoding="utf-8") as fh:
        for r in word_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "words.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        cols = list(word_rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(word_rows)

# ---- write splits ----------------------------------------------------- #
    for name, rows in split_rows.items():
        cols = list(rows[0].keys())
        with open(os.path.join(SPLITS, f"{name}.csv"), "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(SPLITS, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- meta + README ---------------------------------------------------- #
    words_total = len(word_rows)
    verses_total = len(verse_rows)
    audios = {r["audio_file"] for r in verse_rows}
    missing_audio = [a for a in sorted(audios) if not os.path.exists(os.path.join(BASE, a))]
    meta = {
        "dataset": "Quran Complete (114 surahs) — Alafasy recitation",
        "schema": "1.0",
        "sources": {
            "text_words": "quran_arabic_words.json (enriched)",
            "audio": "fullQuran/alafasy_quran_complete/ (Alafasy verse mp3s)",
        },
        "counts": {
            "surahs": len(data),
            "verses": verses_total,
            "words": words_total,
            "juz": 30,
            "rukus": sum(len(s["rukus"]) for s in data),
            "pages": len({r["page"] for r in verse_rows}),
            "sajda_verses": sum(1 for r in verse_rows if r["sajda"]),
            "audio_files": len(audios),
            "missing_audio_files": len(missing_audio),
        },
        "splits": {k: len(v) for k, v in split_rows.items()},
        "columns": {
            "verse": list(verse_rows[0].keys()),
            "word": list(word_rows[0].keys()),
            "split": list(split_rows["train"][0].keys()),
        },
        "sample_verse": verse_rows[69 * 36],
        "sample_word": word_rows[0],
    }
    with open(os.path.join(OUT, "manifest_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(_readme(meta))

    print(f"verses : {verses_total}")
    print(f"words  : {words_total}")
    print(f"splits : { {k: len(v) for k, v in split_rows.items()} }")
    print(f"audio  : {len(audios)} unique, {len(missing_audio)} missing")
    print(f"output : {OUT}")


def _readme(m: dict) -> str:
    c = m["counts"]
    cols_v = ", ".join(f"`{c_}`" for c_ in m["columns"]["verse"])
    return f"""# 📖 Quran Complete Dataset (114 Surahs — Alafasy)

Word/text + audio dataset for the whole Quran.

## Numbers
- Surahs: {c['surahs']} | Verses: {c['verses']} | Words: {c['words']}
- Juz: {c['juz']} | Rukus: {c['rukus']} | Pages: {c['pages']} | Sajda verses: {c['sajda_verses']}
- Audio files: {c['audio_files']} (missing: {c['missing_audio_files']})
- Split (verse-level, deterministic hash): {json.dumps(m['splits'])}

## Files
| File | Rows | Description |
|------|-----:|-------------|
| `verses.jsonl` / `verses.csv` | {c['verses']} | one row per ayah (text + words + metadata + audio path) |
| `words.jsonl` / `words.csv` | {c['words']} | one row per word (for word-level tasks) |
| `splits/train.csv|jsonl` | {m['splits']['train']} | ASR train manifest |
| `splits/valid.csv|jsonl` | {m['splits']['valid']} | validation manifest |
| `splits/test.csv|jsonl` | {m['splits']['test']} | test manifest |

## Verse row columns
{cols_v}

## Usage
```python
import json
rows = [json.loads(l) for l in open("dataset/verses.jsonl", encoding="utf-8")]
# rows[0]["audio_file"]  -> "fullQuran/alafasy_quran_complete/001/001001.mp3"
# rows[0]["text"]        -> Arabic ayah text
```

Audio is the **Mishary Alafasy** full-Quran recitation split verse-by-verse
(already on disk under `fullQuran/alafasy_quran_complete/`).

## Regenerate
```bash
python scripts/build_quran_dataset.py
```
"""


if __name__ == "__main__":
    main()