# 📖 Quran Complete Dataset (114 Surahs — Alafasy)

Word/text + audio dataset for the whole Quran.

## Numbers
- Surahs: 114 | Verses: 6236 | Words: 77433
- Juz: 30 | Rukus: 556 | Pages: 604 | Sajda verses: 15
- Audio files: 6236 (missing: 0)
- Split (verse-level, deterministic hash): {"train": 4956, "valid": 654, "test": 626}

## Files
| File | Rows | Description |
|------|-----:|-------------|
| `verses.jsonl` / `verses.csv` | 6236 | one row per ayah (text + words + metadata + audio path) |
| `words.jsonl` / `words.csv` | 77433 | one row per word (for word-level tasks) |
| `splits/train.csv|jsonl` | 4956 | ASR train manifest |
| `splits/valid.csv|jsonl` | 654 | validation manifest |
| `splits/test.csv|jsonl` | 626 | test manifest |

## Verse row columns
`verse_key`, `surah_id`, `surah_name_ar`, `surah_name_en`, `revelation_type`, `verse_id`, `global_verse_number`, `text`, `words`, `ruku`, `juz`, `manzil`, `page`, `hizb_quarter`, `sajda`, `word_count`, `audio_file`

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
