# 📊 Full-Quran Dataset — Analysis & Build Report

> Generated: 2026-09-15 | Workspace: `c:\Users\CHAND COMPUTER\Desktop\AudioSegment`
> Pipeline: Tarteel Whisper (CT2) → word-level timestamps → reference-word matching → WAV clips.

---

## 1. Environment
| Item | Value |
|---|---|
| Python | 3.13 (`...\Python313\python.exe`) |
| torch | 2.6.0+cu124 |
| CUDA | ✅ available |
| GPU | NVIDIA GeForce GTX 1660 SUPER — 6 GB |
| librosa / soundfile | 0.11.0 / ✅ |
| faster-whisper | ✅ |
| ctranslate2 | 4.8.1 |
| Model | `models/tarteel-whisper-base-ar-quran-ct2` (offline, `HF_HUB_OFFLINE=1`) |

## 2. Corpus inventory — `quran_arabic_words.json` (enriched)
| Check | Result |
|---|---|
| Surahs | 114 (ids sequential 1..114, no dupes) |
| Verses | 6236 (no id sequence issues, per-surah counts match) |
| Words | 77433 |
| Juz | 30 (indexes 01..30 all covered) |
| Rukus | 556 (sequential, no errors) |
| Pages | 604 |
| Sajda verses | 15 |
| Revelation type | meccan 86 / medinan 28 |
| words-vs-textspace mismatch | 0 |
| Empty text/words | none |
| Verse audio (Alafasy) | **6236 / 6236 present — 0 missing** |

## 3. Text dataset — `dataset/`
Rebuilt deterministically with `scripts/build_quran_dataset.py` (offline, hash-based split).

| File | Rows |
|---|---|
| `verses.csv` / `verses.jsonl` | 6236 |
| `words.csv` / `words.jsonl` | 77433 |
| `splits/train.csv|jsonl` | 4956 |
| `splits/valid.csv|jsonl` | 654 |
| `splits/test.csv|jsonl` | 626 |
| `manifest_meta.json`, `README.md` | regenerated |

## 4. Word-level audio dataset — `dataset/word_clips/` ⏳ (IN PROGRESS)
**Snapshot at checkpoint 794/6236 verses (12.7%):**

| Metric | Value |
|---|---|
| Verses processed | 794 / 6236 |
| Word rows cached | 14,552 / 77,433 |
| Words matched & clipped | 13,305 (91.43% match rate) |
| Clip audio so far | ~4.77 hours |

Per-surah match rate (latest snapshot):

| Surah | Matched | Total | Rate |
|---|---:|---:|---:|
| 1 (Al-Fatiha) | 24 | 31 | 77.4% |
| 2 (Al-Baqarah) | 5521 | 6120 | 90.2% |
| 3 (Aal-Imran) | 3248 | 3481 | 93.3% |
| 4 (An-Nisa) | 3458 | 3747 | 92.3% |
| 5 (Al-Ma'idah) | 524 | 619 | 84.7% |
| 6 (Al-An'am) | 26 | 28 | 92.9% (start) |
| 18, 36, 55, 78, 96, 113 | … | … | 83–100% |
| 113 (Al-Falaq) | 23 | 23 | 100% |

> Low match = ASR drop/hallucination (e.g., repeated `لَا`) — words stay in the
> manifest with `"matched": false` and `clip_file: null`, so nothing is silently lost.

### Pipeline (per verse)
1. Load Alafasy verse mp3 → 16 kHz mono.
2. `faster-whisper` (Tarteel CT2, `float16/CUDA`, `word_timestamps=True`, beam=1).
3. ASR words cleaned (p≥0.7, hallucination tokens dropped).
4. Normalized-Arabic greedy matching onto `quran_arabic_words.json` words.
5. Cut `..\\dataset\\word_clips\\<surah>/<surah><verse>/word_<NNN>.wav` (PCM16, 0.02s/0.06s pad).
6. Append manifest row + update `progress.json` checkpoint (resumable/durable).

## 5. Findings — issues caught & fixed during analysis
| # | Issue | Fix / status |
|---|---|---|
| 1 | `summary.json` was **stale** (118 verses) vs `progress.json` (792) — last run died before finalize | rebuilt via snapshot; summary now derives from the checkpoint |
| 2 | Verse `5:20` was in `progress.json` but its rows were **missing** from the cache | removed from checkpoint & re-processed → now consistent |
| 3 | Earlier `manifest.jsonl` had 680 rows only | rebuilt (14,552 rows at snapshot) |
| 4 | `run.log` was empty because Python **buffers stdout** when redirected | re-launched with `python -u` (unbuffered) → live progress in `dataset/word_clips/run.log` |
| 5 | `analyze_quran_json.py` crashed on Arabic console output (cp1252) | cosmetic — run with `PYTHONIOENCODING=utf-8` |

## 6. Current build job
Launched **in background (resumable)** from the last checkpoint — so even if this
session ends / PC restarts, `resume_word_clips.bat` continues exactly where it stopped.

```
python -u scripts/build_word_clips.py --device cuda --compute-type float16
PID     : see dataset\word_clips\build.pid
stdout  : dataset\word_clips\run.log
stderr  : dataset\word_clips\run_err.log
```

**Monitor anytime:**
```powershell
python scripts/word_clips_status.py
python scripts/snapshot_word_dataset.py        # refresh manifest/summary/with_clips files
```

**Resume after a stop:**
```powershell
resume_word_clips.bat   # or: python scripts/build_word_clips.py --device cuda
```

## 7. Deliverables produced in this session
| Item | Path |
|---|---|
| Live word-clip build (checkpoint-resumed) | PID in `dataset/word_clips/build.pid` |
| Snapshot tool | `scripts/snapshot_word_dataset.py` |
| Word rows + clip info (77,433) | `dataset/words_with_clips.csv` / `.jsonl` |
| Verse rows + clip coverage (6,236) | `dataset/verses_with_clips.csv` / `.jsonl` |
| Canonical manifest + summary (fresh) | `dataset/word_clips/manifest.jsonl` / `summary.json` |
| Text dataset (rebuilt, deterministic) | `dataset/verses.csv`, `words.csv`, `splits/…` |
| This report | `ANALYSIS_REPORT.md` |

## 8. Next steps (when build completes)
1. Re-run snapshot → final `manifest.jsonl` + `summary.json` (77,433-word scope).
2. Split word clips train/valid/test **by verse hash** for word-level fine-tuning;
   verse-level corpus stays at 80/10/10 as now.
3. Optional: `scripts/convert_model.py` + fine-tune notebook are ready for 113/114
   style word-level training on the full corpus.
4. Commit dataset + scripts (generated `word_clips/`, `fullQuran/`, `*.zip` stay git-ignored).

---
*Analysis performed with: `scripts/analyze_quran_json.py`, `scripts/verify_full_quran_dataset.py`,
`scripts/word_clips_status.py`, `scripts/snapshot_word_dataset.py`.*