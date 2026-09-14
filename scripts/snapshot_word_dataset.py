"""
Checkpoint snapshot of the full-Quran word-clip dataset.

Safe to run at ANY time while ``build_word_clips.py`` is active (or after it
finished). It builds a consistent, queryable snapshot from the incremental
checkpoint files, so the full dataset always reflects the latest progress:

1. reads  dataset/word_clips/manifest_tmp.jsonl   (incremental cache)
          dataset/word_clips/progress.json        (verse checkpoint)
2. writes dataset/word_clips/manifest.jsonl       (canonical row cache)
          dataset/word_clips/summary.json         (coverage stats)
3. writes dataset/words_with_clips.jsonl/.csv     (word-level +
          clip_file/duration/matched/asr fields)
          dataset/verses_with_clips.jsonl/.csv    (verse-level +
          matched-word counts / clip minutes)

Usage:
    python scripts/snapshot_word_dataset.py [--no-audio-check]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
WC = BASE / "dataset" / "word_clips"
DTOUT = BASE / "dataset"
DATASET_JSON = BASE / "quran_arabic_words.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-audio-check", action="store_true",
                    help="skip verifying that every referenced clip file exists")
    args = ap.parse_args()

    # ------------------------------------------------------- load checkpoint
    done = set()
    if (WC / "progress.json").exists():
        done = set(json.load(open(WC / "progress.json", encoding="utf-8")))
    rows: dict[tuple[str, int], dict] = {}
    if (WC / "manifest_tmp.jsonl").exists():
        for line in (WC / "manifest_tmp.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows[(r["verse_key"], r["word_position"])] = r

    verses_in_rows = {k for k, _ in rows}
    missing_rows = sorted(done - verses_in_rows)
    stale_rows = sorted(verses_in_rows - done)

    print(f"progress verses      : {len(done)}")
    print(f"word rows in cache   : {len(rows)}")
    print(f"done but no rows     : {missing_rows or 'NONE ✓'}")
    print(f"rows for not-done    : {stale_rows or 'NONE ✓'}")

    # ------------------------------------------------------- manifest + summary
    with open(WC / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for key in sorted(rows):
            fh.write(json.dumps(rows[key], ensure_ascii=False) + "\n")

    by_surah: dict[int, dict] = {}
    matched = total = clipped = 0
    dur_total = 0.0
    for r in rows.values():
        st = by_surah.setdefault(int(r["surah_id"]), {"matched": 0, "total": 0})
        st["total"] += 1
        total += 1
        if r.get("matched"):
            st["matched"] += 1
            matched += 1
        if r.get("clip_file"):
            clipped += 1
            dur_total += r.get("duration_sec") or 0.0

    summary = {
        "verses_processed": len(done),
        "words_total": total,
        "words_matched": matched,
        "words_clipped": clipped,
        "clips_duration_sec": round(dur_total, 2),
        "clips_duration_hours": round(dur_total / 3600.0, 2),
        "match_rate": round(100.0 * matched / max(1, total), 2),
        "per_surah": {
            str(k): {
                "matched": v["matched"],
                "total": v["total"],
                "rate": round(100.0 * v["matched"] / max(1, v["total"]), 1),
            }
            for k, v in sorted(by_surah.items())
        },
    }
    json.dump(summary, open(WC / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("manifest.jsonl + summary.json updated")

    # ------------------------------------------------------- verify clips exist
    if not args.no_audio_check:
        missing = [r["clip_file"] for r in rows.values()
                   if r.get("clip_file") and not (BASE / r["clip_file"]).exists()]
        print(f"missing clip files   : {len(missing)} "
              f"{('e.g. ' + missing[0]) if missing else ''}")
    else:
        missing = []

    # ------------------------------------------------------- merge into dataset
    if not DATASET_JSON.exists():
        print("quran_arabic_words.json not found - skipping merged dataset output")
        return

    data = json.load(open(DATASET_JSON, encoding="utf-8"))
    word_rows, verse_rows = [], []
    for s in data:
        for v in s["verses"]:
            vk = v["verse_key"]
            v_match = 0
            v_dur = 0.0
            for pos, w in enumerate(v["words"]):
                r = rows.get((vk, pos))
                cf = r.get("clip_file") if r else None
                matched_w = bool(r and r.get("matched") and cf)
                if matched_w:
                    v_match += 1
                    v_dur += r.get("duration_sec") or 0.0
                word_rows.append({
                    "word": w,
                    "verse_key": vk,
                    "surah_id": s["id"],
                    "verse_id": v["id"],
                    "global_verse_number": v["global_verse_number"],
                    "word_position": pos,
                    "juz": v["juz"],
                    "page": v["page"],
                    "audio_file": v["audio_file"],
                    "clip_file": cf,
                    "clip_start_sec": (r.get("start_sec") if r else None),
                    "clip_end_sec": (r.get("end_sec") if r else None),
                    "clip_duration_sec": (r.get("duration_sec") if r else None),
                    "clip_matched": matched_w,
                    "asr_text": (r.get("asr_text") if r else None),
                    "asr_probability": (r.get("probability") if r else None),
                })
            verse_rows.append({
                "verse_key": vk,
                "surah_id": s["id"],
                "surah_name_ar": s["name"],
                "verse_id": v["id"],
                "global_verse_number": v["global_verse_number"],
                "text": v["text"],
                "word_count": len(v["words"]),
                "words_matched": v_match,
                "words_clipped": v_match,
                "match_pct": round(100.0 * v_match / max(1, len(v["words"])), 1),
                "clips_duration_sec": round(v_dur, 3),
                "audio_file": v["audio_file"],
            })

    for name, rows_out in (("words_with_clips", word_rows), ("verses_with_clips", verse_rows)):
        with open(DTOUT / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows_out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        cols = list(rows_out[0].keys())
        with open(DTOUT / f"{name}.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows_out)

    print(f"words_with_clips     : {len(word_rows)} rows  -> {DTOUT / 'words_with_clips.csv'}")
    print(f"verses_with_clips    : {len(verse_rows)} rows  -> {DTOUT / 'verses_with_clips.csv'}")
    print("\nSNAPSHOT DONE.")


if __name__ == "__main__":
    main()