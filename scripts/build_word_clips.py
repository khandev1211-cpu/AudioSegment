"""
Full-Quran WORD-LEVEL audio dataset builder.

For every ayah (6236 Alafasy verse mp3s):
    1. transcribe the whole verse with faster-whisper (tarteel-whisper-base-ar-quran,
       CT2) and word_timestamps=True
    2. match ASR words against the reference Arabic word list from
       quran_arabic_words.json using normalized-Arabic greedy matching
    3. cut one 16-bit PCM WAV clip per matched reference word
    4. append the clip to an incremental manifest + checkpoint progress

Output (in <workspace>/dataset/word_clips/):
    <surah:03d>/<surah:03d><verse:03d>/word_<NNN>.wav   (NNN = word position)
    manifest.jsonl          final rows: verse_key, surah_id, verse_id, word,
                            word_position, start_sec, end_sec, duration_sec,
                            clip_file, matched, asr_text, probability
    manifest_tmp.jsonl      incremental cache (used for resume; safe to ignore)
    progress.json           set of verse keys already processed
    summary.json            per-surah coverage + overall match rate

Usage:
    python scripts/build_word_clips.py                      # whole Quran
    python scripts/build_word_clips.py --surah 114          # one surah
    python scripts/build_word_clips.py --limit 20           # first N verses
    python scripts/build_word_clips.py --device cuda --compute-type float16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from quran_segmenter.audio_io import load_audio, save_wav_pcm16
from quran_segmenter.align import normalize_arabic
from quran_segmenter.word_aligner import (
    WordAligner,
    HALLUC_PROB,
    HALLUC_TOKENS,
    torch_cuda_available,
)

JSON_PATH = WORKSPACE / "quran_arabic_words.json"
CT2_MODEL_DIR = WORKSPACE / "models" / "tarteel-whisper-base-ar-quran-ct2"
OUT_ROOT = WORKSPACE / "dataset" / "word_clips"
MANIFEST = OUT_ROOT / "manifest.jsonl"
TMP_MANIFEST = OUT_ROOT / "manifest_tmp.jsonl"
PROGRESS = OUT_ROOT / "progress.json"
SUMMARY = OUT_ROOT / "summary.json"

SR = 16000
PAD_START = 0.02   # sec extra before the word start
PAD_END = 0.06     # sec extra after the word end
MAX_SKIP = 2       # tolerated extra ASR words while scanning to a ref word


def load_state() -> tuple[dict, dict]:
    """Return (rows_by_key, done_verse_set)."""
    rows: dict = {}
    if TMP_MANIFEST.exists():
        for line in TMP_MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows[(r["verse_key"], r["word_position"])] = r
    done = set()
    if PROGRESS.exists():
        done = set(json.load(open(PROGRESS, encoding="utf-8")))
    return rows, done


def ref_tokens_of(verse: dict) -> list[str]:
    norm = normalize_arabic(" ".join(verse["words"]))
    return norm.split() if norm else []


def asr_words_of(segments) -> list[dict]:
    """Collect cleaned ASR words from a faster-whisper segment iterator."""
    words = []
    for sgm in segments:
        for w in sgm.words or []:
            if w.probability < HALLUC_PROB:
                continue
            wt = w.word.strip()
            norm = normalize_arabic(wt)
            if not norm or norm in HALLUC_TOKENS:
                continue
            words.append({
                "text": wt,
                "norm": norm,
                "start": float(w.start),
                "end": float(w.end),
                "probability": float(w.probability),
            })
    return words


def match_ref_to_asr(ref_norms: list[str], asr: list[dict]):
    """Greedy forward match of reference words onto ASR word stream.

    Returns list of (word_index_in_asr | None) per reference token.
    """
    from quran_segmenter.word_aligner import _same_norm

    result: list[int | None] = []
    ptr = 0
    for tok in ref_norms:
        found = None
        scan = 0
        while ptr + scan < len(asr):
            if _same_norm(asr[ptr + scan]["norm"], tok):
                found = ptr + scan
                break
            scan += 1
            if scan > MAX_SKIP:
                break
        if found is not None:
            result.append(found)
            ptr = found + 1
        else:
            result.append(None)
    return result


def process_verse(model, s: dict, v: dict):
    """Transcribe + match + cut word clips for one verse. Returns manifest rows."""
    n = s["id"]
    audio_path = WORKSPACE / v["audio_file"]
    if not audio_path.exists():
        return [{
            "verse_key": v["verse_key"], "surah_id": n, "verse_id": v["id"],
            "word": w, "word_position": pos, "start_sec": None, "end_sec": None,
            "duration_sec": None, "clip_file": None, "matched": False,
            "asr_text": None, "probability": None, "error": "audio_missing",
        } for pos, w in enumerate(v["words"])]

    y, sr = load_audio(audio_path, sr=SR)
    duration = len(y) / sr

    try:
        segments, _info = model.transcribe(
            y,
            language="ar",
            task="transcribe",
            word_timestamps=True,
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        asr = asr_words_of(segments)
    except Exception as e:  # noqa: BLE001
        return [{
            "verse_key": v["verse_key"], "surah_id": n, "verse_id": v["id"],
            "word": w, "word_position": pos, "start_sec": None, "end_sec": None,
            "duration_sec": None, "clip_file": None, "matched": False,
            "asr_text": None, "probability": None, "error": f"asr:{type(e).__name__}:{e}",
        } for pos, w in enumerate(v["words"])]

    ref_norms = ref_tokens_of(v)
    ali = match_ref_to_asr(ref_norms, asr)
    rows = []
    for pos, (w_ref, w_norm) in enumerate(zip(v["words"], ref_norms)):
        idx = ali[pos]
        if idx is None:
            rows.append({
                "verse_key": v["verse_key"], "surah_id": n, "verse_id": v["id"],
                "word": w_ref, "word_norm": w_norm, "word_position": pos,
                "start_sec": None, "end_sec": None, "duration_sec": None,
                "clip_file": None, "matched": False, "asr_text": None,
                "probability": None,
            })
            continue

        aw = asr[idx]
        start = max(0.0, aw["start"] - PAD_START)
        end = min(duration, aw["end"] + PAD_END)
        if end - start < 0.05:  # too-short clip guard
            rows.append({
                "verse_key": v["verse_key"], "surah_id": n, "verse_id": v["id"],
                "word": w_ref, "word_norm": w_norm, "word_position": pos,
                "start_sec": None, "end_sec": None, "duration_sec": None,
                "clip_file": None, "matched": False, "asr_text": aw["text"],
                "probability": round(aw["probability"], 4), "error": "too_short",
            })
            continue

        folder = OUT_ROOT / f"{n:03d}" / f"{n:03d}{v['id']:03d}"
        clip = folder / f"word_{pos:03d}.wav"
        s_idx = int(round(start * SR))
        e_idx = int(round(end * SR))
        save_wav_pcm16(clip, y[s_idx:e_idx], SR)
        rows.append({
            "verse_key": v["verse_key"], "surah_id": n, "verse_id": v["id"],
            "word": w_ref, "word_norm": w_norm, "word_position": pos,
            "start_sec": round(start, 4), "end_sec": round(end, 4),
            "duration_sec": round(end - start, 4),
            "clip_file": clip.relative_to(WORKSPACE).as_posix(),
            "matched": True, "asr_text": aw["text"],
            "probability": round(aw["probability"], 4),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build full-Quran word-level audio dataset")
    ap.add_argument("--surah", type=int, default=0, help="only process this surah number")
    ap.add_argument("--limit", type=int, default=0, help="process at most N verses (0 = all)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--compute-type", default=None, help="float16/int8 (default: auto)")
    args = ap.parse_args()

    data = json.load(open(JSON_PATH, encoding="utf-8"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows, done = load_state()
    print(f"checkpoint: {len(done)} verses processed, {len(rows)} rows cached")

    todo = []
    for s in data:
        if args.surah and s["id"] != args.surah:
            continue
        for v in s["verses"]:
            if v["verse_key"] in done:
                continue
            todo.append((s, v))
    if args.limit and args.limit > 0:
        todo = todo[: args.limit]
    print(f"todo verses: {len(todo)}")

    if not todo:
        print("nothing to do.")
        sys.exit(0)

    device = args.device or ("cuda" if torch_cuda_available() else "cpu")
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")
    print(f"model  : {CT2_MODEL_DIR.name}\ndevice : {device} ({compute_type})")
    aligner = WordAligner(CT2_MODEL_DIR, device=device, compute_type=compute_type)
    model = aligner.model

    t0 = time.time()
    with open(TMP_MANIFEST, "a", encoding="utf-8") as f_tmp:
        for i, (s, v) in enumerate(todo, start=1):
            vrows = process_verse(model, s, v)
            for r in vrows:
                f_tmp.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in vrows:
                rows[(r["verse_key"], r["word_position"])] = r
            done.add(v["verse_key"])
            json.dump(sorted(done), open(PROGRESS, "w", encoding="utf-8"))
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / el
                print(f"  [{i:5d}/{len(todo):5d}] verses | {rate:.2f} v/s | "
                      f"elapsed {el/60:.1f} min | ETA {(len(todo)-i)/max(rate,1e-9)/60:.1f} min")

    # ---- final manifest + summary ---------------------------------------- #
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        for key in sorted(rows, key=lambda k: (k[0], k[1])):
            fh.write(json.dumps(rows[key], ensure_ascii=False) + "\n")

    by_surah: dict[int, dict] = {}
    matched = total = clipped = 0
    for r in rows.values():
        st = by_surah.setdefault(r["surah_id"], {"matched": 0, "total": 0})
        st["total"] += 1
        if r.get("matched"):
            st["matched"] += 1
            matched += 1
        if r.get("clip_file"):
            clipped += 1
        total += 1

    summary = {
        "verses_processed": len(done),
        "words_total": total,
        "words_matched": matched,
        "words_clipped": clipped,
        "match_rate": round(100.0 * matched / max(1, total), 2),
        "per_surah": {
            str(k): {"matched": v["matched"], "total": v["total"],
                     "rate": round(100.0 * v["matched"] / max(1, v["total"]), 1)}
            for k, v in sorted(by_surah.items())
        },
    }
    json.dump(summary, open(SUMMARY, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nDONE.")
    print(f"  words total    : {total}")
    print(f"  matched/clipped: {matched}/{clipped}")
    print(f"  match rate     : {summary['match_rate']}%")
    print(f"  manifest       : {MANIFEST}")
    print(f"  summary        : {SUMMARY}")


if __name__ == "__main__":
    main()