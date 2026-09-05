"""Reference-based Quran audio segmentation.

Every qari's ``114.mp3`` is cut into the exact 12 phrase segments that match
``refrence/AnNaas`` (basmalah x2 + An-Nas ayahs split into phrases). Word
timestamps come from the Tarteel Whisper model running under faster-whisper
(CT2, offline), so each qari's own timing is used.

Output per qari:
    output/<qari>/
        surah_114_an-nas__reference_segments.json   <- all details
        segments/surah_114_an_nas_ref/seg_001.wav .. seg_012.wav

Usage:
    python reference_segment.py
    python reference_segment.py --qari minshawi
    python reference_segment.py --limit 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from quran_segmenter.audio_io import audio_file_info, load_audio, rms_db, save_wav_pcm16
from quran_segmenter.config import (
    PITCH_CENTS_REF,
    PITCH_FMAX,
    PITCH_FMIN,
    PITCH_HOP,
    PITCH_SAMPLE_RATE,
)
from quran_segmenter.features import extract_audio_features, extract_pitch_features
from quran_segmenter.surah_meta import QARI_AR_NAMES, get_surah_meta
from quran_segmenter.word_aligner import WordAligner, match_phrases_to_words, phrase_cut_times

WORKSPACE = Path(__file__).resolve().parent
CT2_MODEL_DIR = WORKSPACE / "models" / "tarteel-whisper-base-ar-quran-ct2"
OUTPUT_DIR = WORKSPACE / "output"
SEGMENT_PREFIX = "seg_"
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
SEARCH_EXCLUDE = {"output", "quran_segmenter", "segments", "venv", ".git", "__pycache__", "refrence", "models", "scripts"}


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "unnamed"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reference-based Quran segmentation + detailed JSON")
    p.add_argument("--qari", help="only process qaris whose folder name contains this substring")
    p.add_argument("--limit", type=int, default=0, help="process at most N files (0 = all)")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    p.add_argument("--surah", type=int, default=114, help="surah number to build reference segments for")
    return p.parse_args()


def _find_audio_files(qari_filter: str | None, surah_number: int | None = None) -> list[Path]:
    found = []
    for p in WORKSPACE.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in AUDIO_EXTS):
            continue
        if any(part in SEARCH_EXCLUDE for part in p.parts):
            continue
        # only recitations of the requested surah, e.g. 113.mp3 / 113.wav
        if surah_number is not None:
            m = re.search(r"(^|[^0-9])(\d{1,3})\.\w+$", p.name)
            if not m or int(m.group(2)) != surah_number:
                continue
        if qari_filter and qari_filter.lower() not in p.parent.name.lower():
            continue
        found.append(p.resolve())
    return sorted(found, key=lambda p: p.parent.name.lower())


def main() -> None:
    args = _parse_args()
    surah = get_surah_meta(args.surah)
    ref_segments = surah.get("reference_segments")
    if not ref_segments:
        sys.exit(f"No reference_segments defined for surah {args.surah}.")

    files = _find_audio_files(args.qari, surah_number=args.surah)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if not files:
        sys.exit("No audio files matched.")
    print(f"Found {len(files)} recitation file(s) for surah {args.surah} "
          f"({surah.get('name_roman')}), {len(ref_segments)} reference segments.\n")

    print("Loading CT2 Tarteel model (faster-whisper)...")
    aligner = WordAligner(CT2_MODEL_DIR)
    print(f"  device={aligner.device}, compute_type={aligner.compute_type}\n")

    t_start = time.time()
    saved = []
    for src in files:
        qari = src.parent.name
        print(f"--- {qari} / {src.name} ---")
        try:
            result = process_one(src, args, aligner, surah, ref_segments)
        except Exception as exc:
            print(f"  !! ERROR: {exc}")
            continue
        if result is None:
            continue

        out_dir = args.output / _slug(qari)
        out_path = out_dir / f"surah_{args.surah:03d}_{_slug(surah['name_roman'])}__reference_segments.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)

        saved.append((qari, out_path, result))
        print(f"  JSON -> {out_path} ({out_path.stat().st_size / 1024:.0f} KB)\n")

    # ------------------------------------------------ index
    # Rebuild the *combined* index from every reference-segments JSON on disk
    # so one run never wipes the results of another surah (114 + 113 coexist).
    model_info = aligner.info()
    entries = []
    seen = set()
    for js_path in sorted(args.output.glob("*/*reference_segments.json")):
        try:
            with open(js_path, encoding="utf-8") as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            continue
        rel = js_path.relative_to(args.output).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        entries.append(
            {
                "json_file": rel,
                "qari": r["qari"],
                "surah": {
                    "number": r["surah"]["number"],
                    "name_ar": r["surah"].get("name_ar"),
                    "name_roman": r["surah"].get("name_roman"),
                },
                "segments_amount": r["summary"].get("segments_amount"),
                "segments_matched": r["summary"].get("segments_matched"),
                "total_duration_sec": r["summary"].get("total_duration_sec"),
            }
        )
    index = {
        "schema_version": "1.0",
        "project": "Quran Audio Segmenter — reference segments index",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_info,
        "total_files": len(entries),
        "total_segments": sum(e["segments_amount"] for e in entries),
        "files": entries,
    }
    idx = args.output / "ALL_QURAN_REFERENCE_INDEX.json"
    with open(idx, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)

    print("=" * 64)
    print(f"Done in {time.time() - t_start:.1f}s — {len(entries)} qari(s), {index['total_segments']} segments")
    for e in entries:
        print(f"  {e['qari']['name']:<32} | {e['segments_amount']:>2} segs  ({e['segments_matched']} matched) | {e['json_file']}")
    print(f"Index: {idx}")
    print("=" * 64)


def process_one(src: Path, args: argparse.Namespace, aligner: WordAligner, surah: dict, ref_segments: list[dict]) -> dict | None:
    finfo = audio_file_info(src)
    qari = src.parent.name
    qari_key = re.sub(r"^reciter\s+", "", qari.strip().lower())
    qari_ar = QARI_AR_NAMES.get(qari_key) or next((v for k, v in QARI_AR_NAMES.items() if k in qari_key), None)

    # 16 kHz mono for ASR
    y16, sr16 = load_audio(src, sr=16000)
    duration = len(y16) / sr16

    # ---- word-level timestamps ---------------------------------------------
    t0 = time.time()
    words = aligner.global_words(y16, sr16)
    print(f"  words detected: {len(words)} ({time.time() - t0:.1f}s)")

    # ---- match reference phrases -------------------------------------------
    matches = match_phrases_to_words(words, ref_segments)
    n_matched = sum(1 for m in matches if m["matched"])
    print(f"  phrases matched: {n_matched}/{len(ref_segments)}")

    cuts = phrase_cut_times(matches, duration)
    matched_cuts = [c for c in cuts if c["start_sec"] is not None]

    # ---- cut / analyse / save ----------------------------------------------
    out_dir = args.output / _slug(qari)
    seg_dir = out_dir / "segments" / f"surah_{surah['number']:03d}_{_slug(surah['name_roman'])}_ref"
    seg_dir.mkdir(parents=True, exist_ok=True)

    segments_json = []
    for cut in cuts:
        if cut["start_sec"] is None:
            segments_json.append(
                {
                    "index": cut["id"],
                    "id": f"{SEGMENT_PREFIX}{cut['id']:03d}",
                    "audio_segment_file": None,
                    "start_sec": None,
                    "end_sec": None,
                    "duration_sec": None,
                    "transcription": cut["transcription"],
                    "surah_ayah": cut["ayah"],
                    "matched": False,
                    "words": [],
                    "pitch": None,
                    "audio_info": None,
                }
            )
            continue

        s = int(cut["start_sec"] * sr16)
        e = int(cut["end_sec"] * sr16)
        y_seg = y16[s:e]
        seg_file = seg_dir / f"{SEGMENT_PREFIX}{cut['id']:03d}.wav"
        save_wav_pcm16(seg_file, y_seg, sr16)

        y_p, _ = load_audio(seg_file, sr=PITCH_SAMPLE_RATE)
        pitch = extract_pitch_features(
            y_p, PITCH_SAMPLE_RATE,
            fmin=PITCH_FMIN, fmax=PITCH_FMAX,
            hop_length=PITCH_HOP, cents_ref=PITCH_CENTS_REF,
        )
        ainfo = extract_audio_features(y_p, PITCH_SAMPLE_RATE)

        match = next(m for m in matches if m["id"] == cut["id"])
        phrase_words = [words[i] for i in match["word_indices"]]

        segments_json.append(
            {
                "index": cut["id"],
                "id": seg_file.stem,
                "audio_segment_file": seg_file.relative_to(out_dir).as_posix(),
                "start_sec": cut["start_sec"],
                "end_sec": cut["end_sec"],
                "duration_sec": cut["duration_sec"],
                "transcription": cut["transcription"],
                "surah_ayah": cut["ayah"],
                "matched": True,
                "words": [
                    {"text": w["text"], "start_sec": w["start_sec"], "end_sec": w["end_sec"], "probability": w["probability"]}
                    for w in phrase_words
                ],
                "pitch": pitch,
                "audio_info": ainfo,
            }
        )

    # ---- assemble -----------------------------------------------------------
    transcribed_words = [
        {"start_sec": w["start_sec"], "end_sec": w["end_sec"], "text": w["text"], "probability": w["probability"]}
        for w in words
    ]
    return {
        "schema_version": "2.0",
        "project": "Quran Audio Segmenter (reference-based)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qari": {"name": qari, "name_ar": qari_ar, "source_folder": src.parent.name},
        "surah": {k: v for k, v in surah.items() if k != "reference_segments"},
        "audio_file": finfo.to_dict(),
        "model": aligner.info(),
        "segmentation": {
            "method": "reference phrase matching on word-level timestamps (faster-whisper CT2 Tarteel)",
            "reference_segments_amount": len(ref_segments),
            "vad_threshold_db": -38.0,
            "word_min_probability": 0.7,
            "asr_sample_rate": sr16,
        },
        "pitch_settings": {
            "sample_rate": PITCH_SAMPLE_RATE,
            "fmin_hz": PITCH_FMIN,
            "fmax_hz": PITCH_FMAX,
            "hop_length": PITCH_HOP,
            "cents_reference_hz": PITCH_CENTS_REF,
        },
        "summary": {
            "total_duration_sec": round(duration, 4),
            "segments_amount": len(matched_cuts),
            "segments_matched": n_matched,
            "words_total": len(transcribed_words),
            "overall_rms_db": round(rms_db(y16), 2),
            "overall_peak": round(float(np.max(np.abs(y16))), 4),
        },
        "global_words": transcribed_words,
        "segments": segments_json,
    }


if __name__ == "__main__":
    main()