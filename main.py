"""Quran Audio Segmenter — main entrypoint.

Scans the workspace for Quran recitation audio (one ``.mp3`` per surah inside
a per-qari folder), splits every file into ayah segments with an energy VAD,
transcribes each segment with the local Tarteel AI Whisper model
(``tarteel-ai/whisper-base-ar-quran``), computes detailed pitch + audio
features, aligns transcriptions to ayah numbers, and writes one detailed JSON
per qari/surah plus a combined index.

Usage:
    python main.py                              # process ALL qaris found
    python main.py --qari "Al Ghamdi"           # filter by qari (substring)
    python main.py --input "QuranAudios/<Qari>/114.mp3"
    python main.py --no-transcribe              # skip ASR (faster)
    python main.py --limit 2                    # only first N files
    python main.py --threshold-db -40 --min-silence 0.35
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

from quran_segmenter import (
    ASR_SAMPLE_RATE,
    OUTPUT_DIR,
    PITCH_SAMPLE_RATE,
    audio_file_info,
    extract_audio_features,
    extract_pitch_features,
    get_surah_meta,
    load_audio,
    save_wav_pcm16,
    segment_by_silence,
)
from quran_segmenter.align import align_segments_to_ayahs
from quran_segmenter.audio_io import rms_db
from quran_segmenter.config import (
    PITCH_CENTS_REF,
    PITCH_FMAX,
    PITCH_FMIN,
    PITCH_HOP,
    SEGMENT_FILE_PREFIX,
    SEG_HOP_SEC,
    SEG_MIN_SEGMENT_SEC,
    SEG_MIN_SILENCE_SEC,
    SEG_PAD_SEC,
    SEG_THRESHOLD_DB,
    TARTEEL_MODEL_ID,
)
from quran_segmenter.transcriber import TarteelTranscriber

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
SEARCH_EXCLUDE = {"output", "quran_segmenter", "segments", "venv", ".git", "__pycache__"}

QARI_AR_NAMES = {
    "muhammad siddiq al minshawi": "محمد صديق المنشاوي",
    "abdul basit abdul samad": "عبد الباسط عبد الصمد",
    "ahmed el agamy": "أحمد العجمي",
    "al shatri": "أبو بكر الشاطري",
    "hatem fareed al waer": "حاتم فريد الواعر",
    "ibrahim al-akhdar": "إبراهيم الأخضر",
    "khalid al jalil": "خالد الجليل",
    "khalifa al tunaiji": "خليفة الطنيجي",
    "saad al ghamdi": "سعد الغامدي",
    "salah bukhatir": "صلاح بوخاطر",
    "saud al shuraim": "سعود الشريم",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quran audio segmentation + detailed JSON export")
    p.add_argument("--input", help="single audio file (default: process every recitation found)")
    p.add_argument("--qari", help="only process qaris whose folder name contains this substring")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR, help="output root directory")
    p.add_argument("--limit", type=int, default=0, help="process at most N files (0 = all)")
    p.add_argument("--threshold-db", type=float, default=SEG_THRESHOLD_DB)
    p.add_argument("--min-silence", type=float, default=SEG_MIN_SILENCE_SEC)
    p.add_argument("--min-segment", type=float, default=SEG_MIN_SEGMENT_SEC)
    p.add_argument("--pad", type=float, default=SEG_PAD_SEC)
    p.add_argument("--no-transcribe", action="store_true", help="skip ASR (JSON has empty transcriptions)")
    return p.parse_args()


def _surah_number_from_filename(name: str) -> int:
    m = re.search(r"(^|[^0-9])(\d{1,3})([^0-9]|$)", name)
    return int(m.group(2)) if m else 0


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "unnamed"


def _arabic_qari_name(name: str) -> str | None:
    key = re.sub(r"^reciter\s+", "", name.strip().lower())
    if key in QARI_AR_NAMES:
        return QARI_AR_NAMES[key]
    for k, v in QARI_AR_NAMES.items():
        if k in key or key in k:
            return v
    return None


def _find_audio_files(workspace: Path, cli_input: str | None, qari_filter: str | None) -> list[Path]:
    if cli_input:
        candidate = Path(cli_input)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        if not candidate.is_file():
            sys.exit(f"Input file not found: {candidate}")
        return [candidate.resolve()]

    found = []
    for p in workspace.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in AUDIO_EXTS):
            continue
        if any(part in SEARCH_EXCLUDE for part in p.parts):
            continue
        if qari_filter and qari_filter.lower() not in p.parent.name.lower():
            continue
        found.append(p.resolve())

    found = sorted(found, key=lambda p: (p.parent.name.lower(), p.name.lower()))
    if not found:
        sys.exit("No matching audio files found.")
    return found


def process_single(
    src: Path,
    args: argparse.Namespace,
    transcriber: TarteelTranscriber | None,
) -> dict:
    """Segment one recitation file, extract features and assemble its JSON."""
    workspace = Path(__file__).resolve().parent
    finfo = audio_file_info(src)
    qari = src.parent.name
    qari_slug = _slug(qari)
    surah_num = _surah_number_from_filename(src.name)
    surah = get_surah_meta(surah_num)

    y_orig, orig_sr = load_audio(src, sr=None)
    print(f"  load {src.name}   {len(y_orig)/orig_sr:.2f}s @ {orig_sr} Hz")

    seg_res = segment_by_silence(
        y_orig, orig_sr,
        hop_sec=SEG_HOP_SEC,
        threshold_db=args.threshold_db,
        min_silence_sec=args.min_silence,
        min_segment_sec=args.min_segment,
        pad_sec=args.pad,
    )
    n_seg = len(seg_res.segments)
    if n_seg == 0:
        print(f"  !! no segments for {src.name}, skipping")
        return None

    # ------------------------------------------------------- per segment
    surah_dir_slug = f"surah_{surah_num:03d}_{_slug(surah['name_roman'] or 'unknown')}"
    out_dir = args.output / qari_slug
    seg_dir = out_dir / "segments" / surah_dir_slug
    seg_dir.mkdir(parents=True, exist_ok=True)

    segments_json = []
    for seg in seg_res.segments:
        s, e = seg["start_sample"], seg["end_sample"]
        y_seg = y_orig[s:e]

        y_p = librosa_resample(y_seg, orig_sr, PITCH_SAMPLE_RATE)
        pitch = extract_pitch_features(
            y_p, PITCH_SAMPLE_RATE,
            fmin=PITCH_FMIN, fmax=PITCH_FMAX,
            hop_length=PITCH_HOP, cents_ref=PITCH_CENTS_REF,
        )
        ainfo = extract_audio_features(y_p, PITCH_SAMPLE_RATE)

        seg_id = f"{SEGMENT_FILE_PREFIX}{seg['index']:03d}"
        seg_file = seg_dir / f"{seg_id}.wav"

        y_asr = librosa_resample(y_seg, orig_sr, ASR_SAMPLE_RATE)
        text = transcriber.transcribe(y_asr) if transcriber is not None else ""
        save_wav_pcm16(seg_file, y_asr, ASR_SAMPLE_RATE)

        segments_json.append(
            {
                "index": seg["index"],
                "id": seg_id,
                "audio_segment_file": seg_file.relative_to(out_dir).as_posix(),
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "duration_sec": seg["duration_sec"],
                "start_sample": seg["start_sample"],
                "end_sample": seg["end_sample"],
                "transcription": text,
                "surah_ayah": None,
                "audio_info": ainfo,
                "pitch": pitch,
            }
        )

    # -------------------------------------------- ayah alignment
    ayahs_text = surah.get("ayahs_text")
    if ayahs_text:
        labels = align_segments_to_ayahs([s["transcription"] for s in segments_json], ayahs_text)
        for s, lab in zip(segments_json, labels):
            s["surah_ayah"] = lab

    # -------------------------------------------- full audio transcription
    overall_text = ""
    if transcriber is not None:
        y_asr_full = librosa_resample(y_orig, orig_sr, ASR_SAMPLE_RATE)
        overall_text = transcriber.transcribe(y_asr_full)

    model_info = transcriber.info() if transcriber else {
        "id": TARTEEL_MODEL_ID,
        "loaded_from": "skipped (--no-transcribe)",
        "device": None,
    }

    return {
        "schema_version": "1.1",
        "project": "Quran Audio Segmenter",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qari": {
            "name": qari,
            "name_ar": _arabic_qari_name(qari),
            "source_folder": src.parent.name,
        },
        "surah": surah,
        "audio_file": finfo.to_dict(),
        "model": model_info,
        "segmentation_settings": {
            "method": seg_res.to_dict()["method"],
            "hop_sec": SEG_HOP_SEC,
            "threshold_db": args.threshold_db,
            "min_silence_sec": args.min_silence,
            "min_segment_sec": args.min_segment,
            "pad_sec": args.pad,
            "asr_sample_rate": ASR_SAMPLE_RATE,
        },
        "pitch_settings": {
            "sample_rate": PITCH_SAMPLE_RATE,
            "fmin_hz": PITCH_FMIN,
            "fmax_hz": PITCH_FMAX,
            "hop_length": PITCH_HOP,
            "cents_reference_hz": PITCH_CENTS_REF,
        },
        "ayah_alignment": {
            "method": "normalized-Arabic greedy forward matching",
            "reference": surah.get("ayahs_text"),
            "note": "segments can map to 'basmalah', an ayah number, or None (unmatched)",
        },
        "summary": {
            "total_duration_sec": round(len(y_orig) / orig_sr, 4),
            "segments_amount": n_seg,
            "overall_rms_db": round(rms_db(y_orig), 2),
            "overall_peak": round(float(np.max(np.abs(y_orig))), 4),
            "transcription_full_audio": overall_text,
        },
        "segments": segments_json,
    }


def main() -> None:
    args = _parse_args()
    workspace = Path(__file__).resolve().parent
    files = _find_audio_files(workspace, args.input, args.qari)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    print(f"Found {len(files)} recitation file(s)")
    for f in files:
        print(f"  - {f.parent.name} / {f.name}")

    # One model load shared by every file.
    transcriber = None
    if not args.no_transcribe:
        print(f"\nLoading Tarteel model ({TARTEEL_MODEL_ID}) ...")
        transcriber = TarteelTranscriber()
        print(f"  model on {transcriber.device} ({transcriber.info()['dtype']}), load {transcriber.load_sec}s")

    t_start = time.time()
    per_qari = {}          # qari_slug -> list of result dicts
    saved_json = []        # (qari, qari_slug, surah_name, relative json path, result, out_dir)
    for src in files:
        print(f"\n--- {src.parent.name} / {src.name} ---")
        result = process_single(src, args, transcriber)
        if result is None:
            continue

        surah_num = result["surah"]["number"]
        surah_slug = _slug(result["surah"].get("name_roman") or "unknown")
        qari_slug = _slug(result["qari"]["name"])
        out_dir = args.output / qari_slug
        out_path = out_dir / f"surah_{surah_num:03d}_{surah_slug}__segments.json"

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)

        per_qari.setdefault(qari_slug, []).append(result)
        saved_json.append((result, out_path, out_dir))

        # per-file console summary
        for s in result["segments"]:
            st_hz = (s["pitch"].get("statistics_hz") or {}).get("median")
            med = f"{st_hz}Hz" if st_hz is not None else "—"
            print(f"  [{s['index']:>2}] {s['start_sec']:>7.2f}-{s['end_sec']:>7.2f}s "
                  f"ayah={s['surah_ayah']}  f0_med={med}  \"{s['transcription'][:40]}\"")
        print(f"  JSON -> {out_path} ({out_path.stat().st_size/1024:.0f} KB)")

    # ------------------------------------------------------------- index files
    elapsed = time.time() - t_start
    index_entries = []
    for result, out_path, out_dir in saved_json:
        index_entries.append(
            {
                "json_file": out_path.relative_to(args.output).as_posix(),
                "qari": result["qari"],
                "surah": {
                    "number": result["surah"]["number"],
                    "name_ar": result["surah"].get("name_ar"),
                    "name_roman": result["surah"].get("name_roman"),
                },
                "segments_amount": result["summary"]["segments_amount"],
                "total_duration_sec": result["summary"]["total_duration_sec"],
                "transcription_full_audio": result["summary"]["transcription_full_audio"],
            }
        )

    model_info = transcriber.info() if transcriber else {
        "id": TARTEEL_MODEL_ID, "loaded_from": "skipped (--no-transcribe)", "device": None,
    }
    index_payload = {
        "schema_version": "1.0",
        "project": "Quran Audio Segmenter — index",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_info,
        "total_files": len(index_entries),
        "total_segments": sum(e["segments_amount"] for e in index_entries),
        "files": index_entries,
    }
    idx_path = args.output / "ALL_QURAN_INDEX.json"
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(index_payload, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------- final summary
    print("\n" + "=" * 64)
    print(f"Done in {elapsed:.1f}s — {len(index_entries)} file(s), "
          f"{index_payload['total_segments']} segments")
    for e in index_entries:
        name = e["qari"].get("name_ar") or e["qari"]["name"]
        print(f"  {e['qari']['name']:<32} | {e['surah']['number']:>3} {e['surah']['name_roman']:<12} "
              f"| {e['segments_amount']:>2} segs | {e['json_file']}")
    print(f"Index: {idx_path}")
    print("=" * 64)


def librosa_resample(y: np.ndarray, orig: int, target: int) -> np.ndarray:
    import librosa
    if orig == target:
        return y
    return librosa.resample(y, orig_sr=orig, target_sr=target).astype(np.float32)


if __name__ == "__main__":
    main()