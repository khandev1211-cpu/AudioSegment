"""Export one qari's matched reference-segment clips to ``refrence/AnFalaq/``.

Surah 114's client reference set lives in ``refrence/AnNaas/1..12.mp3`` (audio
provided by the client). Surah 113 has no client audio yet, so we *bootstrap*
the reference set ``refrence/AnFalaq/1..11.wav`` from the pipeline's own cut
clips — the phrase text and boundaries are identical for every qari; only the
performer differs, so these clips serve as the reference deliverable until the
client provides their own audio.

Usage:
    python scripts/export_reference_clips.py                 # Abdul Basit (11/11)
    python scripts/export_reference_clips.py --qari reciter-al-shatri
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE / "output"
REF_DIR = WORKSPACE / "refrence" / "AnFalaq"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap Surah 113 reference clips from segmented output")
    p.add_argument("--surah", type=int, default=113)
    p.add_argument("--qari", default="reciter-abdul-basit-abdul-samad", help="qari slug to export from")
    p.add_argument("--output-dir", type=Path, default=REF_DIR)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    surah_num = args.surah

    json_path = next(OUTPUT_DIR.glob(f"{args.qari}/surah_{surah_num:03d}*reference_segments.json"), None)
    if json_path is None:
        raise SystemExit(f"No reference-segments JSON found under output/{args.qari}/ for surah {surah_num}.")

    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_copied = 0
    for s in data["segments"]:
        if not s.get("matched") or not s.get("audio_segment_file"):
            rows.append((s["index"], "skipped (not matched) — likely no basmalah", s["transcription"], s["surah_ayah"]))
            continue
        src = json_path.parent / Path(s["audio_segment_file"])
        if not src.is_file():
            rows.append((s["index"], "skipped (wav missing)", s["transcription"], s["surah_ayah"]))
            continue
        dst = out_dir / f"{s['index']:02d}.wav"
        shutil.copyfile(src, dst)
        n_copied += 1
        rows.append((s["index"], f"{dst.name}  ({s['duration_sec']:.2f}s)", s["transcription"], s["surah_ayah"]))

    readme = out_dir / "README.txt"
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write(f"# Surah {data['surah']['number']} — {data['surah']['name_roman']} reference segments\n")
        fh.write(f"Source qari: {data['qari']['name']}  ({json_path.name})\n\n")
        fh.write(f"{'id':>3}  {'file':<22} {'phrase':<28} ayah\n")
        fh.write("-" * 72 + "\n")
        for i, fname, text, ayah in rows:
            fh.write(f"{i:>3}  {fname:<22} {text[:26]:<28} {ayah}\n")
    print(f"Copied {n_copied} reference clips -> {out_dir}")
    print(f"Manifest -> {readme}")


if __name__ == "__main__":
    main()