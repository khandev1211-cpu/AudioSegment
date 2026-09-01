"""Convert the Tarteel AI Whisper model into CTranslate2 format (for faster-whisper).

faster-whisper gives us reliable *word-level* timestamps which we use to cut a
recitation into the exact phrase segments that the client defined (e.g. the
12 An-Nas segments in `refrence/AnNaas`).

Run once:
    python scripts/convert_model.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

WORKSPACE = Path(__file__).resolve().parent.parent
MODEL_SRC = "tarteel-ai/whisper-base-ar-quran"
OUT_DIR = WORKSPACE / "models" / "tarteel-whisper-base-ar-quran-ct2"


def main() -> None:
    from ctranslate2.converters import TransformersConverter

    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        print(f"CT2 model already exists at {OUT_DIR} — skipping conversion.")
    else:
        print(f"Converting {MODEL_SRC} -> {OUT_DIR} ...")
        OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        # NOTE: do NOT copy config.json — the converter writes its own ct2
        # flavoured config.json into the output directory.
        converter = TransformersConverter(
            MODEL_SRC,
            copy_files=["preprocessor_config.json"],
        )
        quant = "int8" if len(sys.argv) > 1 and sys.argv[1] == "int8" else "float16"
        converter.convert(output_dir=str(OUT_DIR), quantization=quant)
        print("Conversion done.")

    # Make sure faster-whisper has a tokenizer to work with.
    token = OUT_DIR / "tokenizer.json"
    if not token.exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_SRC)
        tok.save_pretrained(OUT_DIR)
        print("Tokenizer saved to", OUT_DIR)

    print(f"\nDone. faster-whisper model at {OUT_DIR}")
    print("Files:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}  {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()