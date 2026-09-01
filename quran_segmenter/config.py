"""Central configuration for the Quran Audio Segmenter."""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Model — Tarteel AI Whisper base, downloaded in the local HuggingFace cache.
# Forcing offline means we always use the local copy (no network calls).
# ---------------------------------------------------------------------------
TARTEEL_MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
ASR_LANGUAGE = "arabic"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE / "output"

# ---------------------------------------------------------------------------
# Audio rates
# ---------------------------------------------------------------------------
ASR_SAMPLE_RATE = 16000    # Whisper expects 16 kHz
PITCH_SAMPLE_RATE = 22050  # analysis rate used for pitch / DSP features

# ---------------------------------------------------------------------------
# Segmentation (energy-based VAD)
# ---------------------------------------------------------------------------
SEG_HOP_SEC = 0.01           # RMS analysis frame length (10 ms)
SEG_THRESHOLD_DB = -38.0     # frames quieter than this are treated as silence
SEG_MIN_SILENCE_SEC = 0.30   # a quiet gap of this length splits segments
SEG_MIN_SEGMENT_SEC = 0.60   # drop candidate clips shorter than this
SEG_PAD_SEC = 0.10           # small symmetric padding around each clip

# ---------------------------------------------------------------------------
# Pitch analysis (librosa.pyin)
# ---------------------------------------------------------------------------
PITCH_FMIN = 65.0            # lowest expected f0 (Hz) — male qari safe range
PITCH_FMAX = 500.0           # highest expected f0 (Hz)
PITCH_HOP = 512              # hop length in samples (at PITCH_SAMPLE_RATE)
PITCH_CENTS_REF = 100.0      # reference frequency for the cents representation

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
SEGMENT_FILE_PREFIX = "seg_"