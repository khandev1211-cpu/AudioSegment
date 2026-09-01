"""Diagnose a qari with low word detection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import librosa

from quran_segmenter.audio_io import load_audio
from quran_segmenter.silence_segmenter import segment_by_silence

SR = 16000
TARGET = "Ahmed El Agamy"

y, sr = load_audio(f"QuranAudios/Reciter {TARGET}/114.mp3", sr=SR)
vad = segment_by_silence(y, sr, threshold_db=-38.0)
print(f"VAD segments: {len(vad.segments)}")
for seg in vad.segments:
    dur = seg["end_sec"] - seg["start_sec"]
    print(f"  [{seg['start_sec']:7.2f}-{seg['end_sec']:7.2f}]  dur={dur:6.2f}s")

# energy in first 6s
for a, b in [(0, 2), (2, 4), (4, 6), (6, 9)]:
    z = y[a * sr:b * sr]
    rms = 20 * np.log10(np.sqrt(np.mean(z ** 2)) + 1e-10)
    print(f"  RMS[{a}-{b}s] = {rms:.1f} dB")