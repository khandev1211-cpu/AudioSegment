"""Diagnose: per-VAD-segment faster-whisper word timestamps, merged to global times.

This is the core mechanism for reference-segment cutting:
1. VAD splits the recording at silence gaps.
2. Each VAD clip is transcribed alone (leading/trailing pad avoids boundary artifacts).
3. Word timestamps are shifted back to global time and merged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import librosa
from faster_whisper import WhisperModel

from quran_segmenter.silence_segmenter import segment_by_silence

SR = 16000
PAD = 0.12

y, sr = librosa.load("QuranAudios/Muhammad Siddiq Al Minshawi/114.mp3", sr=SR, mono=True)
y = y.astype(np.float32)

vad = segment_by_silence(y, sr, threshold_db=-38.0)
print(f"VAD segments: {len(vad.segments)}")

m = WhisperModel("models/tarteel-whisper-base-ar-quran-ct2", device="cuda", compute_type="float16")

all_words = []
for seg in vad.segments:
    s = seg["start_sec"]
    e = seg["end_sec"]
    s_pad = max(0.0, s - PAD)
    e_pad = min(len(y) / sr, e + PAD)
    clip = y[int(s_pad * sr): int(e_pad * sr)]
    segs = list(m.transcribe(clip, language="ar", task="transcribe",
                             word_timestamps=True, beam_size=1, condition_on_previous_text=False)[0])
    text = " ".join(sg.text.strip() for sg in segs)
    print(f"\nVAD [{s:6.2f}-{e:6.2f}] -> {text[:70]}")
    for sg in segs:
        for w in sg.words:
            if w.probability < 0.5:
                continue
            gs = w.start - PAD + s_pad
            ge = w.end - PAD + s_pad
            all_words.append((gs, ge, w.word.strip(), round(w.probability, 2)))

all_words.sort(key=lambda t: t[0])

# de-dup overlapping (same word text within 0.4s = pad duplicate)
merged = []
for gs, ge, wtext, p in all_words:
    if merged and merged[-1][0] < gs and gs < merged[-1][1] and wtext == merged[-1][2]:
        continue
    merged.append((gs, ge, wtext, p))

print("\n=== merged word list ({} words) ===".format(len(merged)))
for i, (gs, ge, wtext, p) in enumerate(merged):
    print(f"{i:3d}  {gs:6.2f}-{ge:6.2f}  {wtext:<20} {p}")