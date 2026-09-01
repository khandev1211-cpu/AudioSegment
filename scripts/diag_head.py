"""Diagnostic: audio energy in first seconds + faster-whisper on the head clip."""
import numpy as np
import librosa
from faster_whisper import WhisperModel

y, sr = librosa.load("QuranAudios/Muhammad Siddiq Al Minshawi/114.mp3", sr=16000, mono=True)
y = y.astype(np.float32)
for a, b in [(0, 2), (2, 4), (4, 6), (6, 9), (9, 12)]:
    z = y[a * sr:b * sr]
    rms = np.sqrt(np.mean(z ** 2)) + 1e-10
    print(f"{a}-{b}s  peak={np.max(np.abs(z)):.4f}  rms_db={20*np.log10(rms):.1f}")

print("\n-- faster-whisper on 0-6s (basmalah head) --")
m = WhisperModel("models/tarteel-whisper-base-ar-quran-ct2", device="cuda", compute_type="float16")
head = y[:6 * sr]
segs = list(m.transcribe(head, language="ar", task="transcribe", word_timestamps=True, beam_size=1)[0])
for s in segs:
    print(f"  {s.start:.2f}-{s.end:.2f}  |  {s.text.strip()[:80]}")
    for w in s.words:
        print(f"      {w.start:.2f}-{w.end:.2f}  {w.word:<16} {w.probability:.2f}")