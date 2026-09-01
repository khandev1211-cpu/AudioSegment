"""Test faster-whisper word timestamps on the VAD phrase clips of minshawi."""
import numpy as np
import librosa
from faster_whisper import WhisperModel

y, sr = librosa.load("QuranAudios/Muhammad Siddiq Al Minshawi/114.mp3", sr=16000, mono=True)
y = y.astype(np.float32)

CLIPS = [(4.40, 9.30), (9.30, 12.20), (15.20, 20.70), (27.10, 31.70), (32.00, 36.00)]

m = WhisperModel("models/tarteel-whisper-base-ar-quran-ct2", device="cuda", compute_type="float16")

for a, b in CLIPS:
    clip = y[int(a * sr): int(b * sr)]
    pre = np.zeros(int(0.1 * sr), dtype=np.float32)
    segs = list(m.transcribe(np.concatenate([pre, clip]), language="ar", task="transcribe",
                             word_timestamps=True, beam_size=1, condition_on_previous_text=False)[0])
    ws = []
    for s in segs:
        for w in s.words:
            if w.start >= 0.09:  # ignore the padding silence word, if any
                ws.append((w.start - 0.1 + a, w.end - 0.1 + a, w.word, round(w.probability, 2)))
    txt = " ".join(w[2] for w in ws)
    print(f"[{a:5.2f}-{b:5.2f}s] -> {txt[:60]}")
    for st, en, w, p in ws[:14]:
        print(f"      {st:6.2f}-{en:6.2f}  {w:<18} {p}")