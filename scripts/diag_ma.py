"""Check raw faster-whisper word output for Surah 113 (does 'ما' get emitted?)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
from faster_whisper import WhisperModel

qari = sys.argv[1] if len(sys.argv) > 1 else "Reciter Bandar Balila"
y, sr = librosa.load(f"Surah/113/{qari}/113.mp3", sr=16000, mono=True)
y = y.astype(np.float32)

m = WhisperModel("models/tarteel-whisper-base-ar-quran-ct2", device="cuda", compute_type="float16")
segs = list(m.transcribe(y, language="ar", task="transcribe", word_timestamps=True, beam_size=1,
                         condition_on_previous_text=False)[0])
print(f"== {qari} ==")
for sg in segs:
    print("SEG:", sg.text.strip())
    for w in sg.words:
        if "ما" in w.word or "مَا" in w.word:
            print(f"  ** 'ما' found: {w.start:.2f}-{w.end:.2f}  '{w.word}'  p={w.probability:.4f}")
    for w in sg.words:
        print(f"     {w.start:.2f}-{w.end:.2f}  '{w.word}'  p={w.probability:.3f}")