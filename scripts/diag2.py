"""Diagnose why faster-whisper skips the basmalah at t≈0."""
import numpy as np
import librosa
import soundfile as sf

y, sr = librosa.load("QuranAudios/Muhammad Siddiq Al Minshawi/114.mp3", sr=16000, mono=True)
y = y.astype(np.float32)

# exact clip that transformers transcribed as basmalah: 0.00 - 4.41s
clip = y[0 : int(4.41 * sr)]
sf.write("scripts/_seg1_041.wav", clip, sr)

from faster_whisper import WhisperModel

m = WhisperModel("models/tarteel-whisper-base-ar-quran-ct2", device="cuda", compute_type="float16")

def run(tag, audio, **kw):
    segs = list(m.transcribe(audio, language="ar", task="transcribe", word_timestamps=True, beam_size=1, **kw)[0])
    print(f"--- {tag}: {' '.join(s.text.strip() for s in segs)[:80]}")
    for s in segs:
        for w in s.words:
            print(f"     {w.start:5.2f}-{w.end:5.2f}  {w.word:<18} {w.probability:.2f}")

run("4.41s clip (default)", clip)
run("4.41s clip (condition_off)", clip, condition_on_previous_text=False)
run("4.41s clip (no_speech=0.0)", clip, no_speech_threshold=0.0)
run("4.41s clip (vad_filter on)", clip, vad_filter=True)
run("4.41s clip (beam5)", clip, beam_size=5, condition_on_previous_text=False)

# silence-padded version: put 1.0s of silence before the basmalah
pad = np.zeros(int(1.0 * sr), dtype=np.float32)
run("silence(1s)+basmalah", np.concatenate([pad, clip]))