"""Cross-check: transformers vs faster-whisper on the head clip (0-6s)."""
import time
import numpy as np
import librosa

audio, sr = librosa.load("QuranAudios/Muhammad Siddiq Al Minshawi/114.mp3", sr=16000, mono=True)
head = audio[: 6 * sr].astype(np.float32)

print("== faster-whisper ==")
from faster_whisper import WhisperModel

m = WhisperModel("models/tarteel-whisper-base-ar-quran-ct2", device="cuda", compute_type="float16")
t0 = time.time()
segs = list(m.transcribe(head, language="ar", task="transcribe", beam_size=1, word_timestamps=True)[0])
print(f"  ({time.time()-t0:.2f}s) ->", " | ".join(s.text.strip() for s in segs))

print("== transformers (Tarteel) on same clip ==")
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

p = WhisperProcessor.from_pretrained("tarteel-ai/whisper-base-ar-quran")
model = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-base-ar-quran").to("cuda").to(torch.float16)
model.eval()
import re
gc = model.generation_config
if not getattr(gc, "is_multilingual", None):
    gc.is_multilingual = True
if not getattr(gc, "lang_to_id", None):
    vocab = p.tokenizer.get_vocab()
    gc.lang_to_id = {t: i for t, i in vocab.items() if re.fullmatch(r"<\|[a-z]{2}\|>", t)}
if not getattr(gc, "task_to_id", None):
    v = p.tokenizer.get_vocab()
    gc.task_to_id = {"transcribe": v.get("<|transcribe|>"), "translate": v.get("<|translate|>")}
if not getattr(gc, "no_timestamps_token_id", None):
    gc.no_timestamps_token_id = p.tokenizer.get_vocab().get("<|notimestamps|>")

feat = p(head, sampling_rate=16000, return_tensors="pt").input_features
with torch.inference_mode():
    for bs, temp in [(1, 0), (4, 0)]:
        try:
            out = model.generate(feat.to("cuda"), language="arabic", task="transcribe", num_beams=bs, temperature=temp,
                                 no_speech_threshold=0.6, compression_ratio_threshold=2.4, logprob_threshold=-1.0)
        except TypeError:
            out = model.generate(feat.to("cuda"), language="arabic", task="transcribe", num_beams=bs, temperature=temp)
        txt = p.batch_decode(out, skip_special_tokens=True)[0].strip()
        print(f"  beam={bs} temp={temp} -> {txt[:80]}")