import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from quran_segmenter.audio_io import load_audio
from quran_segmenter.align import normalize_arabic
from quran_segmenter.word_aligner import WordAligner, HALLUC_PROB, HALLUC_TOKENS
from scripts.build_word_clips import asr_words_of, ref_tokens_of, match_ref_to_asr

data = json.load(open("quran_arabic_words.json", encoding="utf-8"))
s113 = data[112]
aligner = WordAligner("models/tarteel-whisper-base-ar-quran-ct2", device="cuda", compute_type="float16")

for v in s113["verses"][:2]:
    y, sr = load_audio(Path("fullQuran/alafasy_quran_complete/113/113%03d.mp3" % v["id"]), sr=16000)
    segments, info = aligner.model.transcribe(
        y, language="ar", task="transcribe", word_timestamps=True,
        beam_size=1, condition_on_previous_text=False, vad_filter=False,
    )
    asr = asr_words_of(segments)
    print("=" * 60)
    print("VERSE", v["id"], "| ref words:", v["words"])
    print("REF norms:", ref_tokens_of(v))
    print("ASR words:", [w["text"] for w in asr])
    print("ASR norms:", [w["norm"] for w in asr])
    print("ali:", match_ref_to_asr(ref_tokens_of(v), asr))
    print("segments text:",
          [sg.text for sg in segments])