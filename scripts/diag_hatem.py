"""Debug word streaming + phrase matching for specific qaris."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quran_segmenter.audio_io import load_audio
from quran_segmenter.silence_segmenter import segment_by_silence
from quran_segmenter.surah_meta import get_surah_meta
from quran_segmenter.word_aligner import WordAligner, match_phrases_to_words, _tokenize

SR = 16000
CT2 = Path("models/tarteel-whisper-base-ar-quran-ct2")

for qari in ["Hatem Fareed Al Waer", "Khalifa Al Tunaiji"]:
    y, sr16 = load_audio(f"QuranAudios/Reciter {qari}/114.mp3", sr=SR)
    vad = segment_by_silence(y, sr16, threshold_db=-38.0)
    print(f"\n==== {qari}  dur={len(y)/sr16:.1f}s  VAD={len(vad.segments)} segments ====")
    for seg in vad.segments:
        print(f"    seg [{seg['start_sec']:6.2f}-{seg['end_sec']:6.2f}] dur={seg['duration_sec']:5.2f}")

    aligner = WordAligner(CT2)
    words = aligner.global_words(y, sr16)
    print(f"  words ({len(words)}):")
    for i, w in enumerate(words):
        print(f"     {i:2d}  {w['start_sec']:6.2f}-{w['end_sec']:6.2f}  {w['text']}")

    ref_segments = get_surah_meta(114)["reference_segments"]
    matches = match_phrases_to_words(words, ref_segments)
    print("  matches:")
    for m in matches:
        print(f"    id={m['id']:2d} {'OK ' if m['matched'] else '-- '} {m['text']}  words={m.get('word_indices')}")