"""Quran Audio Segmenter.

A pipeline that takes recitation audio (e.g. surah 114 An-Nas by qari
Muhammad Siddiq Al-Minshawi), segments it into client-reference phrase clips
(12 phrases: basmalah x2 + An-Nas ayah splits), transcribes with the local
Tarteel AI Whisper model (``tarteel-ai/whisper-base-ar-quran``), extracts
word-level timestamps via faster-whisper / CTranslate2, computes detailed
pitch + audio features and writes one detailed JSON per qari containing the
segments, full pitch contour, audio info, transcriptions and qari name.
"""

from .config import (
    TARTEEL_MODEL_ID,
    OUTPUT_DIR,
    ASR_SAMPLE_RATE,
    PITCH_SAMPLE_RATE,
)
from .audio_io import AudioFileInfo, audio_file_info, load_audio, save_wav_pcm16
from .silence_segmenter import SegmentationResult, segment_by_silence
from .features import extract_audio_features, extract_pitch_features
from .surah_meta import get_surah_meta
from .word_aligner import (
    WordAligner,
    match_phrases_to_words,
    phrase_cut_times,
)

__all__ = [
    "TARTEEL_MODEL_ID",
    "OUTPUT_DIR",
    "ASR_SAMPLE_RATE",
    "PITCH_SAMPLE_RATE",
    "AudioFileInfo",
    "audio_file_info",
    "load_audio",
    "save_wav_pcm16",
    "SegmentationResult",
    "segment_by_silence",
    "extract_audio_features",
    "extract_pitch_features",
    "get_surah_meta",
    "WordAligner",
    "match_phrases_to_words",
    "phrase_cut_times",
]