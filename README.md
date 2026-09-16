# Quran Audio Segmenter

An offline Quran recitation processing toolkit for reference-based phrase segmentation, word-level audio dataset creation, Arabic alignment, and acoustic analysis.

The project uses the Tarteel AI Arabic Quran Whisper model with `faster-whisper` and CTranslate2 for local inference. Segment boundaries come from each reciter's own audio and word timestamps; timings are not copied from a reference recording.

## Capabilities

- Reference-based phrase segmentation for configured surahs
- Energy-based VAD and ayah-oriented segmentation
- Word-level timestamps and normalized Arabic matching
- PYIN pitch analysis and detailed audio features
- PCM16 WAV clip export with JSON, JSONL, and CSV metadata
- Resumable full-Quran word-level dataset generation
- Offline inference after dependencies and models are installed

## Requirements and Setup

- Python 3.10 or newer
- FFmpeg available on `PATH` for MP3/M4A decoding
- NVIDIA CUDA GPU recommended for practical processing speed
- Local Tarteel Whisper model in Hugging Face or CTranslate2 format

```bash
python -m pip install -r requirements.txt

# Convert the model once, if the CT2 directory is not already present
python scripts/convert_model.py
```

The converted model is expected at `models/tarteel-whisper-base-ar-quran-ct2/`. The project sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` by default, so inference does not make network requests.

## Input Layout

Recitation files should be grouped by reciter. The filename should contain the Surah number:

```text
QuranAudios/
├── Abdul Basit Abdul Samad/
│   ├── 113.mp3
│   └── 114.mp3
└── Saad Al Ghamdi/
   └── 114.mp3
```

Supported formats include MP3, WAV, FLAC, OGG, M4A, AAC, and OPUS.

## Reference-Based Segmentation

Use `reference_segment.py` when output must follow the configured phrase groups. Reference definitions currently live in `quran_segmenter/surah_meta.py`.

```bash
python reference_segment.py
```

Examples:

```bash
python reference_segment.py --qari minshawi
python reference_segment.py --surah 113
python reference_segment.py --surah 113 --qari hatem --limit 1
python reference_segment.py --output custom_output
```

The reference pipeline detects voiced regions, extracts global word timestamps, matches Arabic reference phrases, cuts boundaries between matched words, and writes audio plus detailed metadata.

## General VAD Segmentation

Use `main.py` when a fixed phrase reference is not required:

```bash
python main.py
python main.py --qari "Al Ghamdi"
python main.py --input "QuranAudios/Saad Al Ghamdi/114.mp3"
python main.py --no-transcribe
python main.py --threshold-db -40 --min-silence 0.35 --min-segment 0.60 --pad 0.10
```

This path automatically segments recordings at quiet gaps and attempts to label segments with Surah ayah numbers.

## Output Structure

```
output/
├── ALL_QURAN_REFERENCE_INDEX.json
├── muhammad-siddiq-al-minshawi/
│   ├── surah_114_an-nas__reference_segments.json
│   └── segments/surah_114_an_nas_ref/
│       ├── seg_001.wav
│       ├── seg_002.wav
│       └── ...
└── reciter-al-shatri/
   └── ...
```

The full-Quran word-level dataset is written to `dataset/word_clips/`:

```text
dataset/word_clips/
├── 001/001001/word_000.wav
├── manifest.jsonl
├── manifest_tmp.jsonl
├── progress.json
└── summary.json
```

Enriched exports are generated at `dataset/words_with_clips.csv`, `dataset/words_with_clips.jsonl`, `dataset/verses_with_clips.csv`, and `dataset/verses_with_clips.jsonl`.

## Full-Quran Word Dataset

The word-clip builder uses the canonical word list in `quran_arabic_words.json` and the corresponding Alafasy verse audio. It transcribes each verse, matches ASR words to the reference words, and writes one PCM16 WAV clip per matched word.

```bash
python -u scripts/build_word_clips.py --device cuda --compute-type float16
```

Targeted or CPU runs:

```bash
python scripts/build_word_clips.py --surah 114
python scripts/build_word_clips.py --limit 20
python scripts/build_word_clips.py --device cpu --compute-type int8
```

The process is resumable and checkpoints after each verse. Monitor or snapshot it with:

```bash
python scripts/word_clips_status.py
python scripts/snapshot_word_dataset.py
```

The snapshot command rebuilds the canonical manifest, summary, and enriched dataset exports from the incremental checkpoint.

## 12 Reference Segments (Surah An-Nas)

| # | Ayah | Text | Reference duration |
|---|------|------|--------------------|
| 1 | Basmalah | بِسْمِ اللَّهِ | 1.49s |
| 2 | Basmalah | الرَّحْمَٰنِ الرَّحِيمِ | 3.03s |
| 3 | 1 | قُلْ أَعُوذُ | 1.31s |
| 4 | 1 | بِرَبِّ النَّاسِ | 2.95s |
| 5 | 2 | مَلِكِ النَّاسِ | 2.98s |
| 6 | 3 | إِلَٰهِ النَّاسِ | 2.77s |
| 7 | 4 | مِنْ شَرِّ | 1.75s |
| 8 | 4 | الْوَسْوَاسِ الْخَنَّاسِ | 3.32s |
| 9 | 5 | الَّذِي يُوَسْوِسُ | 2.30s |
| 10 | 5 | فِي صُدُورِ النَّاسِ | 3.37s |
| 11 | 6 | مِنَ الْجِنَّةِ | 2.35s |
| 12 | 6 | وَالنَّاسِ | 3.03s |

## 21 Reference Segments (Surah Al-Falaq, 113) — word-level

Client ke spec (JSON) ke mutabiq — 5 ayaat ke **word-groups** (koi basmalah nahi):

| # | Ayah | Text | Duration* |
|---|------|------|-----------|
| 1 | 1 | قُلْ | 0.73s |
| 2 | 1 | أَعُوذُ | 1.35s |
| 3 | 1 | بِرَبِّ الْفَلَقِ | 4.77s |
| 4 | 2 | مِن | 1.37s |
| 5 | 2 | شَرِّ | 1.27s |
| 6 | 2 | مَا | 0.63s |
| 7 | 2 | خَلَقَ | 2.85s |
| 8 | 3 | وَمِن | 1.77s |
| 9 | 3 | شَرِّ | 1.25s |
| 10 | 3 | غَاسِقٍ | 2.09s |
| 11 | 3 | إِذَا | 1.37s |
| 12 | 3 | وَقَبَ | 2.59s |
| 13 | 4 | وَمِن | 1.73s |
| 14 | 4 | شَرِّ النَّفَّاثَاتِ | 5.11s |
| 15 | 4 | فِي | 0.37s |
| 16 | 4 | الْعُقَدِ | 3.28s |
| 17 | 5 | وَمِن | 2.13s |
| 18 | 5 | شَرِّ | 1.21s |
| 19 | 5 | حَاسِدٍ | 2.13s |
| 20 | 5 | إِذَا | 1.27s |
| 21 | 5 | حَسَدَ | 1.21s |

\* Durations are examples from Abdul Basit's clips in `refrence/AnFalaq/01–21.wav`.
The reference specification excludes basmalah from the Surah 113 segments, so any
basmalah audio remains outside these 21 clips. Boundaries use each reciter's own
ASR word timestamps.

## JSON Metadata

| Section | Details |
|---|---|
| `qari` | Reciter name, Arabic name when known, and source folder |
| `surah` | Number, Arabic/Roman name, meaning, and ayah count |
| `audio_file` | Format, sample rate, channels, codec, duration, and size |
| `model` | Inference engine, model, device, and data type |
| `segments[].transcription` | Segment transcription or reference phrase |
| `segments[].words` | Word-level timestamps and confidence values |
| `segments[].audio_info` | RMS, peak, crest factor, zero-crossing, and spectral features |
| `segments[].pitch` | PYIN contour, confidence, statistics, cents, and pitch histogram |
| `segments[].surah_ayah` | Basmalah or ayah label when alignment succeeds |
| `global_words` | Word timestamps for the complete recitation |

## Validation Results

| Reciter | Matched |
|------|---------|
| Muhammad Siddiq Al Minshawi | 12/12 |
| Abdul Basit Abdul Samad | 12/12 |
| Al Shatri | 12/12 |
| Ibrahim Al-Akhdar | 12/12 |
| Khalid Al Jalil | 12/12 |
| Saad Al Ghamdi | 12/12 |
| Ahmed El Agamy | 10/12 *(basmalah nahi hai recording mein)* |
| Hatem Fareed Al Waer | 10/12 *(basmalah nahi)* |
| Khalifa Al Tunaiji | 10/12 *(basmalah nahi)* |
| Salah Bukhatir | 10/12 *(basmalah nahi)* |
| Saud Al Shuraim | 10/12 *(basmalah nahi)* |

Recordings that begin directly with `قُلْ أَعُوذُ` do not contain the two basmalah segments, so those segments correctly remain unmatched.

## Surah 113 Validation

| Reciter | Matched |
|------|---------|
| Abdul Basit Abdul Samad | 21/21 |
| Ahmed El Agamy | 21/21 |
| Al Shatri | 21/21 |
| Bandar Balila | 21/21 |
| Hatem Fareed Al Waer | 21/21 |
| Ibrahim Al-Akhdar | 21/21 |
| Khalifa Al Tunaiji | 21/21 |
| Saad Al Ghamdi | 21/21 |
| Salah Bukhatir | 21/21 |
| Saud Al Shuraim | 21/21 |

Each reciter produces `output/<qari>/segments/surah_113_al-falaq_ref/seg_001..021.wav` and a corresponding JSON metadata file.

## Processing Details

1. VAD splits audio at quiet gaps.
2. Faster-Whisper generates word-level timestamps.
3. Arabic text is normalized by removing diacritics and unifying letter forms.
4. Reference phrases are matched against the global word timeline.
5. Boundaries are cut between adjacent matched words.
6. PYIN pitch and audio features are extracted for each saved clip.
7. WAV clips and detailed JSON metadata are written to disk.

## Project Structure

```
├── reference_segment.py          Reference phrase segmentation entry point
├── main.py                       General VAD-based segmentation entry point
├── annas_finetune_colab_updated (1).ipynb
├── scripts/convert_model.py      Hugging Face to CTranslate2 conversion
├── quran_segmenter/
│   ├── config.py                 Model, sample-rate, and VAD settings
│   ├── audio_io.py               Audio loading, metadata, and WAV export
│   ├── silence_segmenter.py      Energy-based VAD
│   ├── word_aligner.py           Word timestamps and phrase matching
│   ├── features.py               PYIN pitch and audio features
│   ├── align.py                  Arabic normalization and ayah alignment
│   ├── transcriber.py            Transformers-based transcription
│   └── surah_meta.py             Surah and reference metadata
└── models/tarteel-whisper-base-ar-quran-ct2/
```

## Quality and Verification

Unmatched reference words remain in the manifest with `matched: false` and `clip_file: null`; rows are not silently discarded.

Useful checks:

```bash
python scripts/verify_full_quran_dataset.py
python scripts/analyze_quran_json.py
python scripts/word_clips_status.py
```

For model training, split word clips by verse rather than by individual word. This prevents words from the same verse appearing in both training and validation data.

## Known Limitations

- Reference phrase definitions are available only for surahs configured in `quran_segmenter/surah_meta.py`.
- ASR and greedy Arabic matching can leave words unmatched, especially with noisy audio or pronunciation differences.
- VAD thresholds may need adjustment for different recordings and noise floors.
- Generated clips and local model files require substantial disk space and should generally remain outside source control.

## Maintainer

**Irfan Khan**
[WhatsApp](https://wa.me/923433791141) · [khandev1211@gmail.com](mailto:khandev1211@gmail.com)
