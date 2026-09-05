# 📖 Quran Audio Segmenter — Reference-Based

Har qari ki Quran recitation (`QuranAudios/<Qari>/114.mp3`) ko **client ke reference
12 phrases** (`refrence/AnNaas` — basmalah 2 parts + An-Nas ke 10 phrase splits)
ke mutabiq cut karta hai, aur ek **detailed JSON** banata hai jis mein har segment
ki **pitch**, **audio info**, **word-level timestamps**, **transcription** aur
**qari ka naam** hota hai.

Model base: **Tarteel AI** `tarteel-ai/whisper-base-ar-quran` (locally downloaded),
CTranslate2 (faster-whisper) format mein convert karke **word-level timestamps**
nikalte hain — is liye har qari ki apni timing use hoti hai, reference timing nahi.

## Setup (pehli dafa)

```bash
pip install -r requirements.txt

# Tarteel model ko CTranslate2 format mein convert karo (1 dafa)
python scripts/convert_model.py
```

> Model already `C:\Users\CHAND COMPUTER\.cache\huggingface\hub\models--tarteel-ai--whisper-base-ar-quran`
> mein downloaded hai. `HF_HUB_OFFLINE=1` set hai, network ki koi zaroorat nahi.

## Run (saare 11 qaris)

```bash
python reference_segment.py
```

Ya ek qari ke liye:

```bash
python reference_segment.py --qari minshawi          # default surah = 114
python reference_segment.py --surah 113              # Surah Al-Falaq (saare qaris)
python reference_segment.py --surah 113 --qari hatem # ek qari
```

Koi aur thresholds chahiye to:

```bash
python reference_segment.py --threshold-db -40 --min-silence 0.35
```

## Output

```
output/
├── ALL_QURAN_REFERENCE_INDEX.json              <- sab qaris ka index
├── muhammad-siddiq-al-minshawi/
│   ├── surah_114_an-nas__reference_segments.json   <- ALL details
│   └── segments/surah_114_an_nas_ref/
│       ├── seg_001.wav   بِسْمِ اللَّهِ           (1.54s)
│       ├── seg_002.wav   الرَّحْمَٰنِ الرَّحِيمِ     (2.88s)
│       ├── seg_003.wav   قُلْ أَعُوذُ            (1.71s)
│       ├── ... up to seg_012.wav وَالنَّاسِ
└── reciter-al-shatri/
    └── ... (har qari ka apna folder)
```

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

\* Duration example: Abdul Basit ke clips se (`refrence/AnFalaq/01–21.wav`). Ayaat
1-5 params ke baad jo basmalah audio hai wo segments ke bahar rehti hai (spec mein
basmalah nahi). Boundaries har qari ki apni ASR word-timings se nikalte hain.

## JSON ke andar kya hota hai

| Section | Details |
|---|---|
| `qari` | Name + Arabic name (محمد صديق المنشاوي) + source folder |
| `surah` | Number, Arabic/Roman name, meaning, ayah count |
| `audio_file` | Format, sample rate, channels, codec, duration, size |
| `model` | faster-whisper (CTranslate2), device, Tarteel base model |
| `segments[].transcription` | Reference phrase text |
| `segments[].words` | Word-level timestamps + probability (har word ka start/end) |
| `segments[].audio_info` | RMS, peak, crest, zero-crossing, spectral centroid/bandwidth/rolloff + RMS contour |
| `segments[].pitch` | **Detailed PYIN pitch**: full f0 contour (Hz + confidence), voiced contour (Hz + cents), robust stats, 12-bin histogram, semitone jumps |
| `segments[].surah_ayah` | basmalah / ayah number 1-6 |
| `global_words` | Poori recitation ke word timestamps ki list |

## Results (11 qaris)

| Qari | Matched |
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

> Jo recordings basmalah ke baghair shuru hoti hain (seedha `قُلْ أَعُوذُ` se),
> unke liye segment 1-2 (basmalah) unmatched rehte hain kyunki audio mein wo
> mojood hi nahi.

## Results — Surah 113 (Al-Falaq, 10 qaris, word-level 21 segments)

| Qari | Matched |
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

> Her qari ka output: `output/<qari>/segments/surah_113_al-falaq_ref/seg_001..021.wav`
> + `surah_113_al-falaq__reference_segments.json`.

## Kaise kaam karta hai

1. **VAD** audio ko silence gaps par split karta hai.
2. Har clip **faster-whisper (Tarteel CT2)** se `word_timestamps=True` ke saath
   transcribe hota hai → har word ka global start/end time.
3. **Reference phrases** ko normalized Arabic (harakat-stripped) ke through
   word stream par strict match kiya jata hai (1 skip-word tolerance).
4. Har phrase ke boundaries par **WAV clip** cut hota hai.
5. Har clip ka **PYIN pitch** + **audio features** extract hota hai.
6. Sab kuch **detailed JSON** mein likha jata hai.

## Files

```
├── reference_segment.py          <- main entry (reference-based)
├── main.py                       <- alternative: VAD-based ayah segmentation
├── scripts/convert_model.py      <- HF Tarteel -> CTranslate2 (1 dafa)
├── quran_segmenter/
│   ├── config.py                 # thresholds + model id
│   ├── audio_io.py               # load / save / file info
│   ├── silence_segmenter.py      # energy VAD
│   ├── word_aligner.py           # faster-whisper word timestamps + phrase matching
│   ├── features.py               # PYIN pitch + audio features
│   ├── align.py                  # Arabic text normalization
│   ├── transcriber.py            # HF transformers version (transcription only)
│   └── surah_meta.py             # surah + reference segments + qari names
└── models/tarteel-whisper-base-ar-quran-ct2/   # converted CT2 model
```

---

## 👨‍💻 Developer

| | |
|---|---|
| **Name** | Irfan Khan |
| **WhatsApp** | [+92 343 3791141](https://wa.me/923433791141) |
| **Email** | [khandev1211@gmail.com](mailto:khandev1211@gmail.com) |

Kisi bhi feature, bug, ya naye surah / qari / reference segments ke liye contact karein.