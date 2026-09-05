"""Static metadata about the surahs used by the pipeline.

Only the surahs actually present in this workspace are listed explicitly
(Surah 1 Al-Fatiha, Surah 113 Al-Falaq, Surah 114 An-Nas). Unknown surah
numbers get a safe fallback entry. Add more entries here to extend the
project later.
"""
from __future__ import annotations

QARI_AR_NAMES = {
    "muhammad siddiq al minshawi": "محمد صديق المنشاوي",
    "abdul basit abdul samad": "عبد الباسط عبد الصمد",
    "ahmed el agamy": "أحمد العجمي",
    "al shatri": "أبو بكر الشاطري",
    "bandar balila": "بندر بليلة",
    "hatem fareed al waer": "حاتم فريد الواعر",
    "ibrahim al-akhdar": "إبراهيم الأخضر",
    "khalid al jalil": "خالد الجليل",
    "khalifa al tunaiji": "خليفة الطنيجي",
    "saad al ghamdi": "سعد الغامدي",
    "salah bukhatir": "صلاح بوخاطر",
    "saud al shuraim": "سعود الشريم",
}

SURAH_META: dict[int, dict] = {
    1: {
        "number": 1,
        "name_ar": "الفاتحة",
        "name_roman": "Al-Fatiha",
        "name_meaning": "The Opening",
        "ayahs_count": 7,
        "revelation_type": "Meccan",
        "juz": 1,
        "translation_urdu": "الفاتحة (کھولنے والی)",
        "note": "basmalah ayah 1 hai; neeche 6 ayah diye hain (align basmalah khud prefix karta hai)",
        "ayahs_text": [
            "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
            "الرَّحْمَٰنِ الرَّحِيمِ",
            "مَالِكِ يَوْمِ الدِّينِ",
            "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
            "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
            "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
        ],
    },
    113: {
        "number": 113,
        "name_ar": "الفلق",
        "name_roman": "Al-Falaq",
        "name_meaning": "The Daybreak",
        "ayahs_count": 5,
        "revelation_type": "Meccan",
        "juz": 30,
        "translation_urdu": "الفلق (صبح کی روشنی)",
        "ayahs_text": [
            "قُلْ أَعُوذُ بِرَبِّ الْفَلَقِ",
            "مِنْ شَرِّ مَا خَلَقَ",
            "وَمِنْ شَرِّ غَاسِقٍ إِذَا وَقَبَ",
            "وَمِنْ شَرِّ النَّفَّاثَاتِ فِي الْعُقَدِ",
            "وَمِنْ شَرِّ حَاسِدٍ إِذَا حَسَدَ",
        ],
        "reference_segments": [
            # Word-level reference splits from the client's surah 113
            # specification (5 ayaat, 21 word-groups, no basmalah).
            {"id": 1, "ayah": 1, "text": "قُلْ"},
            {"id": 2, "ayah": 1, "text": "أَعُوذُ"},
            {"id": 3, "ayah": 1, "text": "بِرَبِّ الْفَلَقِ"},
            {"id": 4, "ayah": 2, "text": "مِن"},
            {"id": 5, "ayah": 2, "text": "شَرِّ"},
            {"id": 6, "ayah": 2, "text": "مَا"},
            {"id": 7, "ayah": 2, "text": "خَلَقَ"},
            {"id": 8, "ayah": 3, "text": "وَمِن"},
            {"id": 9, "ayah": 3, "text": "شَرِّ"},
            {"id": 10, "ayah": 3, "text": "غَاسِقٍ"},
            {"id": 11, "ayah": 3, "text": "إِذَا"},
            {"id": 12, "ayah": 3, "text": "وَقَبَ"},
            {"id": 13, "ayah": 4, "text": "وَمِن"},
            {"id": 14, "ayah": 4, "text": "شَرِّ النَّفَّاثَاتِ"},
            {"id": 15, "ayah": 4, "text": "فِي"},
            {"id": 16, "ayah": 4, "text": "الْعُقَدِ"},
            {"id": 17, "ayah": 5, "text": "وَمِن"},
            {"id": 18, "ayah": 5, "text": "شَرِّ"},
            {"id": 19, "ayah": 5, "text": "حَاسِدٍ"},
            {"id": 20, "ayah": 5, "text": "إِذَا"},
            {"id": 21, "ayah": 5, "text": "حَسَدَ"},
        ],
    },
    114: {
        "number": 114,
        "name_ar": "الناس",
        "name_roman": "An-Nas",
        "name_meaning": "Mankind",
        "ayahs_count": 6,
        "revelation_type": "Meccan",
        "juz": 30,
        "translation_urdu": "الناس (لوگ)",
        "ayahs_text": [
            "قُلْ أَعُوذُ بِرَبِّ النَّاسِ",
            "مَلِكِ النَّاسِ",
            "إِلَٰهِ النَّاسِ",
            "مِنْ شَرِّ الْوَسْوَاسِ الْخَنَّاسِ",
            "الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ",
            "مِنَ الْجِنَّةِ وَالنَّاسِ",
        ],
        "reference_segments": [
            {"id": 1, "ayah": "basmalah", "text": "بِسْمِ اللَّهِ"},
            {"id": 2, "ayah": "basmalah", "text": "الرَّحْمَٰنِ الرَّحِيمِ"},
            {"id": 3, "ayah": 1, "text": "قُلْ أَعُوذُ"},
            {"id": 4, "ayah": 1, "text": "بِرَبِّ النَّاسِ"},
            {"id": 5, "ayah": 2, "text": "مَلِكِ النَّاسِ"},
            {"id": 6, "ayah": 3, "text": "إِلَٰهِ النَّاسِ"},
            {"id": 7, "ayah": 4, "text": "مِنْ شَرِّ"},
            {"id": 8, "ayah": 4, "text": "الْوَسْوَاسِ الْخَنَّاسِ"},
            {"id": 9, "ayah": 5, "text": "الَّذِي يُوَسْوِسُ"},
            {"id": 10, "ayah": 5, "text": "فِي صُدُورِ النَّاسِ"},
            {"id": 11, "ayah": 6, "text": "مِنَ الْجِنَّةِ"},
            {"id": 12, "ayah": 6, "text": "وَالنَّاسِ"},
        ],
    },
}


def get_surah_meta(number: int | str) -> dict:
    """Return metadata for a surah number, with a safe fallback."""
    try:
        n = int(str(number).strip())
    except (TypeError, ValueError):
        n = 0
    entry = SURAH_META.get(n)
    if entry is None:
        return {
            "number": n if n else 0,
            "name_ar": None,
            "name_roman": "Unknown",
            "name_meaning": None,
            "ayahs_count": None,
            "revelation_type": None,
            "juz": None,
            "translation_urdu": None,
            "ayahs_text": None,
            "note": "Surah metadata not present in the bundled table.",
        }
    return dict(entry)