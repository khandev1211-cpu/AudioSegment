"""Align transcribed segments to surah ayah numbers.

Uses normalized Arabic text and greedy forward matching: every segment's
normalized transcription is searched inside one stream of the surah's
reference ayahs (searching only forward from the previous match), so even
ayahs that the reciter split into two clips — or merged into one — are still
labelled with the correct ayah.
"""
from __future__ import annotations

import re
from typing import Sequence

BASMALAH = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u0670\u0640]")


def normalize_arabic(text: str) -> str:
    """Strip harakat/tatweel and unify letter forms so ASR text can match."""
    if not text:
        return ""
    t = _DIACRITICS.sub("", text)
    t = re.sub(r"[إأآا]", "ا", t)
    t = re.sub(r"ى", "ي", t)
    t = re.sub(r"ة", "ه", t)
    t = re.sub(r"ؤ", "و", t)
    t = re.sub(r"ئ", "ي", t)
    t = re.sub(r"[^\u0621-\u064A\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def align_segments_to_ayahs(
    segment_texts: Sequence[str],
    ayah_texts: Sequence[str],
) -> list:
    """Return one label per segment: int (ayah), 'basmalah' or None."""
    prefixed = [BASMALAH] + list(ayah_texts)
    label_refs = []
    for i, r in enumerate(prefixed):
        norm = normalize_arabic(r)
        if norm:
            label_refs.append((norm, "basmalah" if i == 0 else i))

    # Build one stream: ayahs joined by spaces, remember where each starts.
    anchors = []
    stream_parts = []
    acc = 0
    for r, label in label_refs:
        anchors.append((acc, label))
        stream_parts.append(r)
        acc += len(r) + 1  # +1 for the separating space
    stream = " ".join(stream_parts)
    stream_len = len(stream)

    def ayah_at(pos: int):
        best = None
        for o, label in anchors:
            if o <= pos:
                best = label
            else:
                break
        return best

    search_from = 0
    result = []
    for seg in segment_texts:
        norm = normalize_arabic(seg)
        if not norm:
            result.append(None)
            continue

        pos = stream.find(norm, search_from)
        if pos < 0:
            # fallback: anchor on the longest word of the segment
            for w in sorted(norm.split(), key=len, reverse=True):
                pos = stream.find(w, search_from)
                if pos >= 0:
                    break
        if pos < 0:
            result.append(None)
            continue

        result.append(ayah_at(pos))
        search_from = min(stream_len, pos + max(1, len(norm)))

    return result