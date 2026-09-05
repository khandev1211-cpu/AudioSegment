"""Word-level transcription (faster-whisper, CT2 Tarteel model) and reference
phrase matching.

The client wants a fixed set of *phrase segments* (e.g. the 12 An-Nas clips in
``refrence/AnNaas``). Different qaris recite at different speeds, so we cannot
reuse the reference clip timings. Instead:

1. An energy VAD splits the recording at pause points (roughly per ayah).
2. Every VAD clip is transcribed alone with word-level timestamps (padded
   slightly so boundaries are not cut off).
3. Word timestamps from all clips are shifted to global time and merged.
4. The reference phrase texts are matched, word by word, onto that global
   timeline using normalized Arabic (greedy forward matching).
5. Phrase boundaries are cut at the mid-point between adjacent phrase words.
"""
from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np

from .align import normalize_arabic
from .silence_segmenter import segment_by_silence

ASR_SR = 16000
PAD_SEC = 0.12
HALLUC_PROB = 0.7
# NB: "ما" was once a hallucination token for An-Nas, but it is a REAL word in
# Al-Falaq (مِنْ شَرِّ مَا خَلَقَ) and appears with p≈1.0, so it must stay.
HALLUC_TOKENS = {"يغفى", "يحضى", "وشيوم", "مايغفى"}


class WordAligner:
    """Loads the CT2 Tarteel model once; produces global word timestamps."""

    def __init__(self, model_dir: str | Path, device: str | None = None, compute_type: str | None = None):
        from faster_whisper import WhisperModel

        self.model_dir = str(model_dir)
        if torch_cuda_available():
            self.device = device or "cuda"
            self.compute_type = compute_type or "float16"
        else:
            self.device = device or "cpu"
            self.compute_type = compute_type or "int8"
        self.model = WhisperModel(self.model_dir, device=self.device, compute_type=self.compute_type)

    # ------------------------------------------------------------------
    def global_words(self, y16: np.ndarray, sr: int = ASR_SR) -> list[dict]:
        """Transcribe each VAD clip and merge word timestamps to global time."""
        vad = segment_by_silence(y16, sr, threshold_db=-38.0)
        total_samples = len(y16)
        n_samples_pad = int(round(PAD_SEC * sr))

        all_words: list[dict] = []
        for seg in vad.segments:
            s = max(0, seg["start_sample"] - n_samples_pad)
            e = min(total_samples, seg["end_sample"] + n_samples_pad)
            clip = y16[s:e]
            segments = list(
                self.model.transcribe(
                    clip,
                    language="ar",
                    task="transcribe",
                    word_timestamps=True,
                    beam_size=1,
                    condition_on_previous_text=False,
                )[0]
            )
            for sgm in segments:
                for w in sgm.words:
                    if w.probability < HALLUC_PROB:
                        continue
                    wt = w.word.strip()
                    norm = normalize_arabic(wt)
                    if not norm or norm in HALLUC_TOKENS:
                        continue
                    gs = (s / sr) + (w.start - PAD_SEC)
                    ge = (s / sr) + (w.end - PAD_SEC)
                    all_words.append(
                        {
                            "start_sec": round(max(0.0, gs), 4),
                            "end_sec": round(max(0.0, ge), 4),
                            "text": wt,
                            "norm": norm,
                            "probability": round(w.probability, 4),
                        }
                    )

        all_words.sort(key=lambda d: d["start_sec"])

        # remove pad-duplicates (the same word seen in an overlap region)
        merged: list[dict] = []
        for w in all_words:
            if merged:
                prev = merged[-1]
                if w["norm"] == prev["norm"] and w["start_sec"] < prev["end_sec"]:
                    continue
            merged.append(w)
        return merged

    # ------------------------------------------------------------------
    def info(self) -> dict:
        return {
            "engine": "faster-whisper (CTranslate2)",
            "model_directory": self.model_dir,
            "device": self.device,
            "compute_type": self.compute_type,
            "source_base_model": "tarteel-ai/whisper-base-ar-quran",
            "language": "arabic",
        }


def torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reference phrase matching
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    norm = normalize_arabic(text)
    return norm.split() if norm else []


def _same_norm(a: str, b: str) -> bool:
    """Compare two normalized Arabic words.

    Exact first; then two light ASR tolerances:
      1. trailing ``ا`` from tanween (ASR ``حسدا`` ~ reference ``حسد``),
      2. fuzzy n-gram similarity for 3+ letter words — the Tarteel model
         sometimes emits a single extra/mutated letter (``الفلقه`` ~ ``الفلق``,
         ``خلقت`` ~ ``خلق``, ``وقبم`` ~ ``وقب``). Min length 3 keeps short
         function words strictly exact.
    """
    if a == b:
        return True
    for x, y in ((a, b), (b, a)):
        if y and len(x) == len(y) + 1 and x.endswith("ا") and x[:-1] == y:
            return True
    if min(len(a), len(b)) >= 3:
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.75
    return False


def _find_first_match(word_norms: list[str], tokens: list[str], search_from: int, max_skip: int):
    """Greedy forward search for ``tokens`` inside ``word_norms``.

    Returns ``(start_index, spans)`` for the first window (starting at or
    after ``search_from``) that matches — either exact contiguous, or with up
    to ``max_skip`` extra ASR words between two reference tokens.
    """
    ntok = len(tokens)
    if not ntok:
        return None
    for start in range(search_from, len(word_norms) - ntok + 1):
        window = word_norms[start:start + ntok]
        if all(_same_norm(w, t) for w, t in zip(window, tokens)):
            return start, list(range(start, start + ntok))
        if ntok > 1:
            pos = start
            skipped = 0
            ok = True
            for tok in tokens:
                while pos < len(word_norms) and not _same_norm(word_norms[pos], tok):
                    pos += 1
                    skipped += 1
                    if skipped > max_skip:
                        ok = False
                        break
                if not ok or pos >= len(word_norms):
                    ok = False
                    break
                pos += 1
            if ok and (pos - start - ntok) <= max_skip:
                # recompute the true end span (consumes tokens while skipping)
                pos = start
                last_pos = start
                for tok in tokens:
                    while pos < len(word_norms) and not _same_norm(word_norms[pos], tok):
                        pos += 1
                    last_pos = pos
                    pos += 1
                return start, list(range(start, last_pos + 1))
    return None


def match_phrases_to_words(global_words: list[dict], reference_segments: list[dict]) -> list[dict]:
    """Match each reference phrase to a run of global words.

    Strict-first matching: a phrase must map onto normalized tokens with at
    most ``MAX_SKIP`` unknown ASR words inside. ASR streams sometimes *drop* a
    single reference word (e.g. ``ما`` inside آية 2), so when the full phrase
    does not match we also try every single-token-deletion variant.

    Returns the reference segments annotated with:
      ``word_indices``          indexes into ``global_words``
      ``start_sec`` / ``end_sec``  first/last word time
      ``matched``                bool
    """
    MAX_SKIP = 1  # extra ASR words tolerated inside a phrase

    word_norms = [w["norm"] for w in global_words]
    matches = []
    search_from = 0

    for ph in reference_segments:
        tokens = _tokenize(ph["text"])
        ntok = len(tokens)
        if not ntok:
            matches.append({**ph, "matched": False, "word_indices": []})
            continue

        result = _find_first_match(word_norms, tokens, search_from, MAX_SKIP)
        if result is None and ntok > 2:
            # ASR sometimes skips one reference token entirely; try variants
            for drop in range(ntok):
                variant = tokens[:drop] + tokens[drop + 1:]
                result = _find_first_match(word_norms, variant, search_from, MAX_SKIP)
                if result is not None:
                    break

        if result is None:
            matches.append({**ph, "matched": False, "word_indices": []})
            continue

        found, spans = result
        matches.append(
            {
                **ph,
                "matched": True,
                "word_indices": spans,
                "start_sec": global_words[spans[0]]["start_sec"],
                "end_sec": global_words[spans[-1]]["end_sec"],
            }
        )
        search_from = spans[-1] + 1

    return matches


def phrase_cut_times(matches: list[dict], audio_duration: float) -> list[dict]:
    """Turn matched phrases into concrete segment clips.

    A segment runs from phrase start to the mid-point between this phrase's
    last word and the next phrase's first word, so tiny inter-phrase gaps are
    shared evenly. Unmatched phrases get ``None`` boundaries.
    """
    matched = [m for m in matches if m["matched"]]

    cuts = []
    for m in matches:
        if not m["matched"]:
            cuts.append({"id": m["id"], "start_sec": None, "end_sec": None, "transcription": m["text"], "ayah": m["ayah"]})
            continue
        start_sec = m["start_sec"]
        end_sec = m["end_sec"]

        # fill the gap to the next matched phrase
        nxt = None
        for m2 in matched:
            if m2["word_indices"][0] > m["word_indices"][-1]:
                nxt = m2
                break
        if nxt is not None:
            boundary = (end_sec + nxt["start_sec"]) / 2.0
            if boundary > start_sec:
                end_sec = boundary
        else:
            end_sec = min(end_sec + 0.10, audio_duration)

        start_sec = max(0.0, start_sec - 0.03)
        end_sec = min(audio_duration, end_sec)
        cuts.append(
            {
                "id": m["id"],
                "ayah": m["ayah"],
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "duration_sec": round(end_sec - start_sec, 4),
                "transcription": m["text"],
            }
        )
    return cuts