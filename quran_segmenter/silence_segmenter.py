"""Energy-based voice activity detection and ayah/segment splitting.

The reciter almost always pauses between ayat, so a simple and robust RMS
energy VAD (with min-gap merging) is enough to produce clean per-ayah clips.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentationResult:
    """Output of :func:`segment_by_silence`."""

    segments: list = field(default_factory=list)  # list[dict]
    rms_db: np.ndarray = None
    times_sec: np.ndarray = None
    hop_samples: int = 0
    sr: int = 16000

    def to_dict(self) -> dict:
        return {
            "method": "energy-based VAD (RMS in dBFS) with min-gap merging",
            "hop_sec": round(self.hop_samples / self.sr, 5),
            "frame_count": int(len(self.rms_db)) if self.rms_db is not None else 0,
            "segments_amount": len(self.segments),
        }


def _frame_rms_db(y: np.ndarray, sr: int, hop_sec: float) -> tuple[np.ndarray, np.ndarray, int]:
    hop = max(1, int(round(sr * hop_sec)))
    n = len(y) // hop
    frames = np.empty(n, dtype=np.float64)
    for i in range(n):
        w = y[i * hop:(i + 1) * hop]
        r = float(np.sqrt(np.mean(np.square(w)))) if w.size else 0.0
        frames[i] = 20.0 * np.log10(r + 1e-10)
    times = np.arange(n) * hop / sr
    return frames, times, hop


def segment_by_silence(
    y: np.ndarray,
    sr: int,
    *,
    hop_sec: float = 0.01,
    threshold_db: float = -38.0,
    min_silence_sec: float = 0.30,
    min_segment_sec: float = 0.60,
    pad_sec: float = 0.10,
) -> SegmentationResult:
    """Split ``y`` into segments at the quiet gaps.

    Parameters mirror the config module defaults. The segmentation works on
    *frame* resolution (``hop_sec`` frames) — silence runs shorter than
    ``min_silence_sec`` are bridged so small breath pauses inside an ayah do
    not split it, while quiet gaps longer than that become segment borders.
    """
    frames, times, hop = _frame_rms_db(y, sr, hop_sec)
    voiced = frames > threshold_db

    # 1) Bridge short silence gaps (a breath inside one ayah).
    out = voiced.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            silence_sec = (j - i) * hop / sr
            if silence_sec < min_silence_sec:
                out[i:j] = True
            i = j
        else:
            i += 1

    # 2) Build raw segments from contiguous voiced runs.
    raw: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i in range(n):
        v = bool(out[i])
        if v and not in_seg:
            in_seg, start = True, i
        elif not v and in_seg:
            in_seg = False
            raw.append((start, i))
    if in_seg:
        raw.append((start, n))

    # 3) Convert to sample ranges, apply padding and drop tiny clips.
    pad = int(round(sr * pad_sec))
    segments = []
    for idx, (fs, fe) in enumerate(raw, start=1):
        dur_sec = (fe - fs) * hop / sr
        if dur_sec < min_segment_sec:
            continue
        s = max(0, fs * hop - pad)
        e = min(len(y), fe * hop + pad)
        segments.append(
            {
                "index": idx,
                "start_sec": round(s / sr, 6),
                "end_sec": round(e / sr, 6),
                "duration_sec": round((e - s) / sr, 6),
                "start_sample": int(s),
                "end_sample": int(e),
            }
        )

    return SegmentationResult(
        segments=segments, rms_db=frames, times_sec=times, hop_samples=hop, sr=sr
    )