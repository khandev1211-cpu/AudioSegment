"""Detailed pitch and audio feature extraction per segment.

Pitch is computed with librosa's PYIN algorithm which returns a frame-level
f0 in Hz together with voiced/unvoiced flags and confidence values. We export:

* the full contour (f0 + confidence per time frame, ``None`` when unvoiced),
* the voiced-only contour (Hz and cents re. a reference frequency),
* robust statistics, a histogram and the jump-count between adjacent voiced
  frames (a compact way to see how melody/maqam changes inside the ayah).

Audio features (RMS trajectory, peak/crest, zero-crossing and spectral
centroid/bandwidth/rolloff) describe loudness and timbre of the clip.
"""
from __future__ import annotations

import numpy as np
import librosa


def _robust_stats(a: np.ndarray) -> dict | None:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    q1, q3 = np.percentile(a, [25.0, 75.0])
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return {
        "min": round(float(a.min()), 2),
        "max": round(float(a.max()), 2),
        "mean": round(float(a.mean()), 2),
        "median": round(med, 2),
        "std": round(float(a.std()), 2),
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(q3 - q1), 2),
        "range": round(float(a.max() - a.min()), 2),
        "mad": round(float(mad), 2),
    }


def extract_pitch_features(
    y: np.ndarray,
    sr: int,
    *,
    fmin: float = 65.0,
    fmax: float = 500.0,
    hop_length: int = 512,
    cents_ref: float = 100.0,
) -> dict:
    """Extract the detailed pitch (f0) description of a mono signal."""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        fill_na=np.nan,
        frame_length=2048,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    f0 = np.asarray(f0, dtype=np.float64)
    voiced = ~np.isnan(f0)
    voiced_f0 = f0[voiced]
    voiced_times = times[voiced]
    conf = np.asarray(voiced_probs, dtype=np.float64)

    hop_sec = hop_length / sr
    frames = int(len(f0))

    # cents representation: 1200 * log2(f / ref)
    voiced_cents = 1200.0 * np.log2(voiced_f0 / cents_ref)

    # ---- statistics -----------------------------------------------------
    stats_hz = _robust_stats(voiced_f0)
    stats_cents = _robust_stats(voiced_cents)

    # ---- histogram (Hz, 12 equal-width bins between fmin and fmax) ------
    bin_edges = np.linspace(fmin, fmax, 13)
    hist_counts = np.histogram(voiced_f0, bins=bin_edges)[0] if voiced_f0.size else np.zeros(12)

    # ---- melodic jumps: semitone step larger than 2 between voiced frames
    jumps = 0
    if voiced_f0.size > 1:
        semitones = np.abs(1200.0 * np.log2(voiced_f0[1:] / voiced_f0[:-1]))
        jumps = int(np.sum(semitones > 200.0))  # > 2 semitones

    # ---- full contour (unvoiced frames stay None) -----------------------

    def _clean(x: float | None) -> float | None:
        return None if x is None or not np.isfinite(x) else round(float(x), 2)

    contour_f0 = [None] * frames
    contour_conf = [None] * frames
    for k in range(frames):
        if voiced[k]:
            contour_f0[k] = _clean(f0[k])
            contour_conf[k] = round(float(conf[k]), 4)

    return {
        "analysis": {
            "method": "librosa.pyin (Probabilistic YIN)",
            "sample_rate": int(sr),
            "hop_length": int(hop_length),
            "hop_sec": round(hop_sec, 6),
            "fmin_hz": float(fmin),
            "fmax_hz": float(fmax),
            "frame_count": frames,
            "cents_reference_hz": float(cents_ref),
        },
        "voiced_ratio": round(float(voiced.mean()) if frames else 0.0, 5),
        "voiced_time_sec": round(float(voiced.sum() * hop_sec), 4),
        "num_frames": frames,
        "semitone_jumps_gt2": jumps,
        "statistics_hz": stats_hz,
        "statistics_cents_ref100": stats_cents,
        "histogram_hz": {
            "bin_edges": [round(float(x), 2) for x in bin_edges],
            "counts": [int(x) for x in hist_counts],
        },
        "contour_full": {
            "times_sec": [round(float(t), 4) for t in times],
            "f0_hz": contour_f0,
            "confidence": contour_conf,
        },
        "contour_voiced": {
            "times_sec": [round(float(t), 4) for t in voiced_times],
            "f0_hz": [round(float(x), 2) for x in voiced_f0],
            "f0_cents_ref100": [round(float(x), 2) for x in voiced_cents],
        },
    }


def extract_audio_features(
    y: np.ndarray,
    sr: int,
    *,
    hop_sec: float = 0.01,
    n_fft: int = 2048,
) -> dict:
    """Loudness / timbre description of a mono signal."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return {"duration_sec": 0.0, "samples": 0, "note": "empty signal"}

    # RMS trajectory in dBFS
    hop = max(1, int(round(sr * hop_sec)))
    st = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    rms = librosa.feature.rms(S=st)[0]
    rms_db = 20.0 * np.log10(rms + 1e-10)
    rms_times = librosa.times_like(rms, sr=sr, hop_length=hop)

    peak = float(np.max(np.abs(y)))
    r = float(np.sqrt(np.mean(np.square(y)))) + 1e-10
    crest_db = 20.0 * np.log10(peak / r + 1e-10)

    zero_cross = float(np.mean(librosa.zero_crossings(y)))

    n_fft = max(1024, int(2 ** np.ceil(np.log2(len(y)))) if len(y) < 2048 else n_fft)
    spec = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    cent = float(np.mean(spec))
    bwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)))
    roll = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)))

    return {
        "analysis": {
            "sample_rate": int(sr),
            "hop_sec": round(hop / sr, 6),
            "n_fft": int(n_fft),
        },
        "duration_sec": round(len(y) / sr, 6),
        "samples": int(len(y)),
        "rms": {
            "mean_db": round(float(np.mean(rms_db)), 2),
            "median_db": round(float(np.median(rms_db)), 2),
            "std_db": round(float(np.std(rms_db)), 2),
            "max_db": round(float(np.max(rms_db)), 2),
            "min_db": round(float(np.min(rms_db)), 2),
        },
        "peak_amplitude": round(peak, 6),
        "crest_factor_db": round(max(crest_db, 0.0), 2),
        "zero_crossings_mean": round(zero_cross, 4),
        "spectral": {
            "centroid_mean_hz": round(cent, 1),
            "bandwidth_mean_hz": round(bwidth, 1),
            "rolloff_95_hz": round(roll, 1),
        },
        "rms_contour": {
            "times_sec": [round(float(t), 4) for t in rms_times],
            "rms_db": [round(float(x), 2) for x in rms_db],
        },
    }