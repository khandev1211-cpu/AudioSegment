"""Loading, saving and inspecting audio files."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass
class AudioFileInfo:
    """File level metadata read with soundfile."""

    path: str
    filename: str
    format: str
    subtype: str
    sample_rate: int
    channels: int
    frames: int
    duration_sec: float
    size_bytes: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["codec"] = self.format  # friendly alias
        return d


def audio_file_info(path: str | Path) -> AudioFileInfo:
    """Read file-level metadata (works for mp3, wav, flac, ogg ...).

    ``soundfile``/libsndfile does not read mp3 on every build, so we fall
    back to librosa (Audioread/ffmpeg) in that case.
    """
    p = Path(path)
    try:
        info = sf.info(str(p))
        return AudioFileInfo(
            path=str(p.resolve()),
            filename=p.name,
            format=info.format,
            subtype=info.subtype,
            sample_rate=info.samplerate,
            channels=info.channels,
            frames=info.frames,
            duration_sec=round(float(info.duration), 6),
            size_bytes=p.stat().st_size,
        )
    except Exception:
        # Fallback for mp3/compressed codecs: sample rate + duration via librosa,
        # channel count unknown without decoding -> 1 (mono, librosa decodes mono).
        sr = int(librosa.get_samplerate(str(p)))
        dur = float(librosa.get_duration(path=str(p)))
        return AudioFileInfo(
            path=str(p.resolve()),
            filename=p.name,
            format=p.suffix.lstrip(".").upper() or "AUDIO",
            subtype="unknown",
            sample_rate=sr,
            channels=1,
            frames=int(round(dur * sr)),
            duration_sec=round(dur, 6),
            size_bytes=p.stat().st_size,
        )


def load_audio(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio as mono float32.

    When ``sr`` is given the audio is resampled to that rate, otherwise the
    native sample rate is kept. Returns ``(y, sample_rate)``.
    """
    y, native_sr = librosa.load(str(path), sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if sr is None or sr == native_sr:
        return y, int(native_sr)
    y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
    return y.astype(np.float32), int(sr)


def save_wav_pcm16(path: str | Path, y: np.ndarray, sr: int) -> Path:
    """Write mono audio as a 16-bit PCM WAV (small, universally playable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(y, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    sf.write(str(path), x, sr, subtype="PCM_16")
    return path


def rms_db(y: np.ndarray, eps: float = 1e-10) -> float:
    """Root-mean-square amplitude in dBFS for a whole signal."""
    if y.size == 0:
        return -np.inf
    r = float(np.sqrt(np.mean(np.square(y))))
    return 20.0 * np.log10(r + eps)