"""Transcription with the local Tarteel AI Whisper model.

Model: ``tarteel-ai/whisper-base-ar-quran`` — a Whisper *base* model fine-tuned
for Arabic Quranic audio. It is already downloaded in the local HuggingFace
cache, offline mode is forced so no network calls are made.
"""
from __future__ import annotations

import os
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging

# keep the console output readable: transformers/tqdm warnings are noise here
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .config import ASR_LANGUAGE, ASR_SAMPLE_RATE, TARTEEL_MODEL_ID


class TarteelTranscriber:
    """Loads the Tarteel model once and transcribes 16 kHz mono clips."""

    def __init__(self, model_id: str = TARTEEL_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        t0 = time.time()
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(model_id)
            .to(device=self.device)
            .to(dtype=self.dtype)
        )
        self.model.eval()
        self._upgrade_generation_config()
        self.load_sec = round(time.time() - t0, 2)

    # ------------------------------------------------------------------
    def _upgrade_generation_config(self) -> None:
        """Give the generation config the modern Whisper attributes.

        The Tarteel checkpoint ships without a ``generation_config.json`` and
        was created in the era of ``forced_decoder_ids``. New transformers
        requires ``is_multilingual``/``lang_to_id``/``task_to_id`` before the
        ``language`` / ``task`` generate arguments can be used. We rebuild them
        from the tokenizer vocabulary (offline, no network).
        """
        gc = self.model.generation_config
        tok = self.processor.tokenizer
        vocab = tok.get_vocab()

        # default Whisper generation configs may already carry these attributes
        # as empty/None — only replace them when they are actually unusable.
        if not getattr(gc, "is_multilingual", None):
            gc.is_multilingual = True
        if not getattr(gc, "lang_to_id", None):
            gc.lang_to_id = {
                t: i for t, i in vocab.items()
                if re.fullmatch(r"<\|[a-z]{2}\|>", t)
            }
        if not getattr(gc, "task_to_id", None):
            task_ids = {t: i for t, i in vocab.items() if t in ("<|transcribe|>", "<|translate|>")}
            gc.task_to_id = {
                "transcribe": task_ids.get("<|transcribe|>"),
                "translate": task_ids.get("<|translate|>"),
            }
        # short-form decoding needs the "no timestamps" sentinel token id
        if not getattr(gc, "no_timestamps_token_id", None):
            gc.no_timestamps_token_id = vocab.get("<|notimestamps|>")

    # ------------------------------------------------------------------
    def transcribe(
        self,
        y16: np.ndarray,
        *,
        beam: int = 1,
    ) -> str:
        """Transcribe one 16 kHz mono clip, return the Arabic text."""
        if y16 is None or len(y16) < ASR_SAMPLE_RATE // 20:  # < 50 ms -> skip
            return ""

        features = self.processor(
            np.asarray(y16, dtype=np.float32),
            sampling_rate=ASR_SAMPLE_RATE,
            return_tensors="pt",
        ).input_features

        gen_kwargs = {
            "language": ASR_LANGUAGE,
            "task": "transcribe",
            "max_length": 448,
            "num_beams": beam,
            "no_speech_threshold": 0.6,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
        }

        with torch.inference_mode():
            input_ids = features.to(self.device, dtype=self.dtype)
            try:
                generated = self.model.generate(input_ids, **gen_kwargs)
            except TypeError:
                # tolerate older/newer transformers by falling back to the
                # minimal set of generation kwargs.
                gen_kwargs.pop("no_speech_threshold", None)
                gen_kwargs.pop("compression_ratio_threshold", None)
                gen_kwargs.pop("logprob_threshold", None)
                generated = self.model.generate(input_ids, **gen_kwargs)

        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()

    # ------------------------------------------------------------------
    def info(self) -> dict:
        return {
            "id": self.model_id,
            "family": "Whisper base (Tarteel AI fine-tune for Quranic Arabic)",
            "loaded_from": "local huggingface cache (offline)",
            "device": self.device,
            "dtype": str(self.dtype),
            "load_time_sec": self.load_sec,
        }