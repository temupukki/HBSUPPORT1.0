
from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


class LanguageIdentifier:
    """
    English-only voice input mode.

    The identify() method always returns English.
    SpeechBrain is still initialized, but its predictions are
    not used to select the language.

    Note:
        The returned score is a forced-selection marker, not a
        model confidence score.
    """

    MODEL_ID = "speechbrain/lang-id-voxlingua107-ecapa"
    TARGET_SAMPLE_RATE = 16000
    MODEL_DIRECTORY = Path("models/lang_id")

    def __init__(self) -> None:
        print("Initializing language identifier...")

        self.MODEL_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.classifier = EncoderClassifier.from_hparams(
            source=self.MODEL_ID,
            savedir=str(self.MODEL_DIRECTORY),
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": "cpu"},
        )

        self.classifier.eval()

        print("Language identifier ready. Voice input language: English.")

    @staticmethod
    def _to_mono(audio: Any) -> np.ndarray:
        """Convert supported audio input to mono float32."""

        waveform = np.asarray(audio, dtype=np.float32)

        if waveform.size == 0:
            raise ValueError("Audio input is empty.")

        if waveform.ndim == 1:
            mono = waveform

        elif waveform.ndim == 2:
            # Support (samples, channels) and (channels, samples).
            if waveform.shape[0] <= 8 and waveform.shape[1] > 8:
                mono = waveform.mean(axis=0)
            else:
                mono = waveform.mean(axis=1)

        else:
            raise ValueError(
                f"Unsupported audio dimensions: {waveform.ndim}"
            )

        if not np.isfinite(mono).all():
            raise ValueError(
                "Audio contains NaN or infinite values."
            )

        return np.ascontiguousarray(mono, dtype=np.float32)

    def _prepare_audio(
        self,
        audio: Any,
        sampling_rate: int,
    ) -> torch.Tensor:
        """Validate and prepare audio at 16 kHz."""

        if not isinstance(sampling_rate, (int, np.integer)):
            raise ValueError("Sample rate must be an integer.")

        if sampling_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        waveform = self._to_mono(audio)

        if sampling_rate != self.TARGET_SAMPLE_RATE:
            waveform = librosa.resample(
                waveform,
                orig_sr=int(sampling_rate),
                target_sr=self.TARGET_SAMPLE_RATE,
                res_type="soxr_hq",
            ).astype(np.float32)

        if waveform.size == 0:
            raise ValueError(
                "Audio is empty after resampling."
            )

        waveform = waveform - np.mean(
            waveform,
            dtype=np.float64,
        )

        peak = float(np.max(np.abs(waveform)))

        if peak < 1e-5:
            raise ValueError(
                "Audio is too quiet or contains silence."
            )

        duration = waveform.size / self.TARGET_SAMPLE_RATE

        print(f"Audio duration: {duration:.2f} seconds")
        print(f"Audio peak amplitude: {peak:.6f}")

        return torch.from_numpy(
            np.ascontiguousarray(waveform, dtype=np.float32)
        ).unsqueeze(0)

    def identify(
        self,
        audio: Any,
        sampling_rate: int,
    ) -> tuple[str, float]:
        """
        Validate the recording and always select English.

        Returns the same tuple format as the original implementation.
        """

        # Keep audio validation and preprocessing.
        self._prepare_audio(audio, sampling_rate)

        print("Language selection: English (forced)")
        print("Detected language: en: English")

        # 100.0 is a forced-selection marker, not confidence.
        return "en: English", 100.0

    @staticmethod
    def normalize_language(language: str) -> str:
        """Normalize language labels without changing the selection."""

        value = str(language).strip().lower()

        if ":" in value:
            value = value.split(":", 1)[0].strip()

        if value == "en":
            return "en"

        return value if value else "unknown"
