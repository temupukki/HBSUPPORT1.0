from __future__ import annotations

import numpy as np
import torch
import librosa

from transformers import (
    AutoProcessor,
    SeamlessM4TForSpeechToText,
)


class SpeechToText:
    MODEL_ID = "facebook/hf-seamless-m4t-medium"

    TARGET_SAMPLE_RATE = 16000

    def __init__(self) -> None:

        print(
            "Loading SeamlessM4T processor..."
        )

        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID
        )

        print(
            "Loading SeamlessM4T Medium STT model..."
        )

        self.model = SeamlessM4TForSpeechToText.from_pretrained(
            self.MODEL_ID
        )

        self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

        print(
            "SeamlessM4T STT ready."
        )

    def transcribe(
        self,
        audio,
        sampling_rate: int,
        target_language: str,
    ) -> str:

        print(
            f"STT original sample rate: "
            f"{sampling_rate} Hz"
        )

        print(
            f"STT target language: "
            f"{target_language}"
        )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Convert stereo to mono
        # ---------------------------------------------------------

        if audio.ndim > 1:

            audio = np.mean(
                audio,
                axis=1,
            )

        # ---------------------------------------------------------
        # Resample to 16 kHz
        # ---------------------------------------------------------

        if sampling_rate != self.TARGET_SAMPLE_RATE:

            print(
                f"Resampling audio "
                f"{sampling_rate} Hz -> "
                f"{self.TARGET_SAMPLE_RATE} Hz"
            )

            audio = librosa.resample(
                audio,
                orig_sr=sampling_rate,
                target_sr=self.TARGET_SAMPLE_RATE,
            )

            sampling_rate = self.TARGET_SAMPLE_RATE

        print(
            f"STT sample rate after resampling: "
            f"{sampling_rate} Hz"
        )

        print(
            f"STT audio samples: "
            f"{len(audio)}"
        )

        # ---------------------------------------------------------
        # Prepare audio for SeamlessM4T
        # ---------------------------------------------------------

        inputs = self.processor(
            audio=audio,
            sampling_rate=self.TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        print(
            "Generating transcription..."
        )

        # ---------------------------------------------------------
        # Speech-to-text
        # ---------------------------------------------------------

        with torch.no_grad():

            output_tokens = self.model.generate(
                **inputs,
                tgt_lang=target_language,
            )

        # ---------------------------------------------------------
        # Decode transcription
        # ---------------------------------------------------------

        text = self.processor.batch_decode(
            output_tokens,
            skip_special_tokens=True,
        )[0]

        return text.strip()