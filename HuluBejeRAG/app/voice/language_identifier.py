from __future__ import annotations

import torch
import librosa

from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


class LanguageIdentifier:
    MODEL_ID = "speechbrain/lang-id-voxlingua107-ecapa"

    TARGET_SAMPLE_RATE = 16000

    def __init__(self) -> None:
        print(
            "Loading SpeechBrain language identification model..."
        )

        self.classifier = EncoderClassifier.from_hparams(
            source=self.MODEL_ID,
            savedir="models/lang_id",
            local_strategy=LocalStrategy.COPY,
        )

        print(
            "SpeechBrain language identification ready."
        )

    def identify(
        self,
        audio,
        sampling_rate: int,
    ) -> tuple[str, float]:

        print(
            f"Language ID original sample rate: "
            f"{sampling_rate} Hz"
        )

        waveform = torch.tensor(
            audio,
            dtype=torch.float32,
        )

        # ---------------------------------------------------------
        # Convert stereo audio to mono
        # ---------------------------------------------------------

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)

        # ---------------------------------------------------------
        # Resample audio to 16 kHz
        # ---------------------------------------------------------

        if sampling_rate != self.TARGET_SAMPLE_RATE:

            print(
                f"Resampling audio "
                f"{sampling_rate} Hz -> "
                f"{self.TARGET_SAMPLE_RATE} Hz"
            )

            audio = waveform.numpy()

            audio = librosa.resample(
                audio,
                orig_sr=sampling_rate,
                target_sr=self.TARGET_SAMPLE_RATE,
            )

            waveform = torch.tensor(
                audio,
                dtype=torch.float32,
            )

            sampling_rate = self.TARGET_SAMPLE_RATE

        print(
            f"Language ID sample rate after resampling: "
            f"{sampling_rate} Hz"
        )

        print(
            f"Language ID audio samples: "
            f"{waveform.shape[-1]}"
        )

        # ---------------------------------------------------------
        # Add batch dimension
        #
        # [samples] -> [1, samples]
        # ---------------------------------------------------------

        waveform = waveform.unsqueeze(0)

        print(
            "Running language identification..."
        )

        # ---------------------------------------------------------
        # Run SpeechBrain language identification
        # ---------------------------------------------------------

        with torch.no_grad():

            prediction = self.classifier.classify_batch(
                waveform
            )

        scores = prediction[0]

        # ---------------------------------------------------------
        # Convert model scores into probabilities
        # ---------------------------------------------------------

        probabilities = torch.softmax(
            scores,
            dim=1,
        )

        # ---------------------------------------------------------
        # Find language with highest probability
        # ---------------------------------------------------------

        index = probabilities.argmax(
            dim=1
        ).item()

        confidence = (
            probabilities[0, index].item()
            * 100
        )

        # ---------------------------------------------------------
        # Convert model index into language label
        # ---------------------------------------------------------

        language = (
            self.classifier.hparams.label_encoder.decode_torch(
                torch.tensor([index])
            )[0]
        )

        return language, confidence
