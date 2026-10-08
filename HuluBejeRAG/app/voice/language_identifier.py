from __future__ import annotations

import torch
import librosa
import numpy as np

from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


class LanguageIdentifier:
    MODEL_ID = "speechbrain/lang-id-voxlingua107-ecapa"
    TARGET_SAMPLE_RATE = 16000

    def __init__(self) -> None:
        print("Loading SpeechBrain language identification model...")
        self.classifier = EncoderClassifier.from_hparams(
            source=self.MODEL_ID,
            savedir="models/lang_id",
            local_strategy=LocalStrategy.COPY,
        )
        
        print("Loading Silero VAD for live recording cleaning...")
        # Silero VAD is tiny, runs locally, and perfectly extracts speech from noisy mic audio
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            trust_repo=True,
            verbose=False
        )
        self.get_speech_timestamps = utils[0]
        self.collect_chunks = utils[4]

        print("SpeechBrain language identification ready.")

    def identify(
        self,
        audio,
        sampling_rate: int,
    ) -> tuple[str, float]:

        print(f"Language ID original sample rate: {sampling_rate} Hz")

        # Convert input into a standard numpy array first to handle layout variations safely
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = np.asarray(audio, dtype=np.float32)

        # ---------------------------------------------------------
        # FIX 1: Robust Stereo to Mono Conversion
        # ---------------------------------------------------------
        # Live mics often output shape (samples, channels). Librosa expects (channels, samples).
        if audio_np.ndim > 1:
            # If shape is (samples, channels), transpose it to (channels, samples)
            if audio_np.shape[0] > audio_np.shape[1] and audio_np.shape[1] <= 2:
                audio_np = audio_np.T
            audio_np = librosa.to_mono(audio_np)

        # ---------------------------------------------------------
        # Resample audio to 16 kHz
        # ---------------------------------------------------------
        if sampling_rate != self.TARGET_SAMPLE_RATE:
            print(f"Resampling audio {sampling_rate} Hz -> {self.TARGET_SAMPLE_RATE} Hz")
            audio_np = librosa.resample(
                audio_np,
                orig_sr=sampling_rate,
                target_sr=self.TARGET_SAMPLE_RATE,
            )
            sampling_rate = self.TARGET_SAMPLE_RATE

        # Convert back to clean 1D PyTorch tensor for processing
        waveform = torch.tensor(audio_np, dtype=torch.float32)

        # ---------------------------------------------------------
        # FIX 2: Voice Activity Detection (VAD) - Strip Silence & Noise
        # ---------------------------------------------------------
        with torch.no_grad():
            speech_timestamps = self.get_speech_timestamps(
                waveform, 
                self.vad_model, 
                sampling_rate=self.TARGET_SAMPLE_RATE
            )
        
        if speech_timestamps:
            print("VAD: Stripping background noise and dead air...")
            waveform = self.collect_chunks(speech_timestamps, waveform)
        else:
            print("VAD Warning: No clear speech detected. Processing raw audio anyway.")

        # ---------------------------------------------------------
        # FIX 3: Peak Amplitude Normalization
        # ---------------------------------------------------------
        # Scales live microphone volumes up to matching pre-recorded studio data levels
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-7:
            waveform = waveform / max_val

        print(f"Language ID audio samples after cleaning: {waveform.shape[-1]}")

        # Add batch dimension [1, samples]
        waveform = waveform.unsqueeze(0)

        print("Running language identification...")

        # ---------------------------------------------------------
        # Run SpeechBrain language identification
        # ---------------------------------------------------------
        with torch.no_grad():
            # SpeechBrain's classify_batch internally computes log-probabilities
            out_prob, score, index_tensor, text_lab = self.classifier.classify_batch(waveform)

        # Correct extraction mapping directly using SpeechBrain's native output parsing
        language = text_lab[0]
        
        # Convert log-probabilities back to standard confidence percentage
        confidence = torch.exp(score).item() * 100

        return language, confidence
