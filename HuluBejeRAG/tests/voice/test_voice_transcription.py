from __future__ import annotations

from pathlib import Path

import soundfile as sf

from app.voice.language_identifier import LanguageIdentifier
from app.voice.language_codes import speechbrain_to_seamless
from app.voice.speech_to_text import SpeechToText


AUDIO_FILE = Path(
    "tests/voice/amharic.wav"
)


def main():

    print("=" * 70)
    print("HULUBEJE VOICE TRANSCRIPTION TEST")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load audio
    # -------------------------------------------------------------

    print()
    print(f"Loading audio: {AUDIO_FILE}")

    audio, sample_rate = sf.read(
        AUDIO_FILE,
        dtype="float32",
    )

    print(
        f"Sample rate: {sample_rate} Hz"
    )

    print(
        f"Audio samples: {len(audio)}"
    )

    # -------------------------------------------------------------
    # Step 1: Language identification
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 1: LANGUAGE IDENTIFICATION")
    print("=" * 70)

    language_identifier = LanguageIdentifier()

    language, confidence = (
        language_identifier.identify(
            audio=audio,
            sampling_rate=sample_rate,
        )
    )

    print()
    print(
        f"Detected language: {language}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    # -------------------------------------------------------------
    # Step 2: Convert language code
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 2: LANGUAGE CODE CONVERSION")
    print("=" * 70)

    seamless_language = (
        speechbrain_to_seamless(
            language
        )
    )

    print()
    print(
        f"SpeechBrain language: "
        f"{language}"
    )

    print(
        f"SeamlessM4T language: "
        f"{seamless_language}"
    )

    # -------------------------------------------------------------
    # Step 3: SeamlessM4T transcription
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 3: SEAMLESSM4T TRANSCRIPTION")
    print("=" * 70)

    speech_to_text = SpeechToText()

    transcription = (
        speech_to_text.transcribe(
            audio=audio,
            sampling_rate=sample_rate,
            target_language=seamless_language,
        )
    )

    # -------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print()
    print(
        f"Language:      {language}"
    )

    print(
        f"Confidence:    {confidence:.2f}%"
    )

    print(
        f"Seamless code: {seamless_language}"
    )

    print()
    print("TRANSCRIPTION:")
    print(
        transcription
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()