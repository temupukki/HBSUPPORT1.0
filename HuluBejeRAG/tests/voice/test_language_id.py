from pathlib import Path

import soundfile as sf

from app.voice.language_identifier import LanguageIdentifier


AUDIO_FILE = Path(
    "tests/voice/harvard.wav"
)


def main():

    print("=" * 60)
    print("HULUBEJE LANGUAGE IDENTIFICATION TEST")
    print("=" * 60)

    print()
    print(f"Loading audio: {AUDIO_FILE}")

    audio, sample_rate = sf.read(
        AUDIO_FILE,
        dtype="float32",
    )

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio samples: {len(audio)}")

    print()
    print("Initializing language identifier...")

    identifier = LanguageIdentifier()

    print()
    print("Detecting spoken language...")

    language, confidence = identifier.identify(
        audio=audio,
        sampling_rate=sample_rate,
    )

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Language:   {language}")
    print(f"Confidence: {confidence:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()