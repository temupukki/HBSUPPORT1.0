from pathlib import Path

import soundfile as sf

from app.voice.speech_to_text import SpeechToText


AUDIO_FILE = Path("tests/voice/harvard.wav")


def main():
    print(f"Loading audio: {AUDIO_FILE}")

    audio, sample_rate = sf.read(
        AUDIO_FILE,
        dtype="float32",
    )

    print(f"Sample rate: {sample_rate}")
    print(f"Audio samples: {len(audio)}")

    stt = SpeechToText()

    print("Transcribing...")

    text = stt.transcribe(
        audio=audio,
        sampling_rate=sample_rate,
        target_language="eng",
    )

    print()
    print("=" * 60)
    print("TRANSCRIPTION:")
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()