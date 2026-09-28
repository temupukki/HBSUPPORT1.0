from __future__ import annotations

import json
from io import BytesIO

import soundfile as sf

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
)

from app.voice.language_codes import speechbrain_to_seamless


router = APIRouter(
    prefix="/api/v1/voice",
    tags=["Voice"],
)


@router.post("/chat")
def voice_chat(
    request: Request,
    audio: UploadFile = File(...),
    conversation_history: str = "[]",
):

    # ---------------------------------------------------------
    # 1. Validate audio
    # ---------------------------------------------------------

    if not audio.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required.",
        )

    # ---------------------------------------------------------
    # 2. Parse conversation history
    # ---------------------------------------------------------

    try:
        history = json.loads(conversation_history)

        if not isinstance(history, list):
            raise ValueError(
                "Conversation history must be a list."
            )

    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid conversation history.",
        )

    # ---------------------------------------------------------
    # 3. Read audio
    # ---------------------------------------------------------

    try:
        audio_bytes = audio.file.read()

        waveform, sample_rate = sf.read(
            BytesIO(audio_bytes),
            dtype="float32",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read audio: {exc}",
        )

    # ---------------------------------------------------------
    # 4. Detect spoken language
    # ---------------------------------------------------------

    try:
        language, confidence = (
            request.app.state.language_identifier.identify(
                audio=waveform,
                sampling_rate=sample_rate,
            )
        )

        print()
        print("=" * 70)
        print("VOICE LANGUAGE DETECTION")
        print("=" * 70)
        print(f"Detected language: {language}")
        print(f"Confidence: {confidence:.2f}%")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Language identification failed: {exc}",
        )

    # ---------------------------------------------------------
    # 5. Convert SpeechBrain language -> Seamless language
    # ---------------------------------------------------------

    try:
        seamless_language = speechbrain_to_seamless(
            language
        )

        print(
            f"Seamless language code: "
            f"{seamless_language}"
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    # ---------------------------------------------------------
    # 6. Speech -> Text
    # ---------------------------------------------------------

    try:
        transcription = (
            request.app.state.speech_to_text.transcribe(
                audio=waveform,
                sampling_rate=sample_rate,
                target_language=seamless_language,
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Speech transcription failed: {exc}",
        )

    if not transcription.strip():
        raise HTTPException(
            status_code=422,
            detail="No speech could be recognized.",
        )

    print()
    print("=" * 70)
    print("VOICE TRANSCRIPTION")
    print("=" * 70)
    print(transcription)

    # ---------------------------------------------------------
    # 7. SAME RAG PIPELINE AS TEXT CHAT
    # ---------------------------------------------------------

    try:
        rag_result = request.app.state.rag_pipeline.run(
            user_message=transcription,
            conversation_history=history,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG processing failed: {exc}",
        )

    # ---------------------------------------------------------
    # 8. Return result
    # ---------------------------------------------------------

    return {
        "transcription": transcription,
        "answer": rag_result.answer,
        "original_language": rag_result.original_language,
        "retrieval_query": rag_result.retrieval_query,
    }