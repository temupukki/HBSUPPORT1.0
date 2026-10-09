from __future__ import annotations

import json
from io import BytesIO

import numpy as np
import librosa
import soundfile as sf

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
)

from app.voice.language_codes import (
    speechbrain_to_seamless,
)


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

    # =========================================================
    # 1. REQUEST RECEIVED
    # =========================================================

    print()
    print("=" * 70)
    print("VOICE API REQUEST RECEIVED")
    print("=" * 70)
    print(f"Endpoint: /api/v1/voice/chat")
    print(f"Filename: {audio.filename}")
    print(f"Content type: {audio.content_type}")

    # =========================================================
    # 2. VALIDATE AUDIO
    # =========================================================

    if not audio.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required.",
        )

    # =========================================================
    # 3. PARSE CONVERSATION HISTORY
    # =========================================================

    try:

        history = json.loads(
            conversation_history
        )

        if not isinstance(history, list):
            raise ValueError(
                "Conversation history must be a list."
            )

    except (
        json.JSONDecodeError,
        ValueError,
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid conversation history.",
        )

    # =========================================================
    # 4. READ AUDIO
    # =========================================================

    try:

        audio_bytes = audio.file.read()

        print()
        print("=" * 70)
        print("VOICE AUDIO")
        print("=" * 70)
        print(
            f"Audio bytes: {len(audio_bytes)}"
        )

        waveform, sample_rate = sf.read(
            BytesIO(audio_bytes),
            dtype="float32",
        )

        print(
            f"Sample rate: {sample_rate}"
        )

        print(
            f"Sample count: {len(waveform)}"
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read audio: {exc}"
            ),
        )

    # =========================================================
    # 5. LANGUAGE IDENTIFICATION
    # =========================================================

    print()
    print("=" * 70)
    print("VOICE STAGE 1: LANGUAGE IDENTIFICATION")
    print("=" * 70)

    try:

        language, confidence = (
            request
            .app
            .state
            .language_identifier
            .identify(
                audio=waveform,
                sampling_rate=sample_rate,
            )
        )

        print(
            f"Detected language: {language}"
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Language identification failed: "
                f"{exc}"
            ),
        )

    # =========================================================
    # 6. LANGUAGE MAPPING
    # =========================================================

    print()
    print("=" * 70)
    print("VOICE STAGE 2: LANGUAGE MAPPING")
    print("=" * 70)

    try:

        seamless_language = (
            speechbrain_to_seamless(
                language
            )
        )

        print(
            f"Seamless language: "
            f"{seamless_language}"
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    # =========================================================
    # 7. SPEECH TO TEXT
    # =========================================================

    print()
    print("=" * 70)
    print("VOICE STAGE 3: SPEECH TO TEXT")
    print("=" * 70)

    try:

        transcription = (
            request
            .app
            .state
            .speech_to_text
            .transcribe(
                audio=waveform,
                sampling_rate=sample_rate,
                target_language=seamless_language,
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Speech transcription failed: "
                f"{exc}"
            ),
        )

    if not transcription.strip():

        raise HTTPException(
            status_code=422,
            detail=(
                "No speech could be recognized."
            ),
        )

    print()
    print("=" * 70)
    print("VOICE TRANSCRIPTION")
    print("=" * 70)
    print(transcription)

    # =========================================================
    # 8. GENERATE RETRIEVAL QUERY
    # =========================================================

    print()
    print("=" * 70)
    print("VOICE STAGE 4: QUERY UNDERSTANDING")
    print("=" * 70)

    try:

        pipeline = (
            request
            .app
            .state
            .rag_pipeline
        )

        understanding = (
            pipeline
            .query_generator
            .understand(
                transcription
            )
        )

        retrieval_query = (
            understanding.retrieval_query
        )

        original_language = (
            understanding.original_language
        )

        print(
            f"Original language: "
            f"{original_language}"
        )

        print(
            f"Retrieval query: "
            f"{retrieval_query}"
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Retrieval query generation failed: "
                f"{exc}"
            ),
        )

    # =========================================================
    # 9. RUN EXISTING RAG USING RETRIEVAL QUERY
    # =========================================================

    print()
    print("=" * 70)
    print("VOICE STAGE 5: RAG PIPELINE")
    print("=" * 70)

    try:

        rag_result = (
            pipeline
            .run_from_retrieval_query(
                retrieval_query=retrieval_query,
                original_language=original_language,
                current_message=transcription,
                conversation_history=history,
                top_k=3,
            )
        )

        print(
            f"Retrieved chunks: "
            f"{len(rag_result.chunks)}"
        )

        print(
            f"Answer length: "
            f"{len(rag_result.answer)}"
        )

    except Exception as exc:

        print()
        print(
            "VOICE RAG ERROR:"
        )
        print(exc)

        raise HTTPException(
            status_code=500,
            detail=(
                "RAG processing failed: "
                f"{exc}"
            ),
        )

    # =========================================================
    # 10. RESPONSE READY
    # =========================================================

    print()
    print("=" * 70)
    print("VOICE STAGE 6: RESPONSE READY")
    print("=" * 70)

    print(
        f"Answer: {rag_result.answer}"
    )

    print()
    print("=" * 70)
    print("VOICE API REQUEST COMPLETED")
    print("=" * 70)

    # =========================================================
    # 11. RETURN TO BLAZOR
    # =========================================================

    return {
        "transcription": transcription,
        "answer": rag_result.answer,
        "original_language": original_language,
        "retrieval_query": retrieval_query,
    }
# =========================================================
# NEW ENDPOINT: TRANSCRIPTION ONLY
# =========================================================
@router.post("/transcribe")
def voice_transcribe(
    request: Request,
    audio: UploadFile = File(...),
):
    print()
    print("=" * 70)
    print("VOICE TRANSCRIBE API REQUEST RECEIVED")
    print("=" * 70)
    print(f"Endpoint: /api/v1/voice/transcribe")
    print(f"Filename: {audio.filename}")

    if not audio.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required.",
        )

    # 1. READ AUDIO (Telegram/Live Mic parsing logic)
    try:
        audio_bytes = audio.file.read()
        try:
            waveform, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32")
        except Exception:
            print("Soundfile native read failed. Falling back to Librosa parser...")
            waveform, sample_rate = librosa.load(BytesIO(audio_bytes), sr=None, dtype=np.float32)

        if waveform.ndim > 1:
            if waveform.shape[1] > waveform.shape[0]:
                waveform = waveform.T
            waveform = librosa.to_mono(waveform)

        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read audio file: {exc}")

    # 2. LANGUAGE IDENTIFICATION
    try:
        language, confidence = request.app.state.language_identifier.identify(
            audio=waveform, sampling_rate=sample_rate
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Language identification failed: {exc}")

    # 3. LANGUAGE MAPPING
    try:
        seamless_language = speechbrain_to_seamless(language)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 4. SPEECH TO TEXT ONLY (STT)
    try:
        transcription = request.app.state.speech_to_text.transcribe(
            audio=waveform,
            sampling_rate=sample_rate,
            target_language=seamless_language,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech transcription failed: {exc}")

    if not transcription.strip():
        raise HTTPException(status_code=422, detail="No speech could be recognized.")

    # Return only the text details, skipping RAG and LLM completely
    return {
        "transcription": transcription,
        "language": language,
        "confidence": confidence
    }
