from __future__ import annotations

import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.voice import router as voice_router
from app.api.routes.rag import router as rag_router


from app.llm.answer_generator import AnswerGenerator
from app.llm.client import LLMClient
from app.llm.query_generator import QueryGenerator



from app.retrieval.context_builder import ContextBuilder
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.chroma_store import ChromaStore
from app.retrieval.embedder import BGEEmbedder
from app.retrieval.retriever import Retriever


from app.voice.speech_to_text import SpeechToText
# 🎤 IMPORT THE UPDATED LANGUAGE IDENTIFIER CLASS HERE
from app.voice.language_identifier import LanguageIdentifier 



@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 70)
    print("STARTING HULUBEJE SUPPORT SERVICE")
    print("=" * 70)


    print()
    print("Initializing LLM client...")

    llm_client = LLMClient()

    print("LLM client ready.")


    print()
    print("Initializing query generator...")

    query_generator = QueryGenerator(
        client=llm_client
    )

    print("Query generator ready.")

   

    print()
    print("Initializing embedding model...")

    embedder = BGEEmbedder()

    print("Embedding model ready.")


    print()
    print("Connecting to ChromaDB...")

    chroma_store = ChromaStore(
        persist_directory="data/chroma"
    )

    print(
        f"Knowledge chunks: "
        f"{chroma_store.count}"
    )


    print()
    print("Initializing retriever...")

    retriever = Retriever(
        embedder=embedder,
        store=chroma_store,
    )

  
    app.state.retriever = retriever

    print("Retriever ready.")


    print()
    print("Initializing context builder...")

    context_builder = ContextBuilder()

    print("Context builder ready.")

 

    print()
    print("Initializing RAG answer generator...")

    answer_generator = AnswerGenerator(
        client=llm_client
    )

    print("RAG answer generator ready.")


    print()
    print("Initializing RAG pipeline...")

    rag_pipeline = RAGPipeline(
        query_generator=query_generator,
        retriever=retriever,
        context_builder=context_builder,
        answer_generator=answer_generator,
    )

    app.state.rag_pipeline = rag_pipeline

    print("RAG pipeline ready.")



    print()
    print("Initializing conversation context store...")

    app.state.conversations = {}

    print("Conversation context store ready.")


    # 🎤 INITIALIZE SPEECHBRAIN LANGUAGE IDENTIFIER
    print()
    print("Initializing SpeechBrain language identifier...")
    
    app.state.language_identifier = LanguageIdentifier()
    
    print("SpeechBrain language identifier mounted to app state.")


    print()
    print(
        "Initializing SeamlessM4T "
        "speech-to-text model..."
    )

    app.state.speech_to_text = SpeechToText()

    print(
        "SeamlessM4T Speech-to-Text ready."
    )


    print()
    print("=" * 70)
    print("HULUBEJE SUPPORT SERVICE READY")
    print("=" * 70)

    print()
    print("Text Chat API:")
    print("  /api/v1/chat")

    print()
    print("Voice Chat API:")
    print("  /api/v1/voice/chat")

    print()
    print("Knowledge Retrieval API:")
    print("  /api/v1/rag/retrieve")

    print()
    print("Voice Transcribe API:")
    print("  /api/v1/voice/transcribe")


    print()
    print("Health API:")
    print("  /health")


    yield

    # =========================================================
    # SHUTDOWN
    # =========================================================

    print()
    print("=" * 70)
    print("SHUTTING DOWN HULUBEJE SUPPORT SERVICE")
    print("=" * 70)


# =============================================================
# FASTAPI APPLICATION
# =============================================================

app = FastAPI(
    title="HuluBeje Support API",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================
# CORS
# =============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5284",
        "http://127.0.0.1:5284",

        "http://localhost:8000",
        "http://127.0.0.1:8000",

        "http://localhost:3000",
        "http://127.0.0.1:3000",

        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================
# ROUTES
# =============================================================

app.include_router(
    chat_router
)

app.include_router(
    voice_router
)


app.include_router(
    rag_router
)



@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "HuluBeje Support API",
    }
