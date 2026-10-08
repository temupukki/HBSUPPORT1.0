from __future__ import annotations

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

from app.voice.language_identifier import LanguageIdentifier
from app.voice.speech_to_text import SpeechToText


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 70)
    print("STARTING HULUBEJE RAG SERVICE")
    print("=" * 70)

    # =========================================================
    # LLM CLIENT
    # =========================================================

    print()
    print("Initializing LLM client...")

    llm_client = LLMClient()

    print("LLM client ready.")

    # =========================================================
    # QUERY GENERATOR
    # =========================================================

    print()
    print("Initializing query generator...")

    query_generator = QueryGenerator(
        client=llm_client
    )

    print("Query generator ready.")

    # =========================================================
    # EMBEDDING MODEL
    # =========================================================

    print()
    print("Initializing embedding model...")

    embedder = BGEEmbedder()

    print("Embedding model ready.")

    # =========================================================
    # CHROMA DB
    # =========================================================

    print()
    print("Connecting to ChromaDB...")

    chroma_store = ChromaStore(
        persist_directory="data/chroma"
    )

    print(
        f"Knowledge chunks: "
        f"{chroma_store.count}"
    )

    # =========================================================
    # RETRIEVER
    # =========================================================

    print()
    print("Initializing retriever...")

    retriever = Retriever(
        embedder=embedder,
        store=chroma_store,
    )

    # Save retriever so the dedicated RAG
    # retrieval API can use it.
    app.state.retriever = retriever

    print("Retriever ready.")

    # =========================================================
    # CONTEXT BUILDER
    # =========================================================

    print()
    print("Initializing context builder...")

    context_builder = ContextBuilder()

    print("Context builder ready.")

    # =========================================================
    # ANSWER GENERATOR
    # =========================================================

    print()
    print("Initializing answer generator...")

    answer_generator = AnswerGenerator(
        client=llm_client
    )

    print("Answer generator ready.")

    # =========================================================
    # RAG PIPELINE
    # =========================================================

    print()
    print("Initializing RAG pipeline...")

    app.state.rag_pipeline = RAGPipeline(
        query_generator=query_generator,
        retriever=retriever,
        context_builder=context_builder,
        answer_generator=answer_generator,
    )

    print("RAG pipeline ready.")

    # =========================================================
    # SPEECHBRAIN LANGUAGE IDENTIFICATION
    # =========================================================

    print()
    print(
        "Initializing SpeechBrain "
        "language identification..."
    )

    app.state.language_identifier = (
        LanguageIdentifier()
    )

    print(
        "SpeechBrain language identification ready."
    )

    # =========================================================
    # SEAMLESSM4T SPEECH-TO-TEXT
    # =========================================================

    print()
    print(
        "Initializing SeamlessM4T STT model..."
    )

    app.state.speech_to_text = SpeechToText()

    print(
        "SeamlessM4T Speech-to-Text ready."
    )

    # =========================================================
    # SERVICE READY
    # =========================================================

    print()
    print("=" * 70)
    print("HULUBEJE RAG SERVICE READY")
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
    print("Health API:")
    print("  /health")

    print()
    print("=" * 70)

    yield

    # =========================================================
    # SHUTDOWN
    # =========================================================

    print()
    print("=" * 70)
    print("SHUTTING DOWN HULUBEJE RAG SERVICE")
    print("=" * 70)


app = FastAPI(
    title="HuluBeje RAG API",
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


# =============================================================
# HEALTH CHECK
# =============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "HuluBeje RAG API",
    }