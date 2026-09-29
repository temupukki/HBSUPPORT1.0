from __future__ import annotations

from dataclasses import dataclass

from app.llm.answer_generator import AnswerGenerator
from app.llm.query_generator import QueryGenerator
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.retriever import RetrievedChunk, Retriever


@dataclass
class RAGResult:
    original_language: str
    retrieval_query: str
    chunks: list[RetrievedChunk]
    context: str
    answer: str


class RAGPipeline:

    def __init__(
        self,
        query_generator: QueryGenerator,
        retriever: Retriever,
        context_builder: ContextBuilder,
        answer_generator: AnswerGenerator,
    ) -> None:

        self.query_generator = query_generator
        self.retriever = retriever
        self.context_builder = context_builder
        self.answer_generator = answer_generator

    # ---------------------------------------------------------
    # NORMAL TEXT CHAT
    # ---------------------------------------------------------

    def run(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
        top_k: int = 3,
    ) -> RAGResult:

        if not user_message.strip():
            raise ValueError(
                "User message cannot be empty."
            )

        conversation_history = (
            conversation_history or []
        )

        # -----------------------------------------------------
        # Query understanding
        # -----------------------------------------------------

        understanding = (
            self.query_generator.understand(
                user_message
            )
        )

        # -----------------------------------------------------
        # Continue using the normal RAG flow
        # -----------------------------------------------------

        return self.run_from_retrieval_query(
            retrieval_query=(
                understanding.retrieval_query
            ),
            original_language=(
                understanding.original_language
            ),
            current_message=user_message,
            conversation_history=conversation_history,
            top_k=top_k,
        )

    # ---------------------------------------------------------
    # RAG FROM AN ALREADY GENERATED RETRIEVAL QUERY
    # ---------------------------------------------------------

    def run_from_retrieval_query(
        self,
        retrieval_query: str,
        original_language: str,
        current_message: str,
        conversation_history: list[dict] | None = None,
        top_k: int = 3,
    ) -> RAGResult:

        if not retrieval_query.strip():
            raise ValueError(
                "Retrieval query cannot be empty."
            )

        conversation_history = (
            conversation_history or []
        )

        # -----------------------------------------------------
        # Retrieval
        # -----------------------------------------------------

        chunks = self.retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
        )

        # -----------------------------------------------------
        # Context
        # -----------------------------------------------------

        context = self.context_builder.build(
            chunks
        )

        # -----------------------------------------------------
        # Answer generation
        # -----------------------------------------------------

        answer = self.answer_generator.generate(
            original_language=original_language,
            retrieval_query=retrieval_query,
            context=context,
            current_message=current_message,
            conversation_history=conversation_history,
        )

        return RAGResult(
            original_language=original_language,
            retrieval_query=retrieval_query,
            chunks=chunks,
            context=context,
            answer=answer,
        )