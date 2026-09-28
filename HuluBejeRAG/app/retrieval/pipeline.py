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

        # -------------------------------------------------
        # 1. Prepare active conversation history
        # -------------------------------------------------

        conversation_history = (
            conversation_history or []
        )

        # -------------------------------------------------
        # 2. Understand the customer's current message
        # -------------------------------------------------

        understanding = (
            self.query_generator.understand(
                user_message
            )
        )

        # -------------------------------------------------
        # 3. Retrieve relevant HuluBeje knowledge
        # -------------------------------------------------

        chunks = self.retriever.retrieve(
            query=understanding.retrieval_query,
            top_k=top_k,
        )

        # -------------------------------------------------
        # 4. Build knowledge context
        # -------------------------------------------------

        context = self.context_builder.build(
            chunks
        )

        # -------------------------------------------------
        # 5. Generate the final customer answer
        # -------------------------------------------------

        answer = self.answer_generator.generate(
            original_language=(
                understanding.original_language
            ),
            retrieval_query=(
                understanding.retrieval_query
            ),
            context=context,
            current_message=user_message,
            conversation_history=conversation_history,
        )

        # -------------------------------------------------
        # 6. Return the complete RAG result
        # -------------------------------------------------

        return RAGResult(
            original_language=(
                understanding.original_language
            ),
            retrieval_query=(
                understanding.retrieval_query
            ),
            chunks=chunks,
            context=context,
            answer=answer,
        )