from __future__ import annotations

from fastapi import Request

from app.retrieval.pipeline import RAGPipeline


def get_rag_pipeline(
    request: Request,
) -> RAGPipeline:

    return request.app.state.rag_pipeline