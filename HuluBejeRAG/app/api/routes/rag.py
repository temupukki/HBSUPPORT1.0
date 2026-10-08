from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/api/v1/rag",
    tags=["RAG"],
)


class RAGRetrieveRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Customer question to search in the knowledge base.",
    )


class RetrievedSource(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class RAGRetrieveResponse(BaseModel):
    question: str
    results: list[RetrievedSource]


@router.post(
    "/retrieve",
    response_model=RAGRetrieveResponse,
)
def retrieve_knowledge(
    request: Request,
    payload: RAGRetrieveRequest,
):
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    retriever = request.app.state.retriever

    try:
        chunks = retriever.retrieve(
            query=question,
            top_k=3,
        )

        results = [
            RetrievedSource(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                distance=chunk.distance,
            )
            for chunk in chunks
        ]

        return RAGRetrieveResponse(
            question=question,
            results=results,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge retrieval failed: {exc}",
        ) from exc