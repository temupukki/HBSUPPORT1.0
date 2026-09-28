from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
)

from app.core.dependencies import get_rag_pipeline
from app.retrieval.pipeline import RAGPipeline


router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(
        get_rag_pipeline
    ),
) -> ChatResponse:

    result = pipeline.run(
        user_message=request.message,
        conversation_history=[
            message.model_dump()
            for message in request.conversation_history
        ],
        top_k=3,
    )

    return ChatResponse(
        answer=result.answer
    )