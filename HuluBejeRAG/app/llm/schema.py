from __future__ import annotations

from pydantic import BaseModel, Field


class QueryUnderstanding(BaseModel):
    """
    Structured result produced by LLM 1.
    """

    original_language: str = Field(
        description=(
            "The language used by the customer "
            "in the original message."
        )
    )

    retrieval_query: str = Field(
        description=(
            "A clear English version of the customer's "
            "question for knowledge retrieval."
        )
    )