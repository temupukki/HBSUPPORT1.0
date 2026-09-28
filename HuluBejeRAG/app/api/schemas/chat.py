from __future__ import annotations

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: str = Field(
        description="Message role: user or assistant."
    )

    content: str = Field(
        min_length=1,
        description="Message content."
    )


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        description="The customer's current message."
    )

    conversation_history: list[ConversationMessage] = Field(
        default_factory=list,
        description="Previous messages from the active conversation only."
    )


class ChatResponse(BaseModel):
    answer: str