from __future__ import annotations

import json

from app.llm.client import LLMClient
from app.llm.schema import QueryUnderstanding


class QueryGenerator:
    """
    LLM 1 — detects language and generates an English retrieval query.
    Conversation history is used to resolve follow-up references.
    """


    SYSTEM_PROMPT = """
You are the query-understanding component of HuluBeje customer support.

You MUST NOT answer the customer's question.

Your ONLY job:
1. Detect the language the customer used in their current message.
2. Use the conversation history (if provided) to resolve follow-up references like "it", "that", "so it works?", "what about it?".
3. Produce a complete, self-contained English retrieval query that captures the full intent.

CRITICAL RULES:
- The retrieval_query MUST always be in English.
- Translate from any language (Amharic, Romanized Amharic, Arabic, etc.) into English.
- If the message is a follow-up, combine with history context to make it self-contained.
- Never answer the customer.
- Return ONLY valid JSON. No markdown, no explanation.

OUTPUT FORMAT:
{"original_language": "<English name of the language>", "retrieval_query": "<complete English query>"}

EXAMPLES:

Customer: "How long does standard delivery take?"
Output: {"original_language": "English", "retrieval_query": "How long does standard delivery take?"}

Customer: "hulubeje min new"
Output: {"original_language": "Amharic", "retrieval_query": "What is HuluBeje?"}

History: [{"role":"assistant","content":"HuluBeje is an Ethiopian e-commerce platform operating in Addis Ababa."}]
Customer: "so it works in Ethiopia right?"
Output: {"original_language": "English", "retrieval_query": "Does HuluBeje operate in Ethiopia?"}

History: [{"role":"user","content":"hulubeje min new"},{"role":"assistant","content":"HuluBeje is an e-commerce platform."}]
Customer: "Ethiopia wust new yemisera?"
Output: {"original_language": "Amharic", "retrieval_query": "Does HuluBeje operate inside Ethiopia?"}

History: [{"role":"assistant","content":"HuluBeje offers hotel booking services."}]
Customer: "cinema also yelew?"
Output: {"original_language": "Amharic", "retrieval_query": "Does HuluBeje also offer cinema ticket booking?"}

Return ONLY the JSON object. No code blocks, no extra text.
"""

    def __init__(
        self,
        client: LLMClient,
    ) -> None:

        self.client = client

    def understand(
        self,
        user_message: str,
    ) -> QueryUnderstanding:

        if not user_message.strip():
            raise ValueError(
                "User message cannot be empty."
            )

        response = self.client.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_message,
        )

        try:
            data = json.loads(response)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "LLM 1 returned invalid JSON."
            ) from exc

        try:
            return QueryUnderstanding.model_validate(
                data
            )

        except Exception as exc:
            raise RuntimeError(
                "LLM 1 returned an invalid "
                "query-understanding structure."
            ) from exc