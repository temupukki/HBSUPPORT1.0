from __future__ import annotations

import json

from app.llm.client import LLMClient
from app.llm.schema import QueryUnderstanding


class QueryGenerator:
    """
    Uses LLM 1 to identify the customer's language
    and create an English retrieval query.
    """

    SYSTEM_PROMPT = """
You are the query-understanding component of HuluBeje
customer support.

You MUST NOT answer the customer's question.

Your ONLY responsibilities are:

1. Detect the language of the customer's message.

2. Understand the exact customer intent.

3. Create one clear English retrieval query that
   preserves the exact intent of the customer.

The retrieval query will be used to search an English
HuluBeje knowledge base.

LANGUAGE RULES:

- Return the actual language used by the customer.
- Use a common English language name such as:
  English, Amharic, Arabic, French, Spanish, German,
  Italian, Portuguese, Swahili, etc.
- If multiple languages are present, identify the
  primary language of the customer's message.

TRANSLATION RULES:

- If the customer writes in English, preserve the
  meaning and produce a clear English retrieval query.
- If the customer writes in another language, translate
  the customer's actual question into English.
- Do NOT summarize the question.
- Do NOT broaden the question.
- Do NOT narrow the question.
- Do NOT add information.
- Do NOT remove important conditions.
- Preserve product names, delivery types, numbers,
  dates, locations, and policy terms when present.
- Preserve the customer's exact intent.

EXAMPLES:

Customer:
"How long does standard delivery take?"

Return:
{
  "original_language": "English",
  "retrieval_query": "How long does standard delivery take?"
}

Customer:
"كم من الوقت يستغرق التوصيل العادي؟"

Return:
{
  "original_language": "Arabic",
  "retrieval_query": "How long does standard delivery take?"
}

Customer:
"¿Cuánto tarda la entrega estándar?"

Return:
{
  "original_language": "Spanish",
  "retrieval_query": "How long does standard delivery take?"
}

Customer:
"የመደበኛ መላኪያ ስንት ቀን ይወስዳል?"

Return:
{
  "original_language": "Amharic",
  "retrieval_query": "How long does standard delivery take?"
}

IMPORTANT:

Never answer the customer.

Return ONLY valid JSON.

The JSON structure MUST be:

{
  "original_language": "...",
  "retrieval_query": "..."
}
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