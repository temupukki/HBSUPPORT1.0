
from __future__ import annotations

import json
import logging
from typing import Any

from app.llm.client import LLMClient
from app.llm.schema import QueryUnderstanding

logger = logging.getLogger(__name__)


class QueryGenerator:
    """
    LLM 1 — detects the customer's language and generates
    a complete English retrieval query.

    The customer's original language is preserved for the
    AnswerGenerator to use when generating the final answer.
    """

    SYSTEM_PROMPT = """
You are the query-understanding component of HuluBeje customer support.

You MUST NOT answer the customer's question.

Your ONLY job:
1. Detect the language used in the current customer message.
2. Use conversation history, if provided, to resolve follow-up
   references such as "it", "that", "so it works?", and "what about it?".
3. Produce a complete, self-contained English retrieval query
   that captures the customer's full intent.

CRITICAL RULES:
- The retrieval_query MUST always be in English.
- Translate the customer's intent from any language into English.
- Support English, Amharic (Ge'ez script), Romanized Amharic,
  Arabic, and mixed-language messages.
- For Romanized Amharic, identify the intended language as Amharic.
- If the message is a follow-up, use the supplied history to make
  the retrieval query self-contained.
- Never answer the customer's question.
- Never invent missing details.
- Return ONLY one valid JSON object.
- Do not use Markdown or add explanations.

OUTPUT FORMAT:
{
  "original_language": "<English name of the language>",
  "retrieval_query": "<complete English retrieval query>"
}

EXAMPLES:

Customer: "How long does standard delivery take?"
Output:
{"original_language":"English","retrieval_query":"How long does standard delivery take?"}

Customer: "hulubeje min new"
Output:
{"original_language":"Amharic","retrieval_query":"What is HuluBeje?"}

History:
[{"role":"assistant","content":"HuluBeje is an Ethiopian e-commerce platform operating in Addis Ababa."}]
Customer: "so it works in Ethiopia right?"
Output:
{"original_language":"English","retrieval_query":"Does HuluBeje operate in Ethiopia?"}

History:
[{"role":"user","content":"hulubeje min new"},{"role":"assistant","content":"HuluBeje is an e-commerce platform."}]
Customer: "Ethiopia wust new yemisera?"
Output:
{"original_language":"Amharic","retrieval_query":"Does HuluBeje operate inside Ethiopia?"}

History:
[{"role":"assistant","content":"HuluBeje offers hotel booking services."}]
Customer: "cinema also yelew?"
Output:
{"original_language":"Amharic","retrieval_query":"Does HuluBeje also offer cinema ticket booking?"}

Return ONLY the JSON object. No code blocks, no extra text.
"""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @staticmethod
    def _parse_json_response(response: str) -> dict[str, Any]:
        """
        Parse a JSON object returned by the LLM.

        Supports plain JSON, Markdown-fenced JSON, and JSON
        surrounded by explanatory text.
        """
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(
                "LLM 1 returned an empty response. "
                "Check the LLM provider, model configuration, "
                "and response extraction in app/llm/client.py."
            )

        raw_response = response.strip()

        # Remove Markdown code fences when present.
        if raw_response.startswith("```"):
            lines = raw_response.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            raw_response = "\n".join(lines).strip()

        # First, try parsing the entire response.
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            # If extra text surrounds the JSON, extract the
            # outermost JSON object and try parsing again.
            start = raw_response.find("{")
            end = raw_response.rfind("}")

            if start == -1 or end <= start:
                logger.error(
                    "LLM 1 returned no JSON object. Response preview: %r",
                    raw_response[:500],
                )
                raise RuntimeError(
                    "LLM 1 did not return a JSON object. "
                    "Inspect the LLM client and model response."
                )

            json_text = raw_response[start : end + 1]

            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as exc:
                logger.error(
                    "LLM 1 returned malformed JSON. Response preview: %r",
                    raw_response[:500],
                )
                raise RuntimeError(
                    "LLM 1 returned malformed JSON."
                ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "LLM 1 must return a JSON object, not a list or scalar."
            )

        return data

    def understand(
        self,
        user_message: str,
    ) -> QueryUnderstanding:
        """
        Detect the customer's language and generate an English
        retrieval query.

        Note: conversation history must be added to this method
        and the LLM prompt if follow-up resolution is required.
        """
        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        response = self.client.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_message,
        )

        data = self._parse_json_response(response)

        try:
            result = QueryUnderstanding.model_validate(data)
        except Exception as exc:
            logger.error(
                "LLM 1 returned an invalid query-understanding structure."
            )
            raise RuntimeError(
                "LLM 1 returned an invalid query-understanding structure. "
                "Expected original_language and retrieval_query."
            ) from exc

        # Reject empty required fields before retrieval.
        if (
            not result.original_language.strip()
            or not result.retrieval_query.strip()
        ):
            raise RuntimeError(
                "LLM 1 returned empty original_language "
                "or retrieval_query fields."
            )

        return result
