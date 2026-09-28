from __future__ import annotations

import os

from litellm import completion


class LLMClient:
    """
    Client used to communicate with the configured LLM provider.
    """

    def __init__(
        self,
        model: str | None = None,
    ) -> None:

        self.model = model or os.getenv(
            "LLM_MODEL",
        )

        if not self.model:
            raise ValueError(
                "LLM_MODEL is not configured."
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "LLM returned an empty response."
            )

        return content.strip()