from __future__ import annotations

from app.retrieval.retriever import RetrievedChunk


class ContextBuilder:
    """
    Builds the knowledge context that will be provided
    to the answer-generation LLM.
    """

    def build(
        self,
        chunks: list[RetrievedChunk],
    ) -> str:

        if not chunks:
            return ""

        context_parts: list[str] = []

        for index, chunk in enumerate(chunks, start=1):

            source = chunk.metadata.get(
                "source",
                "Unknown source",
            )

            page = chunk.metadata.get(
                "page",
                "Unknown page",
            )

            section = chunk.metadata.get(
                "section",
                "",
            )

            subsection = chunk.metadata.get(
                "subsection",
                "",
            )

            header = (
                f"[Source: {source} | "
                f"Page: {page}"
            )

            if section:
                header += f" | Section: {section}"

            if subsection:
                header += (
                    f" | Subsection: "
                    f"{subsection}"
                )

            header += "]"

            context_parts.append(
                f"Context {index}\n"
                f"{header}\n\n"
                f"{chunk.text.strip()}"
            )

        return "\n\n".join(context_parts)