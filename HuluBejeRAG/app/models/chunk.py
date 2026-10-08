from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeChunk:
    """
    Unified retrieval chunk.

    Every piece of knowledge entering the embedding/vector-store
    pipeline is represented by this model.

    Supported content types include:
        - text
        - structured_section
        - list
        - table
        - workflow
        - workflow_overview
        - workflow_sequence
        - workflow_decision
        - visual
    """

    chunk_id: str

    text: str

    document_id: str

    source: str = ""

    content_type: str = "text"

    page_numbers: list[int] = field(
        default_factory=list
    )

    section: str | None = None

    workflow_id: str | None = None

    workflow_name: str | None = None

    step_ids: list[str] = field(
        default_factory=list
    )

    decision_ids: list[str] = field(
        default_factory=list
    )

    transition_ids: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(
            self.text.split()
        )

    @property
    def is_workflow(self) -> bool:
        return (
            self.content_type == "workflow"
            or self.content_type.startswith(
                "workflow_"
            )
        )

    @property
    def has_source_page(self) -> bool:
        return bool(self.page_numbers)