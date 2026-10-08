from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.models.chunk import KnowledgeChunk
from app.models.workflow import Workflow


@dataclass
class ChunkAssemblerConfig:
    """
    Configuration for assembling unified knowledge chunks.
    """

    minimum_text_length: int = 40

    include_page_context: bool = True

    include_source_context: bool = True

    deduplicate_chunks: bool = True


class ChunkAssembler:
    """
    Converts extracted document knowledge into one unified
    KnowledgeChunk representation.

    Sources can include:

        normal text
        structured sections
        lists
        tables
        workflows
        workflow decisions
        workflow sequences
        visual descriptions

    The assembler does NOT create embeddings.

    Pipeline:

        extracted content
              ↓
        ChunkAssembler
              ↓
        KnowledgeChunk[]
              ↓
        EmbeddingService
              ↓
        FAISS
    """

    def __init__(
        self,
        config: ChunkAssemblerConfig | None = None,
    ):
        self.config = (
            config
            or ChunkAssemblerConfig()
        )

    # ================================================================
    # PUBLIC API
    # ================================================================

    def assemble(
        self,
        document_id: str,
        source: str,
        text_chunks: Iterable[Any] | None = None,
        workflow_chunks: Iterable[Any] | None = None,
        table_chunks: Iterable[Any] | None = None,
        visual_chunks: Iterable[Any] | None = None,
    ) -> list[KnowledgeChunk]:
        """
        Assemble every extracted content type into one list.
        """

        chunks: list[KnowledgeChunk] = []

        if text_chunks:
            chunks.extend(
                self._assemble_text_chunks(
                    document_id=document_id,
                    source=source,
                    chunks=text_chunks,
                )
            )

        if workflow_chunks:
            chunks.extend(
                self._assemble_workflow_chunks(
                    document_id=document_id,
                    source=source,
                    chunks=workflow_chunks,
                )
            )

        if table_chunks:
            chunks.extend(
                self._assemble_generic_chunks(
                    document_id=document_id,
                    source=source,
                    chunks=table_chunks,
                    default_type="table",
                )
            )

        if visual_chunks:
            chunks.extend(
                self._assemble_generic_chunks(
                    document_id=document_id,
                    source=source,
                    chunks=visual_chunks,
                    default_type="visual",
                )
            )

        if self.config.deduplicate_chunks:
            chunks = self._deduplicate(
                chunks
            )

        chunks = self._sort_chunks(
            chunks
        )

        self._finalize_metadata(
            chunks
        )

        return chunks

    # ================================================================
    # TEXT
    # ================================================================

    def _assemble_text_chunks(
        self,
        document_id: str,
        source: str,
        chunks: Iterable[Any],
    ) -> list[KnowledgeChunk]:
        result: list[KnowledgeChunk] = []

        for index, item in enumerate(
            chunks
        ):
            text = self._extract_text(
                item
            )

            if not self._valid_text(
                text
            ):
                continue

            page_numbers = (
                self._extract_page_numbers(
                    item
                )
            )

            content_type = (
                self._extract_content_type(
                    item,
                    default="text",
                )
            )

            metadata = self._extract_metadata(
                item
            )

            chunk_id = (
                self._extract_id(item)
                or self._stable_id(
                    document_id,
                    content_type,
                    text,
                    page_numbers,
                    index,
                )
            )

            result.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    text=self._add_context(
                        text,
                        source,
                        page_numbers,
                    ),
                    document_id=document_id,
                    source=source,
                    content_type=content_type,
                    page_numbers=page_numbers,
                    section=self._extract_section(
                        item
                    ),
                    metadata=metadata,
                )
            )

        return result

    # ================================================================
    # WORKFLOWS
    # ================================================================

    def _assemble_workflow_chunks(
        self,
        document_id: str,
        source: str,
        chunks: Iterable[Any],
    ) -> list[KnowledgeChunk]:
        result: list[KnowledgeChunk] = []

        for index, item in enumerate(
            chunks
        ):
            text = self._extract_text(
                item
            )

            if not self._valid_text(
                text
            ):
                continue

            metadata = self._extract_metadata(
                item
            )

            workflow_id = self._optional_string(
                getattr(
                    item,
                    "workflow_id",
                    None,
                )
            )

            workflow_name = self._optional_string(
                getattr(
                    item,
                    "workflow_name",
                    None,
                )
            )

            step_ids = self._list_value(
                getattr(
                    item,
                    "step_ids",
                    [],
                )
            )

            decision_ids = self._list_value(
                getattr(
                    item,
                    "decision_ids",
                    [],
                )
            )

            transition_ids = self._list_value(
                getattr(
                    item,
                    "transition_ids",
                    [],
                )
            )

            page_numbers = (
                self._list_value(
                    getattr(
                        item,
                        "page_numbers",
                        [],
                    )
                )
            )

            content_type = (
                self._extract_content_type(
                    item,
                    default="workflow",
                )
            )

            metadata.update(
                {
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_name,
                    "step_ids": step_ids,
                    "decision_ids": decision_ids,
                    "transition_ids": transition_ids,
                }
            )

            if workflow_name:
                text = (
                    f"Workflow: "
                    f"{workflow_name}\n\n"
                    f"{text}"
                )

            chunk_id = (
                self._extract_id(item)
                or self._stable_id(
                    document_id,
                    content_type,
                    text,
                    page_numbers,
                    index,
                )
            )

            result.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    text=self._add_context(
                        text,
                        source,
                        page_numbers,
                    ),
                    document_id=document_id,
                    source=source,
                    content_type=content_type,
                    page_numbers=page_numbers,
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    step_ids=step_ids,
                    decision_ids=decision_ids,
                    transition_ids=transition_ids,
                    metadata=metadata,
                )
            )

        return result

    # ================================================================
    # GENERIC CONTENT
    # ================================================================

    def _assemble_generic_chunks(
        self,
        document_id: str,
        source: str,
        chunks: Iterable[Any],
        default_type: str,
    ) -> list[KnowledgeChunk]:
        result: list[KnowledgeChunk] = []

        for index, item in enumerate(
            chunks
        ):
            text = self._extract_text(
                item
            )

            if not self._valid_text(
                text
            ):
                continue

            page_numbers = (
                self._extract_page_numbers(
                    item
                )
            )

            content_type = (
                self._extract_content_type(
                    item,
                    default=default_type,
                )
            )

            metadata = self._extract_metadata(
                item
            )

            chunk_id = (
                self._extract_id(item)
                or self._stable_id(
                    document_id,
                    content_type,
                    text,
                    page_numbers,
                    index,
                )
            )

            result.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    text=self._add_context(
                        text,
                        source,
                        page_numbers,
                    ),
                    document_id=document_id,
                    source=source,
                    content_type=content_type,
                    page_numbers=page_numbers,
                    metadata=metadata,
                )
            )

        return result

    # ================================================================
    # WORKFLOW OBJECT → CHUNKS
    # ================================================================

    def assemble_workflow(
        self,
        workflow: Workflow,
        workflow_chunks: Iterable[Any],
        source: str,
    ) -> list[KnowledgeChunk]:
        """
        Convenience method for a single workflow.

        Useful when processing one diagram/workflow at a time.
        """

        return self._assemble_workflow_chunks(
            document_id=workflow.document_id,
            source=source,
            chunks=workflow_chunks,
        )

    # ================================================================
    # CONTEXT
    # ================================================================

    def _add_context(
        self,
        text: str,
        source: str,
        page_numbers: list[int],
    ) -> str:
        value = text.strip()

        if not value:
            return value

        prefix: list[str] = []

        if (
            self.config.include_source_context
            and source
        ):
            prefix.append(
                f"Source: {source}"
            )

        if (
            self.config.include_page_context
            and page_numbers
        ):
            pages = ", ".join(
                str(page)
                for page in page_numbers
            )

            prefix.append(
                f"Page: {pages}"
            )

        if not prefix:
            return value

        return (
            "\n".join(prefix)
            + "\n\n"
            + value
        )

    # ================================================================
    # EXTRACTION HELPERS
    # ================================================================

    def _extract_text(
        self,
        item: Any,
    ) -> str:
        if isinstance(
            item,
            str,
        ):
            return item.strip()

        text = getattr(
            item,
            "text",
            None,
        )

        if text is not None:
            return str(
                text
            ).strip()

        content = getattr(
            item,
            "content",
            None,
        )

        if content is not None:
            if isinstance(
                content,
                str,
            ):
                return content.strip()

            return str(
                content
            ).strip()

        return ""

    def _extract_id(
        self,
        item: Any,
    ) -> str | None:
        value = getattr(
            item,
            "chunk_id",
            None,
        )

        if value:
            return str(
                value
            )

        value = getattr(
            item,
            "block_id",
            None,
        )

        if value:
            return str(
                value
            )

        return None

    def _extract_content_type(
        self,
        item: Any,
        default: str,
    ) -> str:
        value = getattr(
            item,
            "content_type",
            None,
        )

        if value:
            return str(
                value
            )

        value = getattr(
            item,
            "chunk_type",
            None,
        )

        if value:
            return str(
                value
            )

        value = getattr(
            item,
            "block_type",
            None,
        )

        if value:
            return str(
                value
            )

        return default

    def _extract_page_numbers(
        self,
        item: Any,
    ) -> list[int]:
        value = getattr(
            item,
            "page_numbers",
            None,
        )

        if value:
            return self._list_value(
                value
            )

        page = getattr(
            item,
            "page_number",
            None,
        )

        if page is not None:
            try:
                return [
                    int(page)
                ]
            except (
                ValueError,
                TypeError,
            ):
                pass

        return []

    def _extract_section(
        self,
        item: Any,
    ) -> str | None:
        value = getattr(
            item,
            "section",
            None,
        )

        if value:
            return str(
                value
            )

        metadata = getattr(
            item,
            "metadata",
            None,
        )

        if isinstance(
            metadata,
            dict,
        ):
            section = metadata.get(
                "section"
            )

            if section:
                return str(
                    section
                )

        return None

    def _extract_metadata(
        self,
        item: Any,
    ) -> dict[str, Any]:
        metadata = getattr(
            item,
            "metadata",
            None,
        )

        if isinstance(
            metadata,
            dict,
        ):
            return dict(
                metadata
            )

        return {}

    # ================================================================
    # VALIDATION
    # ================================================================

    def _valid_text(
        self,
        text: str,
    ) -> bool:
        normalized = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return (
            len(normalized)
            >= self.config.minimum_text_length
        )

    # ================================================================
    # DEDUPLICATION
    # ================================================================

    def _deduplicate(
        self,
        chunks: list[KnowledgeChunk],
    ) -> list[KnowledgeChunk]:
        result: list[
            KnowledgeChunk
        ] = []

        seen_exact: set[str] = set()

        for chunk in chunks:
            normalized = self._normalize_for_hash(
                chunk.text
            )

            digest = hashlib.sha256(
                normalized.encode(
                    "utf-8"
                )
            ).hexdigest()

            if digest in seen_exact:
                continue

            seen_exact.add(
                digest
            )

            result.append(
                chunk
            )

        return result

    def _normalize_for_hash(
        self,
        text: str,
    ) -> str:
        return re.sub(
            r"\s+",
            " ",
            text.lower(),
        ).strip()

    # ================================================================
    # SORTING
    # ================================================================

    def _sort_chunks(
        self,
        chunks: list[KnowledgeChunk],
    ) -> list[KnowledgeChunk]:
        type_priority = {
            "text": 10,
            "paragraph": 10,
            "structured_section": 20,
            "list": 30,
            "table": 40,
            "visual": 50,
            "workflow_overview": 60,
            "workflow_sequence": 70,
            "workflow_decision": 80,
            "workflow": 90,
        }

        return sorted(
            chunks,
            key=lambda chunk: (
                min(
                    chunk.page_numbers
                )
                if chunk.page_numbers
                else 999999,
                type_priority.get(
                    chunk.content_type,
                    100,
                ),
                chunk.chunk_id,
            ),
        )

    # ================================================================
    # METADATA
    # ================================================================

    def _finalize_metadata(
        self,
        chunks: list[KnowledgeChunk],
    ) -> None:
        total = len(
            chunks
        )

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            chunk.metadata[
                "chunk_index"
            ] = index

            chunk.metadata[
                "total_chunks"
            ] = total

            chunk.metadata[
                "document_id"
            ] = chunk.document_id

            chunk.metadata[
                "source"
            ] = chunk.source

            chunk.metadata[
                "content_type"
            ] = chunk.content_type

            chunk.metadata[
                "character_count"
            ] = chunk.character_count

            chunk.metadata[
                "word_count"
            ] = chunk.word_count

            chunk.metadata[
                "page_numbers"
            ] = list(
                chunk.page_numbers
            )

    # ================================================================
    # IDS
    # ================================================================

    def _stable_id(
        self,
        document_id: str,
        content_type: str,
        text: str,
        page_numbers: list[int],
        index: int,
    ) -> str:
        raw = (
            f"{document_id}|"
            f"{content_type}|"
            f"{','.join(map(str, page_numbers))}|"
            f"{text}|"
            f"{index}"
        )

        digest = hashlib.sha1(
            raw.encode(
                "utf-8"
            )
        ).hexdigest()[:16]

        return (
            f"chunk_{digest}"
        )

    # ================================================================
    # TYPE HELPERS
    # ================================================================

    def _list_value(
        self,
        value: Any,
    ) -> list[Any]:
        if value is None:
            return []

        if isinstance(
            value,
            list,
        ):
            return list(
                value
            )

        if isinstance(
            value,
            tuple,
        ):
            return list(
                value
            )

        return [value]

    def _optional_string(
        self,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        value = str(
            value
        ).strip()

        return value or None