from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.models.chunk import KnowledgeChunk
from app.chunking.workflow_chunker import WorkflowChunker


@dataclass
class DocumentChunkerConfig:
    """
    Configuration for document chunking.

    Normal text chunks:
        minimum = 140 tokens
        target  = 350 tokens
        maximum = 460 tokens
        overlap = 40 tokens
    """

    min_tokens: int = 140
    target_tokens: int = 350
    max_tokens: int = 460
    overlap_tokens: int = 40

    include_workflows: bool = True
    include_tables: bool = True
    include_visuals: bool = True
    include_normal_text: bool = True

    minimum_text_length: int = 40

    tokenizer_name: str = "BAAI/bge-large-en-v1.5"


class DocumentChunker:
    """
    Main document chunking coordinator.

    Pipeline:

        Document
            ↓
        RegionDetector
            ↓
        ┌─────────────────────────────┐
        │ Normal text                │
        │ token-aware 140–460 tokens │
        └─────────────────────────────┘
            ↓
        WorkflowChunker
            ↓
        Table chunks
            ↓
        Visual chunks
            ↓
        KnowledgeChunk
    """

    NORMAL_TEXT_REGION_TYPES = {
        "paragraph",
        "structured_section",
        "list",
        "heading",
        "unknown",
    }

    WORKFLOW_REGION_TYPES = {
        "workflow_definition",
    }

    TABLE_REGION_TYPES = {
        "table_candidate",
    }

    VISUAL_REGION_TYPES = {
        "visual",
    }

    def __init__(
        self,
        workflow_chunker: WorkflowChunker | None = None,
        chunk_assembler: Any | None = None,
        config: DocumentChunkerConfig | None = None,
    ) -> None:

        self.config = (
            config
            or DocumentChunkerConfig()
        )

        self.workflow_chunker = (
            workflow_chunker
            or WorkflowChunker()
        )

        # Kept for compatibility with the rest of
        # the project, but normal chunks are now
        # assembled directly into KnowledgeChunk.
        self.chunk_assembler = (
            chunk_assembler
        )

        self.tokenizer = self._load_tokenizer()

    # =========================================================
    # TOKENIZER
    # =========================================================

    def _load_tokenizer(self) -> Any:

        try:

            from transformers import AutoTokenizer

            tokenizer = (
                AutoTokenizer.from_pretrained(
                    self.config.tokenizer_name
                )
            )

            return tokenizer

        except Exception as exc:

            print(
                "[DocumentChunker] WARNING: "
                "Could not load HuggingFace tokenizer."
            )

            print(
                f"[DocumentChunker] Reason: {exc}"
            )

            print(
                "[DocumentChunker] Falling back "
                "to whitespace tokenization."
            )

            return None

    # =========================================================
    # PUBLIC API
    # =========================================================

    def chunk_document(
        self,
        document: Any,
        regions: list[Any] | None = None,
        workflow_provider: (
            Callable[
                [Any, Any],
                list[Any],
            ]
            | None
        ) = None,
    ) -> list[KnowledgeChunk]:

        regions = regions or []

        knowledge_chunks: list[
            KnowledgeChunk
        ] = []

        # -----------------------------------------------------
        # NORMAL TEXT
        # -----------------------------------------------------

        if self.config.include_normal_text:

            text_chunks = (
                self._build_text_chunks(
                    document=document,
                    regions=regions,
                )
            )

            knowledge_chunks.extend(
                self._convert_text_chunks(
                    document,
                    text_chunks,
                )
            )

        # -----------------------------------------------------
        # TABLES
        # -----------------------------------------------------

        if self.config.include_tables:

            table_chunks = (
                self._build_table_chunks(
                    document=document,
                    regions=regions,
                )
            )

            knowledge_chunks.extend(
                self._convert_generic_chunks(
                    document,
                    table_chunks,
                    content_type="table",
                )
            )

        # -----------------------------------------------------
        # VISUALS
        # -----------------------------------------------------

        if self.config.include_visuals:

            visual_chunks = (
                self._build_visual_chunks(
                    document=document,
                    regions=regions,
                )
            )

            knowledge_chunks.extend(
                self._convert_generic_chunks(
                    document,
                    visual_chunks,
                    content_type="visual",
                )
            )

        # -----------------------------------------------------
        # WORKFLOWS
        # -----------------------------------------------------

        if (
            self.config.include_workflows
            and workflow_provider is not None
        ):

            workflow_chunks = (
                self._extract_workflow_chunks(
                    document=document,
                    regions=regions,
                    workflow_provider=(
                        workflow_provider
                    ),
                )
            )

            knowledge_chunks.extend(
                self._convert_workflow_chunks(
                    document,
                    workflow_chunks,
                )
            )

        # -----------------------------------------------------
        # DEDUPLICATE
        # -----------------------------------------------------

        knowledge_chunks = (
            self._deduplicate(
                knowledge_chunks
            )
        )

        # -----------------------------------------------------
        # SORT
        # -----------------------------------------------------

        knowledge_chunks.sort(
            key=lambda chunk: (
                min(
                    chunk.page_numbers
                    or [999999]
                ),
                self._content_type_order(
                    chunk.content_type
                ),
                chunk.chunk_id,
            )
        )

        return knowledge_chunks

    # =========================================================
    # NORMAL TEXT
    # =========================================================

    def _build_text_chunks(
        self,
        document: Any,
        regions: list[Any],
    ) -> list[dict[str, Any]]:

        candidate_regions = [
            region
            for region in regions
            if self._region_type(region)
            in self.NORMAL_TEXT_REGION_TYPES
        ]

        candidate_regions.sort(
            key=lambda region: (
                self._page_number(region),
                self._region_top(region),
            )
        )

        chunks: list[
            dict[str, Any]
        ] = []

        current_blocks: list[
            dict[str, Any]
        ] = []

        current_tokens = 0

        for region in candidate_regions:

            text = self._clean_text(
                getattr(
                    region,
                    "text",
                    "",
                )
            )

            if not text:
                continue

            if (
                len(text)
                < self.config.minimum_text_length
            ):
                continue

            block = (
                self._region_to_block(
                    region
                )
            )

            region_tokens = (
                block["token_count"]
            )

            # -------------------------------------------------
            # Oversized region
            # -------------------------------------------------

            if (
                region_tokens
                > self.config.max_tokens
            ):

                if current_blocks:

                    chunks.append(
                        self._make_text_chunk(
                            current_blocks
                        )
                    )

                    current_blocks = []
                    current_tokens = 0

                large_chunks = (
                    self._split_large_region(
                        region
                    )
                )

                chunks.extend(
                    large_chunks
                )

                continue

            # -------------------------------------------------
            # First block
            # -------------------------------------------------

            if not current_blocks:

                current_blocks = [
                    block
                ]

                current_tokens = (
                    region_tokens
                )

                continue

            proposed = (
                current_tokens
                + region_tokens
            )

            # -------------------------------------------------
            # Ideal target
            # -------------------------------------------------

            if (
                proposed
                <= self.config.target_tokens
            ):

                current_blocks.append(
                    block
                )

                current_tokens = (
                    proposed
                )

                continue

            # -------------------------------------------------
            # We are below minimum.
            # Allow the chunk to grow past target,
            # but NEVER past max.
            # -------------------------------------------------

            if (
                current_tokens
                < self.config.min_tokens
                and proposed
                <= self.config.max_tokens
            ):

                current_blocks.append(
                    block
                )

                current_tokens = (
                    proposed
                )

                continue

            # -------------------------------------------------
            # Close current chunk
            # -------------------------------------------------

            chunks.append(
                self._make_text_chunk(
                    current_blocks
                )
            )

            # -------------------------------------------------
            # Start next chunk with overlap
            # -------------------------------------------------

            overlap_blocks = (
                self._build_overlap_blocks(
                    current_blocks
                )
            )

            overlap_tokens = sum(
                block["token_count"]
                for block in overlap_blocks
            )

            current_blocks = (
                overlap_blocks
            )

            current_blocks.append(
                block
            )

            current_tokens = (
                overlap_tokens
                + region_tokens
            )

            # -------------------------------------------------
            # Hard maximum safety
            # -------------------------------------------------

            if (
                current_tokens
                > self.config.max_tokens
            ):

                if overlap_blocks:

                    chunks.append(
                        self._make_text_chunk(
                            overlap_blocks
                        )
                    )

                current_blocks = [
                    block
                ]

                current_tokens = (
                    region_tokens
                )

        # -----------------------------------------------------
        # Final chunk
        # -----------------------------------------------------

        if current_blocks:

            chunks.append(
                self._make_text_chunk(
                    current_blocks
                )
            )

        # -----------------------------------------------------
        # Merge small trailing chunks where possible
        # -----------------------------------------------------

        return self._merge_small_chunks(
            chunks
        )

    # =========================================================
    # REGION -> BLOCK
    # =========================================================

    def _region_to_block(
        self,
        region: Any,
    ) -> dict[str, Any]:

        text = self._clean_text(
            getattr(
                region,
                "text",
                "",
            )
        )

        return {
            "region": region,
            "text": text,
            "token_count": (
                self._count_tokens(
                    text
                )
            ),
            "page_number": (
                self._page_number(
                    region
                )
            ),
            "region_type": (
                self._region_type(
                    region
                )
            ),
            "section": (
                self._extract_section(
                    region
                )
            ),
        }

    # =========================================================
    # LARGE REGION
    # =========================================================

    def _split_large_region(
        self,
        region: Any,
    ) -> list[dict[str, Any]]:

        text = self._clean_text(
            getattr(
                region,
                "text",
                "",
            )
        )

        sentences = (
            self._split_sentences(
                text
            )
        )

        if not sentences:

            return []

        result: list[
            dict[str, Any]
        ] = []

        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:

            sentence_tokens = (
                self._count_tokens(
                    sentence
                )
            )

            # One sentence is itself > 460.
            if (
                sentence_tokens
                > self.config.max_tokens
            ):

                if current:

                    result.append(
                        self._make_text_chunk_from_text(
                            region,
                            " ".join(
                                current
                            ),
                        )
                    )

                    current = []
                    current_tokens = 0

                pieces = (
                    self._split_text_by_tokens(
                        sentence,
                        self.config.max_tokens,
                    )
                )

                for piece in pieces:

                    result.append(
                        self._make_text_chunk_from_text(
                            region,
                            piece,
                        )
                    )

                continue

            proposed = (
                current_tokens
                + sentence_tokens
            )

            if (
                current
                and proposed
                > self.config.max_tokens
            ):

                result.append(
                    self._make_text_chunk_from_text(
                        region,
                        " ".join(
                            current
                        ),
                    )
                )

                overlap = (
                    self._build_sentence_overlap(
                        current
                    )
                )

                if overlap:

                    current = [
                        overlap
                    ]

                    current_tokens = (
                        self._count_tokens(
                            overlap
                        )
                    )

                else:

                    current = []
                    current_tokens = 0

            current.append(
                sentence
            )

            current_tokens += (
                sentence_tokens
            )

            if (
                current_tokens
                >= self.config.target_tokens
            ):

                result.append(
                    self._make_text_chunk_from_text(
                        region,
                        " ".join(
                            current
                        ),
                    )
                )

                overlap = (
                    self._build_sentence_overlap(
                        current
                    )
                )

                if overlap:

                    current = [
                        overlap
                    ]

                    current_tokens = (
                        self._count_tokens(
                            overlap
                        )
                    )

                else:

                    current = []
                    current_tokens = 0

        if current:

            result.append(
                self._make_text_chunk_from_text(
                    region,
                    " ".join(
                        current
                    ),
                )
            )

        return result

    # =========================================================
    # TOKEN SPLITTING
    # =========================================================

    def _split_text_by_tokens(
        self,
        text: str,
        max_tokens: int,
    ) -> list[str]:

        if self.tokenizer is None:

            words = text.split()

            return [
                " ".join(
                    words[
                        i:i + max_tokens
                    ]
                )
                for i in range(
                    0,
                    len(words),
                    max_tokens,
                )
            ]

        try:

            token_ids = (
                self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )

            pieces = []

            for i in range(
                0,
                len(token_ids),
                max_tokens,
            ):

                ids = token_ids[
                    i:i + max_tokens
                ]

                decoded = (
                    self.tokenizer.decode(
                        ids,
                        skip_special_tokens=True,
                    )
                )

                decoded = self._clean_text(
                    decoded
                )

                if decoded:

                    pieces.append(
                        decoded
                    )

            return pieces

        except Exception:

            words = text.split()

            return [
                " ".join(
                    words[
                        i:i + max_tokens
                    ]
                )
                for i in range(
                    0,
                    len(words),
                    max_tokens,
                )
            ]

    # =========================================================
    # OVERLAP
    # =========================================================

    def _build_overlap_blocks(
        self,
        blocks: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:

        selected = []

        total = 0

        for block in reversed(
            blocks
        ):

            tokens = (
                block["token_count"]
            )

            if (
                total + tokens
                > self.config.overlap_tokens
            ):
                break

            selected.insert(
                0,
                block,
            )

            total += tokens

        return selected

    def _build_sentence_overlap(
        self,
        sentences: list[str],
    ) -> str:

        selected = []

        total = 0

        for sentence in reversed(
            sentences
        ):

            tokens = (
                self._count_tokens(
                    sentence
                )
            )

            if (
                total + tokens
                > self.config.overlap_tokens
            ):
                break

            selected.insert(
                0,
                sentence,
            )

            total += tokens

        return " ".join(
            selected
        )

    # =========================================================
    # TEXT CHUNK OBJECT
    # =========================================================

    def _make_text_chunk(
        self,
        blocks: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:

        text = "\n\n".join(
            block["text"]
            for block in blocks
        )

        pages = sorted(
            {
                block["page_number"]
                for block in blocks
                if block["page_number"]
                is not None
            }
        )

        sections = []

        for block in blocks:

            section = (
                block.get(
                    "section"
                )
            )

            if (
                section
                and section not in sections
            ):

                sections.append(
                    section
                )

        return {
            "text": text,
            "page_numbers": pages,
            "section": (
                sections[-1]
                if sections
                else None
            ),
            "metadata": {
                "chunking_strategy": (
                    "token_aware"
                ),
                "token_count": (
                    self._count_tokens(
                        text
                    )
                ),
                "min_tokens": (
                    self.config.min_tokens
                ),
                "target_tokens": (
                    self.config.target_tokens
                ),
                "max_tokens": (
                    self.config.max_tokens
                ),
                "overlap_tokens": (
                    self.config.overlap_tokens
                ),
                "source_region_count": (
                    len(blocks)
                ),
                "source_region_types": sorted(
                    {
                        block[
                            "region_type"
                        ]
                        for block in blocks
                    }
                ),
            },
        }

    def _make_text_chunk_from_text(
        self,
        region: Any,
        text: str,
    ) -> dict[str, Any]:

        text = self._clean_text(
            text
        )

        return {
            "text": text,
            "page_numbers": [
                self._page_number(
                    region
                )
            ],
            "section": (
                self._extract_section(
                    region
                )
            ),
            "metadata": {
                "chunking_strategy": (
                    "token_aware"
                ),
                "token_count": (
                    self._count_tokens(
                        text
                    )
                ),
                "min_tokens": (
                    self.config.min_tokens
                ),
                "target_tokens": (
                    self.config.target_tokens
                ),
                "max_tokens": (
                    self.config.max_tokens
                ),
                "overlap_tokens": (
                    self.config.overlap_tokens
                ),
                "source_region_count": 1,
                "source_region_types": [
                    self._region_type(
                        region
                    )
                ],
            },
        }

    # =========================================================
    # SMALL CHUNK MERGING
    # =========================================================

    def _merge_small_chunks(
        self,
        chunks: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:

        if len(chunks) <= 1:

            return chunks

        result = []

        for chunk in chunks:

            tokens = (
                self._count_tokens(
                    chunk["text"]
                )
            )

            if (
                tokens
                >= self.config.min_tokens
            ):

                result.append(
                    chunk
                )

                continue

            if not result:

                result.append(
                    chunk
                )

                continue

            previous = (
                result[-1]
            )

            previous_tokens = (
                self._count_tokens(
                    previous["text"]
                )
            )

            combined = (
                previous_tokens
                + tokens
            )

            if (
                combined
                <= self.config.max_tokens
            ):

                previous["text"] = (
                    previous["text"]
                    + "\n\n"
                    + chunk["text"]
                )

                previous[
                    "page_numbers"
                ] = sorted(
                    set(
                        previous.get(
                            "page_numbers",
                            [],
                        )
                        + chunk.get(
                            "page_numbers",
                            [],
                        )
                    )
                )

                previous[
                    "metadata"
                ]["token_count"] = (
                    self._count_tokens(
                        previous["text"]
                    )
                )

            else:

                result.append(
                    chunk
                )

        return result

    # =========================================================
    # TABLES
    # =========================================================

    def _build_table_chunks(
        self,
        document: Any,
        regions: list[Any],
    ) -> list[dict[str, Any]]:

        chunks = []

        for region in regions:

            if (
                self._region_type(
                    region
                )
                not in self.TABLE_REGION_TYPES
            ):
                continue

            rows = getattr(
                region,
                "rows",
                [],
            )

            text = self._render_table(
                rows
            )

            if not text:

                text = self._clean_text(
                    getattr(
                        region,
                        "text",
                        "",
                    )
                )

            if not text:
                continue

            chunks.append(
                {
                    "text": text,
                    "page_numbers": [
                        self._page_number(
                            region
                        )
                    ],
                    "section": (
                        self._extract_section(
                            region
                        )
                    ),
                    "metadata": {
                        "chunking_strategy": (
                            "table_preserving"
                        ),
                        "table": True,
                        "region_id": getattr(
                            region,
                            "region_id",
                            None,
                        ),
                    },
                }
            )

        return chunks

    def _render_table(
        self,
        rows: list[Any],
    ) -> str:

        rendered = []

        for row in rows:

            if isinstance(
                row,
                dict,
            ):

                cells = (
                    row.get(
                        "cells"
                    )
                    or row.get(
                        "values"
                    )
                    or row.get(
                        "columns"
                    )
                    or []
                )

            elif isinstance(
                row,
                (list, tuple),
            ):

                cells = row

            else:

                cells = [
                    row
                ]

            values = []

            for cell in cells:

                if isinstance(
                    cell,
                    dict,
                ):

                    value = (
                        cell.get(
                            "text"
                        )
                        or cell.get(
                            "value"
                        )
                        or ""
                    )

                else:

                    value = str(
                        cell
                    )

                value = self._clean_text(
                    value
                )

                if value:

                    values.append(
                        value
                    )

            if values:

                rendered.append(
                    " | ".join(
                        values
                    )
                )

        return "\n".join(
            rendered
        )

    # =========================================================
    # VISUAL
    # =========================================================

    def _build_visual_chunks(
        self,
        document: Any,
        regions: list[Any],
    ) -> list[dict[str, Any]]:

        chunks = []

        for region in regions:

            if (
                self._region_type(
                    region
                )
                not in self.VISUAL_REGION_TYPES
            ):
                continue

            text = self._clean_text(
                getattr(
                    region,
                    "text",
                    "",
                )
            )

            if not text:

                text = (
                    "Visual content detected "
                    f"on page "
                    f"{self._page_number(region)}."
                )

            chunks.append(
                {
                    "text": text,
                    "page_numbers": [
                        self._page_number(
                            region
                        )
                    ],
                    "section": (
                        self._extract_section(
                            region
                        )
                    ),
                    "metadata": {
                        "chunking_strategy": (
                            "visual_preserving"
                        ),
                        "visual": True,
                        "region_id": getattr(
                            region,
                            "region_id",
                            None,
                        ),
                        "confidence": getattr(
                            region,
                            "confidence",
                            None,
                        ),
                    },
                }
            )

        return chunks

    # =========================================================
    # WORKFLOWS
    # =========================================================

    def _extract_workflow_chunks(
        self,
        document: Any,
        regions: list[Any],
        workflow_provider: Callable[
            [Any, Any],
            list[Any],
        ],
    ) -> list[Any]:

        workflow_chunks = []

        seen_workflows: set[
            str
        ] = set()

        for region in regions:

            if (
                self._region_type(
                    region
                )
                not in self.WORKFLOW_REGION_TYPES
            ):
                continue

            try:

                workflows = (
                    workflow_provider(
                        document,
                        region,
                    )
                    or []
                )

            except Exception as exc:

                print(
                    "[DocumentChunker] "
                    f"Workflow extraction failed "
                    f"on page "
                    f"{self._page_number(region)}: "
                    f"{exc}"
                )

                continue

            for workflow in workflows:

                workflow_id = getattr(
                    workflow,
                    "workflow_id",
                    None,
                )

                if (
                    workflow_id
                    and workflow_id
                    in seen_workflows
                ):
                    continue

                if workflow_id:

                    seen_workflows.add(
                        workflow_id
                    )

                try:

                    generated = (
                        self.workflow_chunker.chunk(
                            workflow
                        )
                    )

                    workflow_chunks.extend(
                        generated
                    )

                except Exception as exc:

                    print(
                        "[DocumentChunker] "
                        "Workflow chunking failed: "
                        f"{exc}"
                    )

        return workflow_chunks

    # =========================================================
    # CONVERT NORMAL CHUNKS
    # =========================================================

    def _convert_text_chunks(
        self,
        document: Any,
        chunks: list[
            dict[str, Any]
        ],
    ) -> list[KnowledgeChunk]:

        result = []

        for index, chunk in enumerate(
            chunks
        ):

            text = self._clean_text(
                chunk["text"]
            )

            if not text:
                continue

            page_numbers = [
                page
                for page in chunk.get(
                    "page_numbers",
                    [],
                )
                if page is not None
            ]

            chunk_id = (
                f"{document.document_id}"
                f"_text_{index + 1}"
            )

            metadata = dict(
                chunk.get(
                    "metadata",
                    {},
                )
            )

            metadata.update(
                {
                    "document_id": (
                        document.document_id
                    ),
                    "source": document.source,
                    "content_type": "text",
                    "page_numbers": (
                        page_numbers
                    ),
                }
            )

            result.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    text=text,
                    document_id=(
                        document.document_id
                    ),
                    source=document.source,
                    content_type="text",
                    page_numbers=page_numbers,
                    section=chunk.get(
                        "section"
                    ),
                    metadata=metadata,
                )
            )

        return result

    # =========================================================
    # CONVERT TABLE/VISUAL
    # =========================================================

    def _convert_generic_chunks(
        self,
        document: Any,
        chunks: list[
            dict[str, Any]
        ],
        content_type: str,
    ) -> list[KnowledgeChunk]:

        result = []

        for index, chunk in enumerate(
            chunks
        ):

            text = self._clean_text(
                chunk["text"]
            )

            if not text:
                continue

            page_numbers = [
                page
                for page in chunk.get(
                    "page_numbers",
                    [],
                )
                if page is not None
            ]

            chunk_id = (
                f"{document.document_id}"
                f"_{content_type}_"
                f"{index + 1}"
            )

            metadata = dict(
                chunk.get(
                    "metadata",
                    {},
                )
            )

            metadata.update(
                {
                    "document_id": (
                        document.document_id
                    ),
                    "source": document.source,
                    "content_type": (
                        content_type
                    ),
                    "page_numbers": (
                        page_numbers
                    ),
                }
            )

            result.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    text=text,
                    document_id=(
                        document.document_id
                    ),
                    source=document.source,
                    content_type=(
                        content_type
                    ),
                    page_numbers=page_numbers,
                    section=chunk.get(
                        "section"
                    ),
                    metadata=metadata,
                )
            )

        return result

    # =========================================================
    # CONVERT WORKFLOW CHUNKS
    # =========================================================

    def _convert_workflow_chunks(
        self,
        document: Any,
        chunks: list[Any],
    ) -> list[KnowledgeChunk]:

        result = []

        for index, chunk in enumerate(
            chunks
        ):

            text = self._clean_text(
                getattr(
                    chunk,
                    "text",
                    "",
                )
            )

            if not text:
                continue

            page_numbers = list(
                getattr(
                    chunk,
                    "page_numbers",
                    [],
                )
                or []
            )

            metadata = dict(
                getattr(
                    chunk,
                    "metadata",
                    {},
                )
                or {}
            )

            workflow_id = getattr(
                chunk,
                "workflow_id",
                None,
            )

            workflow_name = getattr(
                chunk,
                "workflow_name",
                None,
            )

            step_ids = list(
                getattr(
                    chunk,
                    "step_ids",
                    [],
                )
                or []
            )

            decision_ids = list(
                getattr(
                    chunk,
                    "decision_ids",
                    [],
                )
                or []
            )

            transition_ids = list(
                getattr(
                    chunk,
                    "transition_ids",
                    [],
                )
                or []
            )

            metadata.update(
                {
                    "document_id": (
                        document.document_id
                    ),
                    "source": document.source,
                    "content_type": (
                        "workflow"
                    ),
                    "workflow_id": (
                        workflow_id
                    ),
                    "workflow_name": (
                        workflow_name
                    ),
                    "step_ids": step_ids,
                    "decision_ids": (
                        decision_ids
                    ),
                    "transition_ids": (
                        transition_ids
                    ),
                    "page_numbers": (
                        page_numbers
                    ),
                }
            )

            chunk_id = (
                f"{document.document_id}"
                f"_workflow_"
                f"{index + 1}"
            )

            result.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    text=text,
                    document_id=(
                        document.document_id
                    ),
                    source=document.source,
                    content_type="workflow",
                    page_numbers=page_numbers,
                    workflow_id=workflow_id,
                    workflow_name=(
                        workflow_name
                    ),
                    step_ids=step_ids,
                    decision_ids=(
                        decision_ids
                    ),
                    transition_ids=(
                        transition_ids
                    ),
                    metadata=metadata,
                )
            )

        return result

    # =========================================================
    # WORKFLOW PROVIDER COMPATIBILITY
    # =========================================================

    @staticmethod
    def call_workflow_extractor(
        extractor: Any,
        region: Any,
        document_id: str,
    ) -> list[Any]:

        """
        Compatibility adapter for the existing
        WorkflowExtractor.

        We intentionally support several possible method
        names because the extractor has already evolved
        during development.
        """

        method_names = [
            "extract_workflows",
            "extract_workflow",
            "process",
            "detect",
            "parse",
            "build_workflows",
        ]

        for method_name in method_names:

            method = getattr(
                extractor,
                method_name,
                None,
            )

            if method is None:
                continue

            attempts = [
                (
                    [region],
                    document_id,
                ),
                (
                    region,
                    document_id,
                ),
                (
                    [region],
                ),
                (
                    region,
                ),
            ]

            for args in attempts:

                try:

                    result = method(
                        *args
                    )

                    if result is None:

                        return []

                    if isinstance(
                        result,
                        list,
                    ):

                        return result

                    if isinstance(
                        result,
                        tuple,
                    ):

                        return list(
                            result
                        )

                    return [result]

                except TypeError:

                    continue

        raise AttributeError(
            "WorkflowExtractor does not expose "
            "a supported workflow extraction method."
        )

    # =========================================================
    # DEDUPLICATION
    # =========================================================

    @staticmethod
    def _deduplicate(
        chunks: list[KnowledgeChunk],
    ) -> list[KnowledgeChunk]:

        seen: set[str] = set()

        result = []

        for chunk in chunks:

            normalized = (
                " ".join(
                    chunk.text.lower().split()
                )
            )

            if not normalized:
                continue

            key = (
                f"{chunk.content_type}:"
                f"{normalized}"
            )

            if key in seen:
                continue

            seen.add(key)

            result.append(
                chunk
            )

        return result

    # =========================================================
    # STATISTICS
    # =========================================================

    @staticmethod
    def summarize(
        chunks: list[KnowledgeChunk],
    ) -> dict[str, Any]:

        statistics = {
            "total_chunks": len(
                chunks
            ),
            "workflow_chunks": 0,
            "total_characters": 0,
            "total_words": 0,
            "total_tokens": 0,
            "by_type": {},
        }

        for chunk in chunks:

            content_type = (
                chunk.content_type
            )

            statistics[
                "by_type"
            ][content_type] = (
                statistics[
                    "by_type"
                ].get(
                    content_type,
                    0,
                )
                + 1
            )

            if chunk.is_workflow:

                statistics[
                    "workflow_chunks"
                ] += 1

            statistics[
                "total_characters"
            ] += chunk.character_count

            statistics[
                "total_words"
            ] += chunk.word_count

            token_count = (
                chunk.metadata.get(
                    "token_count",
                    0,
                )
            )

            try:

                statistics[
                    "total_tokens"
                ] += int(
                    token_count
                )

            except (
                TypeError,
                ValueError,
            ):

                pass

        return statistics

    # =========================================================
    # HELPERS
    # =========================================================

    def _count_tokens(
        self,
        text: str,
    ) -> int:

        if not text:
            return 0

        if self.tokenizer is None:

            return len(
                text.split()
            )

        try:

            return len(
                self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )

        except Exception:

            return len(
                text.split()
            )

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:

        import re

        text = self._clean_text(
            text
        )

        if not text:
            return []

        parts = re.split(
            r"(?<=[.!?])\s+|"
            r"(?<=:)\s+(?=[A-Z0-9])|"
            r"\n+",
            text,
        )

        return [
            self._clean_text(
                part
            )
            for part in parts
            if self._clean_text(
                part
            )
        ]

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        import re

        if not text:
            return ""

        text = str(text)

        text = text.replace(
            "\x00",
            " ",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    @staticmethod
    def _region_type(
        region: Any,
    ) -> str:

        return str(
            getattr(
                region,
                "region_type",
                "unknown",
            )
        ).lower()

    @staticmethod
    def _page_number(
        region: Any,
    ) -> int:

        value = getattr(
            region,
            "page_number",
            0,
        )

        try:

            return int(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0

    @staticmethod
    def _region_top(
        region: Any,
    ) -> float:

        bbox = getattr(
            region,
            "bbox",
            None,
        )

        if bbox is None:
            return 0.0

        try:

            return float(
                getattr(
                    bbox,
                    "top",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    @staticmethod
    def _extract_section(
        region: Any,
    ) -> str | None:

        metadata = getattr(
            region,
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            return None

        section = (
            metadata.get(
                "section"
            )
            or metadata.get(
                "parent_section"
            )
            or metadata.get(
                "section_title"
            )
        )

        if section:

            return str(
                section
            ).strip()

        return None

    @staticmethod
    def _content_type_order(
        content_type: str,
    ) -> int:

        order = {
            "text": 1,
            "table": 2,
            "workflow": 3,
            "visual": 4,
        }

        return order.get(
            content_type,
            99,
        )