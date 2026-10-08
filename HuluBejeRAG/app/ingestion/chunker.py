from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from transformers import AutoTokenizer

from app.models.chunk import Chunk
from app.models.document import Document, Page


# ============================================================
# TEXT UNIT
# ============================================================

@dataclass
class TextUnit:
    """
    Smallest semantic unit used during chunk construction.

    A unit can represent:
        - normal prose
        - heading
        - list
        - workflow step
        - decision
        - diagram-like extracted text
        - table-like content
    """

    text: str
    page_number: int
    section: str | None = None
    subsection: str | None = None

    # Used to preserve semantic structure.
    unit_type: Literal[
        "text",
        "heading",
        "list",
        "workflow",
        "decision",
        "table",
        "diagram",
    ] = "text"


# ============================================================
# CHUNKER
# ============================================================

class Chunker:
    """
    Structure-aware tokenizer-based chunker for BGE-large.

    Designed for RAG documents that may contain:

        - normal paragraphs
        - headings
        - numbered procedures
        - bullet lists
        - workflows
        - flowchart text
        - decision trees
        - table-like extracted text
        - diagram-like extracted text

    BGE-large maximum:
        512 tokens

    Chunk policy:

        Minimum:
            130 tokens

        Target:
            420 tokens

        Maximum:
            450 tokens

        Overlap:
            60 tokens

    IMPORTANT:
        The chunker works on the text available in Page.text.

        If a PDF contains a purely visual diagram with NO
        meaningful text extraction, this class cannot magically
        understand the pixels. The PDF ingestion layer must first
        OCR / analyze the page image and place the resulting
        semantic description into Page.text.
    """

    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    MIN_TOKENS = 130
    TARGET_TOKENS = 420
    MAX_TOKENS = 450
    OVERLAP_TOKENS = 60

    # Avoid creating tiny semantic pieces.
    MAX_HEADING_WORDS = 12

    # Workflow keywords commonly found in process documents.
    WORKFLOW_KEYWORDS = (
        "workflow",
        "process",
        "procedure",
        "flow",
        "flowchart",
        "process flow",
        "business process",
        "approval process",
        "request process",
        "steps",
        "procedure steps",
    )

    DECISION_PATTERNS = (
        r"\bif\b",
        r"\belse\b",
        r"\byes\b.*\bno\b",
        r"\bno\b.*\byes\b",
        r"\bdecision\b",
        r"\bcondition\b",
        r"\bapproved\b.*\brejected\b",
        r"\brejected\b.*\bapproved\b",
        r"\btrue\b.*\bfalse\b",
        r"\bpass\b.*\bfail\b",
    )

    STEP_PATTERN = re.compile(
        r"""
        ^
        (?:
            \d+[\.\)] |
            \d+\.\d+[\.\)]? |
            [A-Za-z][\.\)] |
            [-*•▪◦] |
            step\s+\d+ |
            stage\s+\d+
        )
        \s+
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    ARROW_PATTERN = re.compile(
        r"(?:->|=>|→|➡|➜|➔|↓|↑|←|→|↳|⇒)"
    )

    def __init__(self) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def chunk_document(
        self,
        document: Document,
    ) -> list[Chunk]:

        raw_chunks: list[
            tuple[
                str,
                int,
                str | None,
                str | None,
                str,
            ]
        ] = []

        # --------------------------------------------------------
        # Process page by page.
        # --------------------------------------------------------

        for page in document.pages:

            units = self._extract_units(page)

            page_chunks = self._build_chunks(units)

            raw_chunks.extend(page_chunks)

        # --------------------------------------------------------
        # Merge small chunks.
        # --------------------------------------------------------

        raw_chunks = self._merge_small_chunks(
            raw_chunks
        )

        # --------------------------------------------------------
        # Build final Chunk objects.
        # --------------------------------------------------------

        chunks: list[Chunk] = []

        for index, (
            text,
            page_number,
            section,
            subsection,
            chunk_type,
        ) in enumerate(raw_chunks):

            token_count = self._token_count(text)

            metadata = dict(document.metadata)

            metadata.update(
                {
                    "document_id": document.document_id,
                    "source": document.source,
                    "page": str(page_number),
                    "chunk_index": str(index),

                    # Important for RAG filtering/debugging.
                    "chunk_type": chunk_type,
                }
            )

            if section:
                metadata["section"] = section

            if subsection:
                metadata["subsection"] = subsection

            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document.document_id}"
                        f"_chunk_{index}"
                    ),
                    document_id=document.document_id,
                    text=text,
                    metadata=metadata,
                    token_count=token_count,
                )
            )

        return chunks

    # ============================================================
    # MULTIPLE DOCUMENTS
    # ============================================================

    def chunk_documents(
        self,
        documents: list[Document],
    ) -> list[Chunk]:

        chunks: list[Chunk] = []

        for document in documents:

            chunks.extend(
                self.chunk_document(document)
            )

        return chunks

    # ============================================================
    # EXTRACT SEMANTIC UNITS
    # ============================================================

    def _extract_units(
        self,
        page: Page,
    ) -> list[TextUnit]:

        raw_text = page.text.strip()

        if not raw_text:
            return []

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT destroy newlines before detecting structure.
        # Workflow/diagram extraction often depends on them.
        # --------------------------------------------------------

        blocks = self._split_blocks(raw_text)

        units: list[TextUnit] = []

        current_section: str | None = None
        current_subsection: str | None = None

        for block in blocks:

            block = self._clean_block(block)

            if not block:
                continue

            # ----------------------------------------------------
            # Heading
            # ----------------------------------------------------

            if self._looks_like_heading(block):

                level = self._heading_level(block)

                if level == 1:

                    current_section = self._normalize_heading(
                        block
                    )

                    current_subsection = None

                else:

                    current_subsection = self._normalize_heading(
                        block
                    )

                continue

            # ----------------------------------------------------
            # Detect workflow / diagram-like blocks FIRST.
            # ----------------------------------------------------

            if self._looks_like_workflow(block):

                units.append(
                    TextUnit(
                        text=self._normalize_workflow(block),
                        page_number=page.page_number,
                        section=current_section,
                        subsection=current_subsection,
                        unit_type="workflow",
                    )
                )

                continue

            # ----------------------------------------------------
            # Decision block
            # ----------------------------------------------------

            if self._looks_like_decision(block):

                units.append(
                    TextUnit(
                        text=self._normalize_structured_block(
                            block
                        ),
                        page_number=page.page_number,
                        section=current_section,
                        subsection=current_subsection,
                        unit_type="decision",
                    )
                )

                continue

            # ----------------------------------------------------
            # Table-like block
            # ----------------------------------------------------

            if self._looks_like_table(block):

                units.append(
                    TextUnit(
                        text=self._normalize_table(block),
                        page_number=page.page_number,
                        section=current_section,
                        subsection=current_subsection,
                        unit_type="table",
                    )
                )

                continue

            # ----------------------------------------------------
            # List / numbered steps
            # ----------------------------------------------------

            if self._looks_like_list(block):

                list_units = self._extract_list_units(
                    block=block,
                    page_number=page.page_number,
                    section=current_section,
                    subsection=current_subsection,
                )

                units.extend(list_units)

                continue

            # ----------------------------------------------------
            # Normal prose.
            #
            # Split into sentences while preserving semantic
            # boundaries.
            # ----------------------------------------------------

            sentences = self._split_sentences(block)

            for sentence in sentences:

                sentence = self._clean_inline_text(
                    sentence
                )

                if not sentence:
                    continue

                units.append(
                    TextUnit(
                        text=sentence,
                        page_number=page.page_number,
                        section=current_section,
                        subsection=current_subsection,
                        unit_type="text",
                    )
                )

        return units

    # ============================================================
    # BLOCK SPLITTING
    # ============================================================

    @staticmethod
    def _split_blocks(
        text: str,
    ) -> list[str]:

        # Normalize Windows line endings first.
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Preserve single newlines for diagrams/lists.
        blocks = re.split(
            r"\n\s*\n+",
            text,
        )

        return [
            block.strip()
            for block in blocks
            if block.strip()
        ]

    # ============================================================
    # BUILD CHUNKS
    # ============================================================

    def _build_chunks(
        self,
        units: list[TextUnit],
    ) -> list[
        tuple[
            str,
            int,
            str | None,
            str | None,
            str,
        ]
    ]:

        chunks = []

        current_units: list[TextUnit] = []

        for unit in units:

            unit_tokens = self._token_count(
                unit.text
            )

            # ----------------------------------------------------
            # Very large semantic unit.
            # ----------------------------------------------------

            if unit_tokens > self.MAX_TOKENS:

                if current_units:

                    chunks.append(
                        self._finalize_chunk(
                            current_units
                        )
                    )

                    current_units = []

                long_chunks = self._split_long_unit(
                    unit
                )

                chunks.extend(long_chunks)

                continue

            # ----------------------------------------------------
            # IMPORTANT:
            #
            # Keep workflow/decision/table units together when
            # possible.
            # ----------------------------------------------------

            candidate = current_units + [unit]

            candidate_text = self._join_units(
                candidate
            )

            candidate_tokens = self._token_count(
                candidate_text
            )

            # ----------------------------------------------------
            # Candidate exceeds maximum.
            # ----------------------------------------------------

            if (
                current_units
                and candidate_tokens > self.MAX_TOKENS
            ):

                chunks.append(
                    self._finalize_chunk(
                        current_units
                    )
                )

                # Semantic overlap.
                current_units = self._create_overlap(
                    current_units
                )

                candidate = current_units + [unit]

                candidate_text = self._join_units(
                    candidate
                )

                candidate_tokens = self._token_count(
                    candidate_text
                )

                # ------------------------------------------------
                # Remove overlap units if still too large.
                # ------------------------------------------------

                while (
                    current_units
                    and candidate_tokens > self.MAX_TOKENS
                ):

                    current_units.pop(0)

                    candidate = (
                        current_units + [unit]
                    )

                    candidate_text = (
                        self._join_units(candidate)
                    )

                    candidate_tokens = (
                        self._token_count(
                            candidate_text
                        )
                    )

            current_units.append(unit)

            current_tokens = self._token_count(
                self._join_units(current_units)
            )

            # ----------------------------------------------------
            # Target reached.
            # ----------------------------------------------------

            if current_tokens >= self.TARGET_TOKENS:

                chunks.append(
                    self._finalize_chunk(
                        current_units
                    )
                )

                current_units = self._create_overlap(
                    current_units
                )

        # --------------------------------------------------------
        # Remaining units.
        # --------------------------------------------------------

        if current_units:

            chunks.append(
                self._finalize_chunk(
                    current_units
                )
            )

        return chunks

    # ============================================================
    # SMALL CHUNK MERGING
    # ============================================================

    def _merge_small_chunks(
        self,
        chunks: list[
            tuple[
                str,
                int,
                str | None,
                str | None,
                str,
            ]
        ],
    ) -> list[
        tuple[
            str,
            int,
            str | None,
            str | None,
            str,
        ]
    ]:

        if not chunks:
            return []

        result = list(chunks)

        changed = True

        while changed:

            changed = False

            for index in range(len(result)):

                (
                    text,
                    page,
                    section,
                    subsection,
                    chunk_type,
                ) = result[index]

                token_count = self._token_count(
                    text
                )

                # Already valid.
                if token_count >= self.MIN_TOKENS:
                    continue

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Never blindly destroy a workflow chunk's
                # structure just to satisfy MIN_TOKENS.
                # ------------------------------------------------

                # ------------------------------------------------
                # Previous chunk.
                # ------------------------------------------------

                if index > 0:

                    previous = result[index - 1]

                    merged_text = self._merge_chunk_text(
                        previous[0],
                        text,
                        previous[4],
                        chunk_type,
                    )

                    merged_tokens = self._token_count(
                        merged_text
                    )

                    if merged_tokens <= self.MAX_TOKENS:

                        result[index - 1] = (
                            merged_text,
                            previous[1],
                            previous[2],
                            previous[3],
                            self._combined_chunk_type(
                                previous[4],
                                chunk_type,
                            ),
                        )

                        result.pop(index)

                        changed = True

                        break

                # ------------------------------------------------
                # Next chunk.
                # ------------------------------------------------

                if index < len(result) - 1:

                    next_chunk = result[index + 1]

                    merged_text = self._merge_chunk_text(
                        text,
                        next_chunk[0],
                        chunk_type,
                        next_chunk[4],
                    )

                    merged_tokens = self._token_count(
                        merged_text
                    )

                    if merged_tokens <= self.MAX_TOKENS:

                        result[index] = (
                            merged_text,
                            page,
                            section or next_chunk[2],
                            subsection or next_chunk[3],
                            self._combined_chunk_type(
                                chunk_type,
                                next_chunk[4],
                            ),
                        )

                        result.pop(index + 1)

                        changed = True

                        break

        # --------------------------------------------------------
        # Broader repair.
        # --------------------------------------------------------

        result = self._repair_remaining_small_chunks(
            result
        )

        return result

    # ============================================================
    # SECOND SMALL CHUNK REPAIR
    # ============================================================

    def _repair_remaining_small_chunks(
        self,
        chunks: list[
            tuple[
                str,
                int,
                str | None,
                str | None,
                str,
            ]
        ],
    ) -> list[
        tuple[
            str,
            int,
            str | None,
            str | None,
            str,
        ]
    ]:

        result = list(chunks)

        index = 0

        while index < len(result):

            (
                text,
                page,
                section,
                subsection,
                chunk_type,
            ) = result[index]

            token_count = self._token_count(
                text
            )

            if token_count >= self.MIN_TOKENS:

                index += 1

                continue

            best_index = None
            best_size = None

            # Search nearby chunks.
            for distance in range(
                1,
                min(4, len(result)),
            ):

                # Previous.
                previous_index = (
                    index - distance
                )

                if previous_index >= 0:

                    previous = result[
                        previous_index
                    ]

                    merged_text = (
                        self._merge_chunk_text(
                            previous[0],
                            text,
                            previous[4],
                            chunk_type,
                        )
                    )

                    merged_tokens = (
                        self._token_count(
                            merged_text
                        )
                    )

                    if (
                        merged_tokens
                        <= self.MAX_TOKENS
                    ):

                        if (
                            best_size is None
                            or merged_tokens < best_size
                        ):

                            best_index = (
                                previous_index
                            )

                            best_size = (
                                merged_tokens
                            )

                # Next.
                next_index = (
                    index + distance
                )

                if next_index < len(result):

                    next_chunk = result[
                        next_index
                    ]

                    merged_text = (
                        self._merge_chunk_text(
                            text,
                            next_chunk[0],
                            chunk_type,
                            next_chunk[4],
                        )
                    )

                    merged_tokens = (
                        self._token_count(
                            merged_text
                        )
                    )

                    if (
                        merged_tokens
                        <= self.MAX_TOKENS
                    ):

                        if (
                            best_size is None
                            or merged_tokens < best_size
                        ):

                            best_index = (
                                next_index
                            )

                            best_size = (
                                merged_tokens
                            )

            # Merge.
            if best_index is not None:

                target = result[best_index]

                if best_index < index:

                    merged_text = (
                        self._merge_chunk_text(
                            target[0],
                            text,
                            target[4],
                            chunk_type,
                        )
                    )

                    result[best_index] = (
                        merged_text,
                        target[1],
                        target[2],
                        target[3],
                        self._combined_chunk_type(
                            target[4],
                            chunk_type,
                        ),
                    )

                    result.pop(index)

                else:

                    merged_text = (
                        self._merge_chunk_text(
                            text,
                            target[0],
                            chunk_type,
                            target[4],
                        )
                    )

                    result[index] = (
                        merged_text,
                        page,
                        section or target[2],
                        subsection or target[3],
                        self._combined_chunk_type(
                            chunk_type,
                            target[4],
                        ),
                    )

                    result.pop(best_index)

                index = 0

                continue

            index += 1

        return result

    # ============================================================
    # SPLIT VERY LONG UNITS
    # ============================================================

    def _split_long_unit(
        self,
        unit: TextUnit,
    ):

        tokens = self.tokenizer.encode(
            unit.text,
            add_special_tokens=False,
        )

        chunks = []

        start = 0

        while start < len(tokens):

            end = min(
                start + self.MAX_TOKENS,
                len(tokens),
            )

            token_slice = tokens[
                start:end
            ]

            text = self.tokenizer.decode(
                token_slice,
                skip_special_tokens=True,
            ).strip()

            if text:

                chunks.append(
                    (
                        text,
                        unit.page_number,
                        unit.section,
                        unit.subsection,
                        unit.unit_type,
                    )
                )

            if end >= len(tokens):
                break

            start = max(
                end - self.OVERLAP_TOKENS,
                start + 1,
            )

        return chunks

    # ============================================================
    # OVERLAP
    # ============================================================

    def _create_overlap(
        self,
        units: list[TextUnit],
    ) -> list[TextUnit]:

        if not units:
            return []

        overlap: list[TextUnit] = []

        token_count = 0

        for unit in reversed(units):

            unit_tokens = self._token_count(
                unit.text
            )

            if (
                token_count + unit_tokens
                > self.OVERLAP_TOKENS
            ):
                break

            overlap.insert(
                0,
                unit,
            )

            token_count += unit_tokens

        return overlap

    # ============================================================
    # FINALIZE CHUNK
    # ============================================================

    def _finalize_chunk(
        self,
        units: list[TextUnit],
    ):

        text = self._join_units(units)

        page_number = units[0].page_number

        section = units[0].section

        subsection = units[0].subsection

        chunk_type = self._infer_chunk_type(
            units
        )

        return (
            text,
            page_number,
            section,
            subsection,
            chunk_type,
        )

    # ============================================================
    # TOKEN COUNT
    # ============================================================

    def _token_count(
        self,
        text: str,
    ) -> int:

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=True,
            )
        )

    # ============================================================
    # JOIN UNITS
    # ============================================================

    @staticmethod
    def _join_units(
        units: list[TextUnit],
    ) -> str:

        if not units:
            return ""

        parts: list[str] = []

        for unit in units:

            text = unit.text.strip()

            if not text:
                continue

            # Preserve line-oriented structure for:
            # workflows, decisions, lists and tables.
            if unit.unit_type in {
                "workflow",
                "decision",
                "list",
                "table",
                "diagram",
            }:
                parts.append(text)

            else:
                parts.append(text)

        # If structured content exists, preserve line breaks.
        has_structured = any(
            unit.unit_type in {
                "workflow",
                "decision",
                "list",
                "table",
                "diagram",
            }
            for unit in units
        )

        if has_structured:

            return "\n".join(parts).strip()

        return " ".join(parts).strip()

    # ============================================================
    # TEXT CLEANING
    # ============================================================

    @staticmethod
    def _clean_block(
        text: str,
    ) -> str:

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # Remove excessive spaces but preserve newlines.
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # Prevent excessive blank lines.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    @staticmethod
    def _clean_inline_text(
        text: str,
    ) -> str:

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ============================================================
    # SENTENCE SPLITTING
    # ============================================================

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:

        # Preserve explicit line boundaries when they represent
        # independent sentences.
        text = re.sub(
            r"\n+",
            " ",
            text,
        )

        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                text,
            )
            if sentence.strip()
        ]

    # ============================================================
    # HEADING DETECTION
    # ============================================================

    @classmethod
    def _looks_like_heading(
        cls,
        text: str,
    ) -> bool:

        text = text.strip()

        words = text.split()

        if not words:
            return False

        if len(words) > cls.MAX_HEADING_WORDS:
            return False

        if text.endswith(
            (".", "?", "!")
        ):
            return False

        uppercase_words = sum(
            1
            for word in words
            if word.isupper()
        )

        # Numbered headings.
        numbered = bool(
            re.match(
                r"^\d+(?:\.\d+)*[\.\)]?\s+",
                text,
            )
        )

        return (
            uppercase_words >= 1
            or text.istitle()
            or numbered
        )

    @staticmethod
    def _heading_level(
        text: str,
    ) -> int:

        match = re.match(
            r"^(\d+(?:\.\d+)*)",
            text,
        )

        if match:

            number = match.group(1)

            if "." in number:
                return 2

            return 1

        words = text.split()

        if len(words) <= 6:
            return 1

        return 2

    @staticmethod
    def _normalize_heading(
        text: str,
    ) -> str:

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ============================================================
    # WORKFLOW DETECTION
    # ============================================================

    @classmethod
    def _looks_like_workflow(
        cls,
        text: str,
    ) -> bool:

        lowered = text.lower()

        # Strong workflow language.
        if any(
            keyword in lowered
            for keyword in cls.WORKFLOW_KEYWORDS
        ):
            return True

        # Multiple arrows strongly suggest a diagram/workflow.
        arrows = len(
            cls.ARROW_PATTERN.findall(text)
        )

        if arrows >= 2:
            return True

        # Multiple numbered steps.
        lines = text.splitlines()

        step_lines = sum(
            1
            for line in lines
            if cls.STEP_PATTERN.match(
                line.strip()
            )
        )

        if step_lines >= 2:
            return True

        # Diagram-style vertical flow.
        if (
            "↓" in text
            and len(lines) >= 3
        ):
            return True

        return False

    # ============================================================
    # DECISION DETECTION
    # ============================================================

    @classmethod
    def _looks_like_decision(
        cls,
        text: str,
    ) -> bool:

        lowered = text.lower()

        matches = 0

        for pattern in cls.DECISION_PATTERNS:

            if re.search(
                pattern,
                lowered,
            ):
                matches += 1

        # Explicit decision blocks.
        if "decision:" in lowered:
            return True

        if "condition:" in lowered:
            return True

        return matches >= 2

    # ============================================================
    # LIST DETECTION
    # ============================================================

    @classmethod
    def _looks_like_list(
        cls,
        text: str,
    ) -> bool:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if len(lines) < 2:
            return False

        matches = sum(
            1
            for line in lines
            if cls.STEP_PATTERN.match(line)
        )

        return matches >= 2

    # ============================================================
    # LIST EXTRACTION
    # ============================================================

    def _extract_list_units(
        self,
        block: str,
        page_number: int,
        section: str | None,
        subsection: str | None,
    ) -> list[TextUnit]:

        lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip()
        ]

        units: list[TextUnit] = []

        for line in lines:

            # Remove only the list marker.
            cleaned = re.sub(
                r"""
                ^
                (?:
                    \d+[\.\)] |
                    \d+\.\d+[\.\)]? |
                    [A-Za-z][\.\)] |
                    [-*•▪◦] |
                    step\s+\d+ |
                    stage\s+\d+
                )
                \s+
                """,
                "",
                line,
                flags=re.IGNORECASE | re.VERBOSE,
            ).strip()

            if not cleaned:
                continue

            unit_type = "list"

            if self._looks_like_decision(
                cleaned
            ):
                unit_type = "decision"

            units.append(
                TextUnit(
                    text=cleaned,
                    page_number=page_number,
                    section=section,
                    subsection=subsection,
                    unit_type=unit_type,
                )
            )

        return units

    # ============================================================
    # TABLE DETECTION
    # ============================================================

    @staticmethod
    def _looks_like_table(
        text: str,
    ) -> bool:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if len(lines) < 2:
            return False

        pipe_lines = sum(
            1
            for line in lines
            if "|" in line
        )

        tab_like_lines = sum(
            1
            for line in lines
            if "\t" in line
        )

        # Markdown/table extraction.
        if pipe_lines >= 2:
            return True

        if tab_like_lines >= 2:
            return True

        # Repeated multi-column whitespace.
        column_lines = sum(
            1
            for line in lines
            if len(
                re.split(
                    r"\s{3,}",
                    line,
                )
            ) >= 2
        )

        return column_lines >= 2

    # ============================================================
    # TABLE NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_table(
        text: str,
    ) -> str:

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            # Preserve pipes because they communicate columns.
            line = re.sub(
                r"[ \t]+",
                " ",
                line,
            )

            lines.append(line)

        return "\n".join(lines)

    # ============================================================
    # WORKFLOW NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_workflow(
        text: str,
    ) -> str:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        normalized: list[str] = []

        for line in lines:

            line = re.sub(
                r"[ \t]+",
                " ",
                line,
            ).strip()

            if not line:
                continue

            normalized.append(line)

        return "\n".join(normalized)

    # ============================================================
    # STRUCTURED BLOCK NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_structured_block(
        text: str,
    ) -> str:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return "\n".join(
            re.sub(
                r"[ \t]+",
                " ",
                line,
            )
            for line in lines
        )

    # ============================================================
    # CHUNK TYPE
    # ============================================================

    @staticmethod
    def _infer_chunk_type(
        units: list[TextUnit],
    ) -> str:

        if not units:
            return "text"

        types = {
            unit.unit_type
            for unit in units
        }

        # Prioritize special semantic structures.
        if "workflow" in types:
            return "workflow"

        if "decision" in types:
            return "decision"

        if "table" in types:
            return "table"

        if "diagram" in types:
            return "diagram"

        if "list" in types:
            return "list"

        return "text"

    # ============================================================
    # COMBINED CHUNK TYPE
    # ============================================================

    @staticmethod
    def _combined_chunk_type(
        first: str,
        second: str,
    ) -> str:

        priority = {
            "workflow": 6,
            "decision": 5,
            "diagram": 4,
            "table": 3,
            "list": 2,
            "text": 1,
        }

        return (
            first
            if priority.get(first, 1)
            >= priority.get(second, 1)
            else second
        )

    # ============================================================
    # MERGE CHUNK TEXT
    # ============================================================

    @staticmethod
    def _merge_chunk_text(
        first: str,
        second: str,
        first_type: str,
        second_type: str,
    ) -> str:

        structured = {
            "workflow",
            "decision",
            "diagram",
            "table",
            "list",
        }

        if (
            first_type in structured
            or second_type in structured
        ):
            return (
                first.rstrip()
                + "\n"
                + second.lstrip()
            ).strip()

        return (
            first.rstrip()
            + " "
            + second.lstrip()
        ).strip()