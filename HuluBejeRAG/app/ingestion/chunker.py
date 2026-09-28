from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import AutoTokenizer

from app.models.chunk import Chunk
from app.models.document import Document, Page


@dataclass
class TextUnit:
    """
    Smallest meaningful unit used during chunk construction.
    """

    text: str
    page_number: int
    section: str | None = None
    subsection: str | None = None


class Chunker:
    """
    Smart tokenizer-aware chunker designed for BGE-large.

    BGE-large maximum:
        512 tokens

    Our chunk policy:

        Minimum : 140 tokens
        Target  : 420 tokens
        Maximum : 450 tokens
        Overlap : 60 tokens
    """

    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    MIN_TOKENS = 130
    TARGET_TOKENS = 420
    MAX_TOKENS = 450
    OVERLAP_TOKENS = 60

    def __init__(self) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def chunk_document(
        self,
        document: Document,
    ) -> list[Chunk]:

        raw_chunks: list[
            tuple[str, int, str | None, str | None]
        ] = []

        # -----------------------------------------------------
        # Build chunks page by page.
        # -----------------------------------------------------

        for page in document.pages:

            units = self._extract_units(page)

            page_chunks = self._build_chunks(units)

            raw_chunks.extend(page_chunks)

        # -----------------------------------------------------
        # Merge small chunks.
        # -----------------------------------------------------

        raw_chunks = self._merge_small_chunks(
            raw_chunks
        )

        # -----------------------------------------------------
        # Build final Chunk objects.
        # -----------------------------------------------------

        chunks: list[Chunk] = []

        for index, (
            text,
            page_number,
            section,
            subsection,
        ) in enumerate(raw_chunks):

            token_count = self._token_count(text)

            metadata = dict(document.metadata)

            metadata.update(
                {
                    "document_id": document.document_id,
                    "source": document.source,
                    "page": str(page_number),
                    "chunk_index": str(index),
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

    # =========================================================
    # EXTRACT SEMANTIC UNITS
    # =========================================================

    def _extract_units(
        self,
        page: Page,
    ) -> list[TextUnit]:

        blocks = re.split(
            r"\n\s*\n+",
            page.text.strip(),
        )

        units: list[TextUnit] = []

        current_section: str | None = None
        current_subsection: str | None = None

        for block in blocks:

            block = self._clean_text(block)

            if not block:
                continue

            # -------------------------------------------------
            # Detect heading.
            # -------------------------------------------------

            if self._looks_like_heading(block):

                level = self._heading_level(block)

                if level == 1:

                    current_section = block
                    current_subsection = None

                else:

                    current_subsection = block

                continue

            # -------------------------------------------------
            # Split normal content into sentences.
            # -------------------------------------------------

            sentences = self._split_sentences(block)

            for sentence in sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                units.append(
                    TextUnit(
                        text=sentence,
                        page_number=page.page_number,
                        section=current_section,
                        subsection=current_subsection,
                    )
                )

        return units

    # =========================================================
    # BUILD NORMAL CHUNKS
    # =========================================================

    def _build_chunks(
        self,
        units: list[TextUnit],
    ) -> list[
        tuple[str, int, str | None, str | None]
    ]:

        chunks = []

        current_units: list[TextUnit] = []

        for unit in units:

            unit_tokens = self._token_count(
                unit.text
            )

            # -------------------------------------------------
            # Very large sentence / paragraph.
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Try adding the next sentence.
            # -------------------------------------------------

            candidate = current_units + [unit]

            candidate_text = self._join_units(
                candidate
            )

            candidate_tokens = self._token_count(
                candidate_text
            )

            # -------------------------------------------------
            # Candidate would exceed MAX.
            # -------------------------------------------------

            if (
                current_units
                and candidate_tokens > self.MAX_TOKENS
            ):

                chunks.append(
                    self._finalize_chunk(
                        current_units
                    )
                )

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

                # -------------------------------------------------
                # Remove overlap units if necessary.
                # -------------------------------------------------

                while (
                    current_units
                    and candidate_tokens > self.MAX_TOKENS
                ):

                    current_units.pop(0)

                    candidate = current_units + [unit]

                    candidate_text = self._join_units(
                        candidate
                    )

                    candidate_tokens = self._token_count(
                        candidate_text
                    )

            current_units.append(unit)

            current_tokens = self._token_count(
                self._join_units(current_units)
            )

            # -------------------------------------------------
            # Target reached.
            # -------------------------------------------------

            if current_tokens >= self.TARGET_TOKENS:

                chunks.append(
                    self._finalize_chunk(
                        current_units
                    )
                )

                current_units = self._create_overlap(
                    current_units
                )

        # -----------------------------------------------------
        # Remaining units.
        # -----------------------------------------------------

        if current_units:

            chunks.append(
                self._finalize_chunk(
                    current_units
                )
            )

        return chunks

    # =========================================================
    # SMALL CHUNK MERGING
    # =========================================================

    def _merge_small_chunks(
        self,
        chunks: list[
            tuple[str, int, str | None, str | None]
        ],
    ) -> list[
        tuple[str, int, str | None, str | None]
    ]:

        if not chunks:
            return []

        result = list(chunks)

        # -----------------------------------------------------
        # Keep attempting merges until no more changes
        # can be made.
        # -----------------------------------------------------

        changed = True

        while changed:

            changed = False

            for index in range(len(result)):

                text, page, section, subsection = (
                    result[index]
                )

                token_count = self._token_count(
                    text
                )

                # Already valid.
                if token_count >= self.MIN_TOKENS:
                    continue

                # =================================================
                # OPTION 1
                # Merge with previous chunk.
                # =================================================

                if index > 0:

                    previous = result[index - 1]

                    merged_text = (
                        previous[0]
                        + " "
                        + text
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
                        )

                        result.pop(index)

                        changed = True

                        break

                # =================================================
                # OPTION 2
                # Merge with next chunk.
                # =================================================

                if index < len(result) - 1:

                    next_chunk = result[index + 1]

                    merged_text = (
                        text
                        + " "
                        + next_chunk[0]
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
                        )

                        result.pop(index + 1)

                        changed = True

                        break

        # -----------------------------------------------------
        # Second pass:
        #
        # Sometimes a small chunk cannot merge with either
        # immediate neighbor because both would exceed 450.
        #
        # Try a broader local merge by moving the small chunk
        # into the closest compatible neighbor.
        # -----------------------------------------------------

        result = self._repair_remaining_small_chunks(
            result
        )

        return result

    # =========================================================
    # SECOND SMALL-CHUNK REPAIR
    # =========================================================

    def _repair_remaining_small_chunks(
        self,
        chunks: list[
            tuple[str, int, str | None, str | None]
        ],
    ) -> list[
        tuple[str, int, str | None, str | None]
    ]:

        result = list(chunks)

        index = 0

        while index < len(result):

            text, page, section, subsection = (
                result[index]
            )

            token_count = self._token_count(
                text
            )

            if token_count >= self.MIN_TOKENS:

                index += 1
                continue

            best_index = None
            best_size = None

            # -------------------------------------------------
            # Look at nearby chunks.
            # -------------------------------------------------

            for distance in range(
                1,
                min(4, len(result))
            ):

                # Previous candidate.
                previous_index = index - distance

                if previous_index >= 0:

                    previous = result[
                        previous_index
                    ]

                    merged_text = (
                        previous[0]
                        + " "
                        + text
                    )

                    merged_tokens = self._token_count(
                        merged_text
                    )

                    if (
                        merged_tokens <= self.MAX_TOKENS
                    ):

                        if (
                            best_size is None
                            or merged_tokens < best_size
                        ):

                            best_index = previous_index
                            best_size = merged_tokens

                # Next candidate.
                next_index = index + distance

                if next_index < len(result):

                    next_chunk = result[
                        next_index
                    ]

                    merged_text = (
                        text
                        + " "
                        + next_chunk[0]
                    )

                    merged_tokens = self._token_count(
                        merged_text
                    )

                    if (
                        merged_tokens <= self.MAX_TOKENS
                    ):

                        if (
                            best_size is None
                            or merged_tokens < best_size
                        ):

                            best_index = next_index
                            best_size = merged_tokens

            # -------------------------------------------------
            # Merge if possible.
            # -------------------------------------------------

            if best_index is not None:

                target = result[best_index]

                if best_index < index:

                    merged_text = (
                        target[0]
                        + " "
                        + text
                    )

                    result[best_index] = (
                        merged_text,
                        target[1],
                        target[2],
                        target[3],
                    )

                    result.pop(index)

                else:

                    merged_text = (
                        text
                        + " "
                        + target[0]
                    )

                    result[index] = (
                        merged_text,
                        page,
                        section or target[2],
                        subsection or target[3],
                    )

                    result.pop(best_index)

                index = 0
                continue

            # -------------------------------------------------
            # No valid merge possible.
            #
            # We leave it temporarily and continue.
            # The final validation will reveal it.
            # -------------------------------------------------

            index += 1

        return result

    # =========================================================
    # SPLIT VERY LONG UNITS
    # =========================================================

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

            token_slice = tokens[start:end]

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
                    )
                )

            if end >= len(tokens):
                break

            start = max(
                end - self.OVERLAP_TOKENS,
                start + 1,
            )

        return chunks

    # =========================================================
    # OVERLAP
    # =========================================================

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

            overlap.insert(0, unit)

            token_count += unit_tokens

        return overlap

    # =========================================================
    # FINALIZE
    # =========================================================

    def _finalize_chunk(
        self,
        units: list[TextUnit],
    ):

        text = self._join_units(units)

        page_number = units[0].page_number

        section = units[0].section

        subsection = units[0].subsection

        return (
            text,
            page_number,
            section,
            subsection,
        )

    # =========================================================
    # TOKEN COUNT
    # =========================================================

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

    # =========================================================
    # TEXT HELPERS
    # =========================================================

    @staticmethod
    def _join_units(
        units: list[TextUnit],
    ) -> str:

        return " ".join(
            unit.text
            for unit in units
        ).strip()

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:

        return re.split(
            r"(?<=[.!?])\s+",
            text,
        )

    @staticmethod
    def _looks_like_heading(
        text: str,
    ) -> bool:

        words = text.split()

        if not words:
            return False

        if len(words) > 12:
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

        return (
            uppercase_words >= 1
            or text.istitle()
        )

    @staticmethod
    def _heading_level(
        text: str,
    ) -> int:

        words = text.split()

        if len(words) <= 6:
            return 1

        return 2