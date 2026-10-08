import re
from collections import Counter
from math import ceil

from app.models.document import Document, Page


class DocumentStructureProcessor:
    """
    Performs document-level structural normalization.

    Responsibilities:
    - Detect repeated headers using page geometry.
    - Detect page-number footers.
    - Reconstruct readable rows from positional PDF words.
    - Remove repeated headers and footers from page text.
    - Detect likely page-to-page continuation.
    - Preserve layout information for downstream processing.

    This class does NOT classify regions as:
    - table
    - workflow
    - diagram
    - image
    - normal text

    Region classification is handled later by the Region Detector.
    """

    def __init__(
        self,
        header_zone_ratio: float = 0.15,
        footer_zone_ratio: float = 0.10,
        header_min_occurrences: int = 3,
        header_min_page_ratio: float = 0.20,
        row_tolerance: float = 3.0,
    ):
        self.header_zone_ratio = header_zone_ratio
        self.footer_zone_ratio = footer_zone_ratio

        self.header_min_occurrences = (
            header_min_occurrences
        )

        self.header_min_page_ratio = (
            header_min_page_ratio
        )

        self.row_tolerance = row_tolerance

    # ============================================================
    # PUBLIC API
    # ============================================================

    def process(
        self,
        document: Document,
    ) -> Document:
        """
        Process the complete document.

        The complete document is processed together because
        repeated headers and page continuation require
        document-level context.
        """

        if not document.pages:
            return document

        # --------------------------------------------------------
        # 1. Detect repeated headers across the document.
        # --------------------------------------------------------

        repeated_headers = (
            self._find_repeated_headers(
                document
            )
        )

        # --------------------------------------------------------
        # 2. Process every page.
        # --------------------------------------------------------

        for page in document.pages:

            self._process_page(
                page=page,
                repeated_headers=repeated_headers,
            )

        # --------------------------------------------------------
        # 3. Detect page-to-page continuation.
        # --------------------------------------------------------

        self._detect_page_continuations(
            document
        )

        # --------------------------------------------------------
        # 4. Store document-level metadata.
        # --------------------------------------------------------

        document.metadata[
            "structure_processed"
        ] = True

        document.metadata[
            "repeated_headers"
        ] = sorted(
            repeated_headers
        )

        document.metadata[
            "continuation_detection_enabled"
        ] = True

        return document

    # ============================================================
    # PAGE PROCESSING
    # ============================================================

    def _process_page(
        self,
        page: Page,
        repeated_headers: set[str],
    ) -> None:
        """
        Process one page.

        The original extracted text is preserved in:

            page.raw_text

        The cleaned/reconstructed text is stored in:

            page.text
        """

        # --------------------------------------------------------
        # Preserve original extracted text.
        # --------------------------------------------------------

        if not page.raw_text:
            page.raw_text = page.text

        # --------------------------------------------------------
        # Detect repeated header.
        # --------------------------------------------------------

        header = self._detect_page_header(
            page=page,
            repeated_headers=repeated_headers,
        )

        # --------------------------------------------------------
        # Detect footer.
        # --------------------------------------------------------

        footer = self._detect_page_footer(
            page
        )

        page.header_text = header
        page.footer_text = footer

        # --------------------------------------------------------
        # Reconstruct readable text from positional words.
        #
        # This fixes PDFs where pypdf produces:
        #
        # The
        # user
        # arrive
        # early
        #
        # instead of:
        #
        # The user arrive early
        # --------------------------------------------------------

        reconstructed_text = (
            self._reconstruct_page_text(
                page
            )
        )

        # --------------------------------------------------------
        # Remove header/footer.
        # --------------------------------------------------------

        cleaned_text = (
            self._remove_structural_lines(
                text=reconstructed_text,
                header=header,
                footer=footer,
            )
        )

        page.text = cleaned_text

        # --------------------------------------------------------
        # Page metadata.
        # --------------------------------------------------------

        page.metadata[
            "header_removed"
        ] = bool(header)

        page.metadata[
            "footer_removed"
        ] = bool(footer)

        page.metadata[
            "cleaned_text_length"
        ] = len(page.text)

        page.metadata[
            "reconstructed_from_layout"
        ] = bool(
            page.layout_words
        )

    # ============================================================
    # TEXT RECONSTRUCTION
    # ============================================================

    def _reconstruct_page_text(
        self,
        page: Page,
    ) -> str:
        """
        Reconstruct readable text from positional words.

        Uses the PDF coordinates to group words that visually
        belong to the same row.

        If layout information is unavailable, the original
        extracted page text is returned.
        """

        if not page.layout_words:
            return page.text

        rows = self._build_layout_rows(
            page.layout_words
        )

        if not rows:
            return page.text

        lines: list[str] = []

        for row in rows:

            text = row["text"].strip()

            if not text:
                continue

            lines.append(text)

        if not lines:
            return page.text

        return "\n".join(lines)

    # ============================================================
    # HEADER DETECTION
    # ============================================================

    def _find_repeated_headers(
        self,
        document: Document,
    ) -> set[str]:
        """
        Find text rows that repeatedly occur near the top
        of multiple pages.
        """

        candidates: list[str] = []

        for page in document.pages:

            page_height = (
                self._get_page_height(
                    page
                )
            )

            if page_height <= 0:
                continue

            top_limit = (
                page_height
                * self.header_zone_ratio
            )

            rows = self._build_layout_rows(
                page.layout_words
            )

            for row in rows:

                if row["top"] > top_limit:
                    continue

                text = row["text"].strip()

                if not text:
                    continue

                if self._is_page_footer_text(
                    text
                ):
                    continue

                if not self._is_reasonable_header_candidate(
                    text
                ):
                    continue

                candidates.append(
                    self._normalize_line(
                        text
                    )
                )

        if not candidates:
            return set()

        counter = Counter(
            candidates
        )

        page_count = len(
            document.pages
        )

        ratio_threshold = ceil(
            page_count
            * self.header_min_page_ratio
        )

        minimum_occurrences = max(
            self.header_min_occurrences,
            ratio_threshold,
        )

        repeated_headers = {
            text
            for text, count
            in counter.items()
            if count >= minimum_occurrences
        }

        return repeated_headers

    def _detect_page_header(
        self,
        page: Page,
        repeated_headers: set[str],
    ) -> str | None:
        """
        Detect a repeated header on a specific page.
        """

        if not repeated_headers:
            return None

        page_height = (
            self._get_page_height(
                page
            )
        )

        if page_height > 0:

            top_limit = (
                page_height
                * self.header_zone_ratio
            )

            rows = self._build_layout_rows(
                page.layout_words
            )

            for row in rows:

                if row["top"] > top_limit:
                    continue

                text = row["text"].strip()

                normalized = (
                    self._normalize_line(
                        text
                    )
                )

                if normalized in repeated_headers:
                    return text

        # --------------------------------------------------------
        # Fallback when layout words are unavailable.
        # --------------------------------------------------------

        lines = self._meaningful_lines(
            page.text
        )

        for line in lines[:3]:

            normalized = (
                self._normalize_line(
                    line
                )
            )

            if normalized in repeated_headers:
                return line

        return None

    # ============================================================
    # FOOTER DETECTION
    # ============================================================

    def _detect_page_footer(
        self,
        page: Page,
    ) -> str | None:
        """
        Detect page-number footer.
        """

        page_height = (
            self._get_page_height(
                page
            )
        )

        if page_height > 0:

            bottom_limit = (
                page_height
                * (
                    1.0
                    - self.footer_zone_ratio
                )
            )

            rows = self._build_layout_rows(
                page.layout_words
            )

            for row in reversed(rows):

                if row["top"] < bottom_limit:
                    continue

                text = row["text"].strip()

                if self._is_page_footer_text(
                    text
                ):
                    return text

        # --------------------------------------------------------
        # Fallback using extracted text.
        # --------------------------------------------------------

        lines = self._meaningful_lines(
            page.text
        )

        for line in reversed(
            lines[-3:]
        ):

            if self._is_page_footer_text(
                line
            ):
                return line

        return None

    # ============================================================
    # STRUCTURAL CLEANUP
    # ============================================================

    def _remove_structural_lines(
        self,
        text: str,
        header: str | None,
        footer: str | None,
    ) -> str:
        """
        Remove detected header/footer from reconstructed text.

        layout_words remain untouched.
        """

        if not text:
            return ""

        header_normalized = (
            self._normalize_line(
                header
            )
            if header
            else None
        )

        footer_normalized = (
            self._normalize_line(
                footer
            )
            if footer
            else None
        )

        cleaned_lines: list[str] = []

        for line in text.splitlines():

            stripped = line.strip()

            if not stripped:
                continue

            normalized = (
                self._normalize_line(
                    stripped
                )
            )

            # Remove repeated header.
            if (
                header_normalized
                and normalized
                == header_normalized
            ):
                continue

            # Remove footer.
            if (
                footer_normalized
                and normalized
                == footer_normalized
            ):
                continue

            # Remove page-number footer patterns.
            if self._is_page_footer_text(
                stripped
            ):
                continue

            cleaned_lines.append(
                stripped
            )

        return "\n".join(
            cleaned_lines
        ).strip()

    # ============================================================
    # PAGE CONTINUATION
    # ============================================================

    def _detect_page_continuations(
        self,
        document: Document,
    ) -> None:
        """
        Detect likely page-to-page continuation.

        This does NOT merge pages.

        It only records evidence for later processing.
        """

        pages = document.pages

        for index in range(
            1,
            len(pages),
        ):

            previous = pages[
                index - 1
            ]

            current = pages[
                index
            ]

            result = (
                self._pages_are_likely_continuation(
                    previous=previous,
                    current=current,
                )
            )

            previous.continues_next = (
                bool(
                    result["continues"]
                )
            )

            current.continues_previous = (
                bool(
                    result["continues"]
                )
            )

            current.metadata[
                "continuation_confidence"
            ] = result[
                "confidence"
            ]

            current.metadata[
                "continuation_reason"
            ] = result[
                "reason"
            ]

    def _pages_are_likely_continuation(
        self,
        previous: Page,
        current: Page,
    ) -> dict[str, object]:
        """
        Determine whether current page likely continues
        previous page.

        This remains deliberately conservative.
        """

        previous_text = (
            previous.text.strip()
        )

        current_text = (
            current.text.strip()
        )

        if (
            not previous_text
            or not current_text
        ):
            return {
                "continues": False,
                "confidence": 0.0,
                "reason": "empty_page",
            }

        previous_lines = (
            self._meaningful_lines(
                previous_text
            )
        )

        current_lines = (
            self._meaningful_lines(
                current_text
            )
        )

        if (
            not previous_lines
            or not current_lines
        ):
            return {
                "continues": False,
                "confidence": 0.0,
                "reason": "no_meaningful_lines",
            }

        previous_last = (
            previous_lines[-1]
        )

        current_first = (
            current_lines[0]
        )

        score = 0.0

        reasons: list[str] = []

        # --------------------------------------------------------
        # Signal 1:
        # Previous page ends mid-sentence.
        # --------------------------------------------------------

        if self._looks_like_open_sentence(
            previous_last
        ):
            score += 0.25

            reasons.append(
                "previous_page_ends_open"
            )

        # --------------------------------------------------------
        # Signal 2:
        # Current page starts with bullet/list item.
        # --------------------------------------------------------

        if self._looks_like_continuation_item(
            current_first
        ):
            score += 0.35

            reasons.append(
                "current_page_starts_with_continuation_item"
            )

        # --------------------------------------------------------
        # Signal 3:
        # Current page starts with numbered item.
        # --------------------------------------------------------

        if self._looks_like_numbered_item(
            current_first
        ):
            score += 0.20

            reasons.append(
                "current_page_starts_with_numbered_item"
            )

        # --------------------------------------------------------
        # Signal 4:
        # Same major section.
        # --------------------------------------------------------

        previous_number = (
            self._extract_section_prefix(
                previous_last
            )
        )

        current_number = (
            self._extract_section_prefix(
                current_first
            )
        )

        if (
            previous_number
            and current_number
            and previous_number
            == current_number
        ):
            score += 0.35

            reasons.append(
                "same_section_prefix"
            )

        # --------------------------------------------------------
        # Signal 5:
        # Structured content starts on current page.
        # --------------------------------------------------------

        if self._looks_like_structured_continuation(
            current_first
        ):
            score += 0.20

            reasons.append(
                "structured_content_continuation"
            )

        # --------------------------------------------------------
        # Final decision.
        #
        # IMPORTANT:
        # A weak 0.25 signal is NOT enough.
        # --------------------------------------------------------

        confidence = min(
            score,
            1.0,
        )

        continues = (
            confidence >= 0.50
        )

        if reasons:

            reason = ", ".join(
                reasons
            )

        else:

            reason = (
                "no_continuation_signal"
            )

        return {
            "continues": continues,
            "confidence": confidence,
            "reason": reason,
        }

    # ============================================================
    # LAYOUT ROW RECONSTRUCTION
    # ============================================================

    def _build_layout_rows(
        self,
        words: list[dict],
    ) -> list[dict]:
        """
        Group positional PDF words into visual rows.

        Example:

            Word A  x=70  y=100
            Word B  x=120 y=101
            Word C  x=180 y=100

        becomes:

            "Word A Word B Word C"

        Coordinates are preserved for later region detection.
        """

        if not words:
            return []

        normalized_words: list[
            dict
        ] = []

        for word in words:

            text = str(
                word.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            normalized_words.append(
                {
                    "text": text,

                    "x0": float(
                        word.get(
                            "x0",
                            0.0,
                        )
                    ),

                    "x1": float(
                        word.get(
                            "x1",
                            0.0,
                        )
                    ),

                    "top": float(
                        word.get(
                            "top",
                            0.0,
                        )
                    ),

                    "bottom": float(
                        word.get(
                            "bottom",
                            0.0,
                        )
                    ),
                }
            )

        normalized_words.sort(
            key=lambda word: (
                word["top"],
                word["x0"],
            )
        )

        rows: list[
            list[dict]
        ] = []

        for word in normalized_words:

            if not rows:

                rows.append(
                    [word]
                )

                continue

            current_row = rows[-1]

            reference_top = (
                current_row[0]["top"]
            )

            if abs(
                word["top"]
                - reference_top
            ) <= self.row_tolerance:

                current_row.append(
                    word
                )

            else:

                rows.append(
                    [word]
                )

        result: list[
            dict
        ] = []

        for index, row in enumerate(
            rows
        ):

            row.sort(
                key=lambda word: (
                    word["x0"]
                )
            )

            result.append(
                {
                    "row_index": index,

                    "top": min(
                        word["top"]
                        for word in row
                    ),

                    "bottom": max(
                        word["bottom"]
                        for word in row
                    ),

                    "x_start": min(
                        word["x0"]
                        for word in row
                    ),

                    "x_end": max(
                        word["x1"]
                        for word in row
                    ),

                    "text": " ".join(
                        word["text"]
                        for word in row
                    ),
                }
            )

        return result

    # ============================================================
    # GENERAL HELPERS
    # ============================================================

    @staticmethod
    def _get_page_height(
        page: Page,
    ) -> float:
        """
        Read PDF page height stored by DocumentLoader.
        """

        try:

            return float(
                page.metadata.get(
                    "page_height",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    @staticmethod
    def _meaningful_lines(
        text: str,
    ) -> list[str]:
        """
        Return non-empty lines.
        """

        return [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    @staticmethod
    def _normalize_line(
        value: str,
    ) -> str:
        """
        Normalize whitespace and case for comparisons.
        """

        value = value.strip()

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.casefold()

    @staticmethod
    def _is_page_footer_text(
        text: str,
    ) -> bool:
        """
        Recognize common PDF page footer formats.
        """

        value = text.strip()

        patterns = [
            r"^\d+\s*\|\s*Page$",
            r"^Page\s+\d+$",
            r"^Page\s+\d+\s+of\s+\d+$",
            r"^-\s*\d+\s*-$",
            r"^\d+\s*/\s*\d+$",
        ]

        return any(
            re.match(
                pattern,
                value,
                re.IGNORECASE,
            )
            for pattern in patterns
        )

    @staticmethod
    def _is_reasonable_header_candidate(
        text: str,
    ) -> bool:
        """
        Filter obvious body content from header candidates.
        """

        if len(text) < 5:
            return False

        if len(text) > 180:
            return False

        if DocumentStructureProcessor._is_page_footer_text(
            text
        ):
            return False

        # Numbered subsection such as:
        #
        # 8.1 A user places an order
        #
        # should not normally be treated as a header.
        if re.match(
            r"^\d+(?:\.\d+)+\s+",
            text,
        ):
            return False

        return True

    @staticmethod
    def _looks_like_open_sentence(
        text: str,
    ) -> bool:
        """
        Determine whether a line appears to continue.
        """

        text = text.strip()

        if not text:
            return False

        if text.endswith(
            (
                ":",
                ";",
                ",",
                "and",
                "or",
                "to",
            )
        ):
            return True

        return not text.endswith(
            (
                ".",
                "?",
                "!",
                ":",
                ";",
            )
        )

    @staticmethod
    def _looks_like_continuation_item(
        text: str,
    ) -> bool:
        """
        Detect bullet/list continuation.
        """

        return bool(
            re.match(
                r"^(?:•|▪|●|○|-|–|\*)\s+",
                text,
            )
        )

    @staticmethod
    def _looks_like_numbered_item(
        text: str,
    ) -> bool:
        """
        Detect numbered structures.

        Examples:

            8.1
            8.2
            12.5
            3.1.2
        """

        return bool(
            re.match(
                r"^\d+(?:\.\d+)+\s+",
                text,
            )
        )

    @staticmethod
    def _extract_section_prefix(
        text: str,
    ) -> str | None:
        """
        Extract major section number.

        Example:

            8.5 A user arrives
            -> 8

            12.4 Driver performance
            -> 12
        """

        match = re.match(
            r"^(\d+)(?:\.\d+)+\s+",
            text.strip(),
        )

        if not match:
            return None

        return match.group(1)

    @staticmethod
    def _looks_like_structured_continuation(
        text: str,
    ) -> bool:
        """
        Detect common structured-content starts.
        """

        if DocumentStructureProcessor._looks_like_numbered_item(
            text
        ):
            return True

        if DocumentStructureProcessor._looks_like_continuation_item(
            text
        ):
            return True

        lower = text.casefold()

        structured_prefixes = (
            "code",
            "name",
            "description",
            "stakeholders",
            "general note",
            "quality parameter",
            "place of execution",
            "probability",
            "impact",
        )

        return lower.startswith(
            structured_prefixes
        )