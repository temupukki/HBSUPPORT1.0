import re
from dataclasses import dataclass
from typing import Any

from app.models.document import Document, Page
from app.models.region import BoundingBox, Region


@dataclass
class DetectorConfig:
    """
    Configuration for spatial region detection.
    """

    # Words whose vertical positions are within this
    # tolerance are considered part of the same physical row.
    row_tolerance: float = 3.0

    # Large vertical gaps can indicate a new logical region.
    vertical_gap_threshold: float = 14.0

    # Minimum words expected for a normal content region.
    minimum_words: int = 2

    # Percentage of page width used when deciding whether
    # two rows are likely in different columns.
    column_gap_ratio: float = 0.08

    # Percentage of page width considered "full width".
    full_width_ratio: float = 0.70

    # Classification thresholds.
    workflow_confidence_threshold: float = 0.60
    table_confidence_threshold: float = 0.60


class RegionDetector:
    """
    Detects logical/spatial regions inside PDF pages.

    The RegionDetector answers:

        "Where are the different pieces of content?"

    It does NOT perform deep extraction.

    Possible region types:

        paragraph
        heading
        list
        structured_section
        table_candidate
        workflow_definition
        visual
        visual_candidate
        unknown

    Important:

    The detector preserves PDF geometry.

    Later components can use this information for:

        tables
        workflows
        diagrams
        OCR
        visual analysis
        structure assembly
        chunking
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
    ):
        self.config = config or DetectorConfig()

    # =========================================================
    # PUBLIC API
    # =========================================================

    def detect_document(
        self,
        document: Document,
    ) -> list[Region]:
        """
        Detect regions across the entire document.
        """

        regions: list[Region] = []

        for page in document.pages:
            page_regions = self.detect_page(
                document=document,
                page=page,
            )

            regions.extend(page_regions)

        return regions

    def detect_page(
        self,
        document: Document,
        page: Page,
    ) -> list[Region]:
        """
        Detect logical/spatial regions on a single page.
        """

        # -----------------------------------------------------
        # No layout information.
        #
        # We cannot perform reliable spatial detection,
        # so fall back to one page-level region.
        # -----------------------------------------------------

        if not page.layout_words:
            return self._fallback_page_region(
                document=document,
                page=page,
            )

        # -----------------------------------------------------
        # Convert words into physical rows.
        # -----------------------------------------------------

        rows = self._build_rows(
            page.layout_words
        )

        if not rows:
            return self._fallback_page_region(
                document=document,
                page=page,
            )

        # -----------------------------------------------------
        # Remove headers and footers that were already detected
        # by DocumentStructureProcessor.
        # -----------------------------------------------------

        rows = self._remove_structural_rows(
            page=page,
            rows=rows,
        )

        if not rows:
            return []

        # -----------------------------------------------------
        # Split rows into spatial/logical groups.
        # -----------------------------------------------------

        spatial_groups = self._split_spatial_groups(
            rows=rows,
            page=page,
        )

        regions: list[Region] = []

        # -----------------------------------------------------
        # Classify every spatial group.
        # -----------------------------------------------------

        for index, group in enumerate(
            spatial_groups
        ):
            region = self._classify_group(
                document=document,
                page=page,
                rows=group,
                index=index,
            )

            if region is not None:
                regions.append(region)

        # -----------------------------------------------------
        # Detect image/scanned/visual regions.
        # -----------------------------------------------------

        regions.extend(
            self._detect_visual_regions(
                document=document,
                page=page,
                existing_regions=regions,
            )
        )

        # -----------------------------------------------------
        # Final ordering.
        # -----------------------------------------------------

        regions.sort(
            key=lambda region: (
                region.page_number,
                region.bbox.top,
                region.bbox.x0,
            )
        )

        return regions

    # =========================================================
    # ROW BUILDING
    # =========================================================

    def _build_rows(
        self,
        words: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert positional PDF words into physical rows.

        IMPORTANT:

        We preserve x0/x1 because region detection depends
        on horizontal geometry.

        We do NOT simply flatten everything into text.
        """

        normalized: list[dict[str, Any]] = []

        for word in words:
            text = str(
                word.get("text", "")
            ).strip()

            if not text:
                continue

            normalized.append(
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

        normalized.sort(
            key=lambda word: (
                word["top"],
                word["x0"],
            )
        )

        rows: list[list[dict[str, Any]]] = []

        for word in normalized:

            if not rows:
                rows.append([word])
                continue

            current_row = rows[-1]

            reference_top = (
                current_row[0]["top"]
            )

            if (
                abs(
                    word["top"]
                    - reference_top
                )
                <= self.config.row_tolerance
            ):
                current_row.append(word)

            else:
                rows.append([word])

        result: list[dict[str, Any]] = []

        for index, row_words in enumerate(
            rows
        ):
            row_words.sort(
                key=lambda word: word["x0"]
            )

            result.append(
                {
                    "row_index": index,

                    "words": row_words,

                    "text": " ".join(
                        word["text"]
                        for word in row_words
                    ),

                    "top": min(
                        word["top"]
                        for word in row_words
                    ),

                    "bottom": max(
                        word["bottom"]
                        for word in row_words
                    ),

                    "x0": min(
                        word["x0"]
                        for word in row_words
                    ),

                    "x1": max(
                        word["x1"]
                        for word in row_words
                    ),
                }
            )

        return result

    # =========================================================
    # STRUCTURAL CLEANUP
    # =========================================================

    def _remove_structural_rows(
        self,
        page: Page,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Remove headers and footers detected by the
        DocumentStructureProcessor.
        """

        header = self._normalize(
            page.header_text or ""
        )

        footer = self._normalize(
            page.footer_text or ""
        )

        result: list[dict[str, Any]] = []

        for row in rows:

            text = row["text"].strip()

            normalized = self._normalize(
                text
            )

            if (
                header
                and normalized == header
            ):
                continue

            if (
                footer
                and normalized == footer
            ):
                continue

            if self._is_footer(text):
                continue

            result.append(row)

        return result

    # =========================================================
    # SPATIAL GROUPING
    # =========================================================

    def _split_spatial_groups(
        self,
        rows: list[dict[str, Any]],
        page: Page,
    ) -> list[list[dict[str, Any]]]:
        """
        Split rows into logical/spatial groups.

        Uses:

            vertical distance
            horizontal position
            heading detection
            numbered-section continuity

        The numbered-section continuity is important.

        Example:

            12 Performance metrics
            12.1 Delivery Time
            12.2 Order Accuracy
            12.3 Customer Satisfaction
            12.4 Driver Performance
            12.5 Fleet Utilization

        These should remain one logical region even if
        there is a larger vertical gap before 12.5.
        """

        if not rows:
            return []

        groups: list[list[dict[str, Any]]] = []

        current_group = [rows[0]]

        for row in rows[1:]:

            previous = current_group[-1]

            vertical_gap = (
                row["top"]
                - previous["bottom"]
            )

            horizontal_relation = (
                self._horizontal_relation(
                    previous=previous,
                    current=row,
                    page=page,
                )
            )

            previous_text = (
                previous["text"].strip()
            )

            current_text = (
                row["text"].strip()
            )

            previous_number = (
                self._number_prefix(
                    previous_text
                )
            )

            current_number = (
                self._number_prefix(
                    current_text
                )
            )

            # -------------------------------------------------
            # Detect whether two numbered rows belong to the
            # same top-level section.
            #
            # 12.4
            # 12.5
            #
            # => same section
            # -------------------------------------------------

            same_numbered_section = (
                previous_number is not None
                and current_number is not None
                and self._same_numbered_section(
                    previous_number,
                    current_number,
                )
            )

            should_split = False

            # -------------------------------------------------
            # Numbered children have priority.
            #
            # If both belong to the same parent section,
            # do NOT split merely because of a vertical gap.
            # -------------------------------------------------

            if same_numbered_section:

                should_split = False

            else:

                # ---------------------------------------------
                # Large vertical gap.
                # ---------------------------------------------

                if (
                    vertical_gap
                    >= self.config.vertical_gap_threshold
                ):
                    should_split = True

                # ---------------------------------------------
                # Different spatial column.
                # ---------------------------------------------

                if (
                    horizontal_relation
                    == "different_column"
                ):
                    should_split = True

                # ---------------------------------------------
                # A new heading normally starts a new region.
                # ---------------------------------------------

                if self._looks_like_heading(
                    current_text
                ):
                    if len(current_group) >= 2:
                        should_split = True

            # -------------------------------------------------
            # Start new group or continue current group.
            # -------------------------------------------------

            if should_split:

                if current_group:
                    groups.append(
                        current_group
                    )

                current_group = [row]

            else:

                current_group.append(row)

        if current_group:
            groups.append(
                current_group
            )

        return groups

    def _horizontal_relation(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
        page: Page,
    ) -> str:
        """
        Determine whether two rows occupy the same horizontal
        region or different columns.
        """

        page_width = self._page_width(
            page
        )

        if page_width <= 0:
            return "same_region"

        previous_center = (
            previous["x0"]
            + previous["x1"]
        ) / 2.0

        current_center = (
            current["x0"]
            + current["x1"]
        ) / 2.0

        center_distance = abs(
            current_center
            - previous_center
        )

        threshold = (
            page_width
            * self.config.column_gap_ratio
        )

        # -----------------------------------------------------
        # Actual horizontal overlap.
        # -----------------------------------------------------

        overlap = (
            min(
                previous["x1"],
                current["x1"],
            )
            -
            max(
                previous["x0"],
                current["x0"],
            )
        )

        if overlap > 0:
            return "same_region"

        # -----------------------------------------------------
        # If centers are sufficiently separated, assume
        # different spatial columns.
        # -----------------------------------------------------

        if center_distance > threshold:
            return "different_column"

        return "same_region"

    # =========================================================
    # CLASSIFICATION
    # =========================================================

    def _classify_group(
        self,
        document: Document,
        page: Page,
        rows: list[dict[str, Any]],
        index: int,
    ) -> Region | None:
        """
        Classify one spatial group.
        """

        if not rows:
            return None

        text_lines = [
            row["text"].strip()
            for row in rows
            if row["text"].strip()
        ]

        if not text_lines:
            return None

        text = "\n".join(
            text_lines
        )

        words = [
            word
            for row in rows
            for word in row["words"]
        ]

        if not words:
            return None

        bbox = self._calculate_bbox(
            rows
        )

        scores = {
            "workflow_definition": (
                self._workflow_score(
                    rows
                )
            ),
            "table_candidate": (
                self._table_score(
                    rows
                )
            ),
            "structured_section": (
                self._structured_score(
                    rows
                )
            ),
            "list": (
                self._list_score(
                    rows
                )
            ),
            "heading": (
                self._heading_score(
                    rows
                )
            ),
            "paragraph": (
                self._paragraph_score(
                    rows
                )
            ),
        }

        region_type = max(
            scores,
            key=scores.get,
        )

        confidence = scores[
            region_type
        ]

        # -----------------------------------------------------
        # Very weak classifications are considered unknown.
        # -----------------------------------------------------

        if confidence < 0.35:

            region_type = "unknown"

            confidence = 0.35

        region_id = (
            f"{document.document_id}"
            f"_page_{page.page_number}"
            f"_region_{index}"
        )

        return Region(
            region_id=region_id,

            document_id=document.document_id,

            page_number=page.page_number,

            region_type=region_type,

            confidence=round(
                min(
                    confidence,
                    1.0,
                ),
                3,
            ),

            bbox=bbox,

            text=text,

            words=words,

            rows=rows,

            metadata={
                "source": document.source,

                "word_count": len(words),

                "row_count": len(rows),

                "classification_scores": {
                    key: round(
                        value,
                        3,
                    )
                    for key, value
                    in scores.items()
                },
            },
        )

    # =========================================================
    # WORKFLOW DETECTION
    # =========================================================

    def _workflow_score(
        self,
        rows: list[dict[str, Any]],
    ) -> float:
        """
        Detect workflow-definition style content.

        Example:

            8.1 A user places an order
            8.2 The order is accepted
            8.3 Start preparation
            8.4 The order is ready
            8.5 Customer arrives

        This is NOT the final workflow parser.

        It only identifies regions that deserve deeper
        workflow extraction later.
        """

        if not rows:
            return 0.0

        score = 0.0

        texts = [
            row["text"].strip()
            for row in rows
        ]

        # -----------------------------------------------------
        # Numbered workflow steps.
        # -----------------------------------------------------

        numbered = [
            text
            for text in texts
            if re.match(
                r"^\d+\.\d+(?:\.\d+)?\s+",
                text,
            )
        ]

        if len(numbered) >= 2:
            score += 0.35

        if len(numbered) >= 4:
            score += 0.20

        # -----------------------------------------------------
        # Workflow vocabulary.
        # -----------------------------------------------------

        workflow_words = (
            "procedure",
            "process",
            "workflow",
            "order",
            "accepted",
            "accept",
            "start",
            "preparation",
            "prepare",
            "ready",
            "arrives",
            "arrive",
            "approved",
            "approve",
            "rejected",
            "reject",
            "completed",
            "complete",
            "cancelled",
            "cancel",
            "submit",
            "review",
            "verify",
            "deliver",
            "delivery",
            "payment",
        )

        lower_text = " ".join(
            texts
        ).casefold()

        matches = sum(
            1
            for word in workflow_words
            if word in lower_text
        )

        score += min(
            matches * 0.04,
            0.25,
        )

        # -----------------------------------------------------
        # Explicit workflow/procedure/process heading.
        # -----------------------------------------------------

        if any(
            (
                "procedure"
                in text.casefold()
            )
            or (
                "workflow"
                in text.casefold()
            )
            or (
                "process"
                in text.casefold()
            )
            for text in texts
        ):
            score += 0.20

        return min(
            score,
            1.0,
        )

    # =========================================================
    # TABLE DETECTION
    # =========================================================

    def _table_score(
        self,
        rows: list[dict[str, Any]],
    ) -> float:
        """
        Detect visually structured table-like text.

        This is intentionally called table_candidate because
        a later table extractor must confirm it.
        """

        if len(rows) < 2:
            return 0.0

        score = 0.0

        column_counts = [
            self._estimate_column_count(
                row
            )
            for row in rows
        ]

        multi_column_rows = sum(
            1
            for count in column_counts
            if count >= 2
        )

        ratio = (
            multi_column_rows
            / len(rows)
        )

        if ratio >= 0.60:
            score += 0.40

        if ratio >= 0.80:
            score += 0.20

        # Repeated horizontal positions are a strong
        # table signal.
        if self._has_aligned_columns(
            rows
        ):
            score += 0.30

        text = " ".join(
            row["text"]
            for row in rows
        ).casefold()

        table_keywords = (
            "code",
            "name",
            "description",
            "quantity",
            "price",
            "status",
            "date",
            "value",
        )

        if any(
            keyword in text
            for keyword in table_keywords
        ):
            score += 0.10

        return min(
            score,
            1.0,
        )

    def _estimate_column_count(
        self,
        row: dict[str, Any],
    ) -> int:
        """
        Estimate how many visual columns exist in a row.
        """

        words = row["words"]

        if len(words) < 2:
            return 1

        gaps: list[float] = []

        for previous, current in zip(
            words,
            words[1:],
        ):
            gap = (
                current["x0"]
                - previous["x1"]
            )

            if gap > 20:
                gaps.append(gap)

        return 1 + len(gaps)

    def _has_aligned_columns(
        self,
        rows: list[dict[str, Any]],
    ) -> bool:
        """
        Detect repeated x positions across rows.

        This helps identify tables whose cells are represented
        as ordinary PDF text rather than a native PDF table.
        """

        if len(rows) < 3:
            return False

        positions: list[float] = []

        for row in rows:
            for word in row["words"]:
                positions.append(
                    round(
                        word["x0"] / 10.0
                    )
                    * 10.0
                )

        if not positions:
            return False

        counts: dict[float, int] = {}

        for position in positions:
            counts[position] = (
                counts.get(
                    position,
                    0,
                )
                + 1
            )

        repeated_positions = sum(
            1
            for count in counts.values()
            if count >= 3
        )

        return repeated_positions >= 2

    # =========================================================
    # STRUCTURED SECTION
    # =========================================================

    def _structured_score(
        self,
        rows: list[dict[str, Any]],
    ) -> float:
        """
        Score numbered structured content.

        Handles:

            10.1 Something
            10.2 Something
            10.3 Something

        and:

            12 Performance metrics
            12.1 Delivery Time
            12.2 Order Accuracy
            12.3 Customer Satisfaction
        """

        if not rows:
            return 0.0

        score = 0.0

        texts = [
            row["text"].strip()
            for row in rows
            if row["text"].strip()
        ]

        if not texts:
            return 0.0

        numbered_items = []

        for text in texts:

            if re.match(
                r"^\d+(?:\.\d+)+\s+",
                text,
            ):
                numbered_items.append(
                    text
                )

        # -----------------------------------------------------
        # Multiple numbered children.
        # -----------------------------------------------------

        if len(numbered_items) >= 2:
            score += 0.50

        if len(numbered_items) >= 4:
            score += 0.20

        # -----------------------------------------------------
        # Parent section heading.
        # -----------------------------------------------------

        has_parent_section = any(
            re.match(
                r"^\d+\s+.+",
                text,
            )
            for text in texts
        )

        if has_parent_section:
            score += 0.20

        # -----------------------------------------------------
        # Single numbered item.
        # -----------------------------------------------------

        if (
            len(numbered_items) == 1
            and re.match(
                r"^\d+(?:\.\d+)+\s+",
                texts[0],
            )
        ):
            score += 0.25

        return min(
            score,
            1.0,
        )

    # =========================================================
    # LIST DETECTION
    # =========================================================

    def _list_score(
        self,
        rows: list[dict[str, Any]],
    ) -> float:
        """
        Detect bullet/list regions.
        """

        if not rows:
            return 0.0

        list_rows = sum(
            1
            for row in rows
            if re.match(
                r"^(?:•|▪|●|○|-|–|—|\*)\s+",
                row["text"].strip(),
            )
        )

        if list_rows == 0:
            return 0.0

        ratio = (
            list_rows
            / len(rows)
        )

        return min(
            0.30 + ratio * 0.60,
            1.0,
        )

    # =========================================================
    # HEADING DETECTION
    # =========================================================

    def _heading_score(
        self,
        rows: list[dict[str, Any]],
    ) -> float:
        """
        Detect a single heading row.
        """

        if len(rows) != 1:
            return 0.0

        text = rows[0]["text"].strip()

        if not text:
            return 0.0

        score = 0.0

        # Numbered heading.
        if re.match(
            r"^\d+\s+",
            text,
        ):
            score += 0.60

        # Colon-ended heading.
        if text.endswith(":"):
            score += 0.20

        # Short text is more likely to be a heading.
        if len(text) < 100:
            score += 0.10

        # Mostly uppercase.
        letters = [
            char
            for char in text
            if char.isalpha()
        ]

        if letters:

            uppercase_ratio = (
                sum(
                    1
                    for char in letters
                    if char.isupper()
                )
                / len(letters)
            )

            if uppercase_ratio >= 0.80:
                score += 0.10

        return min(
            score,
            1.0,
        )

    # =========================================================
    # PARAGRAPH DETECTION
    # =========================================================

    def _paragraph_score(
        self,
        rows: list[dict[str, Any]],
    ) -> float:
        """
        Score ordinary paragraph-like content.
        """

        if not rows:
            return 0.0

        if len(rows) >= 3:
            return 0.65

        if len(rows) == 2:
            return 0.50

        text = rows[0]["text"]

        if len(text) > 80:
            return 0.60

        return 0.40

    # =========================================================
    # VISUAL REGIONS
    # =========================================================

    def _detect_visual_regions(
        self,
        document: Document,
        page: Page,
        existing_regions: list[Region],
    ) -> list[Region]:
        """
        Register visual content.

        This does NOT perform OCR or computer vision.

        It simply creates regions that tell later components:

            "This page/image needs visual processing."
        """

        if not page.has_visual_content:
            return []

        # -----------------------------------------------------
        # Actual embedded images.
        # -----------------------------------------------------

        if page.images:

            regions: list[Region] = []

            for index, image in enumerate(
                page.images
            ):

                bbox = self._image_bbox(
                    image=image,
                    page=page,
                )

                regions.append(
                    Region(
                        region_id=(
                            f"{document.document_id}"
                            f"_page_{page.page_number}"
                            f"_visual_{index}"
                        ),

                        document_id=(
                            document.document_id
                        ),

                        page_number=(
                            page.page_number
                        ),

                        region_type="visual",

                        confidence=0.90,

                        bbox=bbox,

                        text="",

                        words=[],

                        rows=[],

                        metadata={
                            "source": (
                                document.source
                            ),
                            "requires_visual_extraction": True,
                            "image_index": index,
                        },
                    )
                )

            return regions

        # -----------------------------------------------------
        # Sparse page with no embedded image.
        #
        # Possible:
        #
        #   scanned page
        #   diagram
        #   screenshot
        #   vector graphic
        # -----------------------------------------------------

        if not existing_regions:

            page_width = (
                self._page_width(page)
            )

            page_height = (
                self._page_height(page)
            )

            return [
                Region(
                    region_id=(
                        f"{document.document_id}"
                        f"_page_{page.page_number}"
                        f"_visual_page"
                    ),

                    document_id=(
                        document.document_id
                    ),

                    page_number=(
                        page.page_number
                    ),

                    region_type="visual_candidate",

                    confidence=0.70,

                    bbox=BoundingBox(
                        x0=0.0,
                        top=0.0,
                        x1=page_width,
                        bottom=page_height,
                    ),

                    text=page.text,

                    words=page.layout_words,

                    rows=[],

                    metadata={
                        "source": (
                            document.source
                        ),
                        "requires_rendering": True,
                        "possible_scan_or_diagram": True,
                    },
                )
            ]

        return []

    # =========================================================
    # BOUNDING BOX
    # =========================================================

    @staticmethod
    def _calculate_bbox(
        rows: list[dict[str, Any]],
    ) -> BoundingBox:
        """
        Calculate bounding box covering all rows.
        """

        return BoundingBox(
            x0=min(
                row["x0"]
                for row in rows
            ),

            top=min(
                row["top"]
                for row in rows
            ),

            x1=max(
                row["x1"]
                for row in rows
            ),

            bottom=max(
                row["bottom"]
                for row in rows
            ),
        )

    @staticmethod
    def _image_bbox(
        image: dict[str, Any],
        page: Page,
    ) -> BoundingBox:
        """
        Create a bounding box for an embedded image.
        """

        page_width = (
            RegionDetector._page_width(
                page
            )
        )

        page_height = (
            RegionDetector._page_height(
                page
            )
        )

        return BoundingBox(
            x0=float(
                image.get(
                    "x0",
                    0.0,
                )
            ),

            top=float(
                image.get(
                    "top",
                    0.0,
                )
            ),

            x1=float(
                image.get(
                    "x1",
                    page_width,
                )
            ),

            bottom=float(
                image.get(
                    "bottom",
                    page_height,
                )
            ),
        )

    # =========================================================
    # NUMBERING HELPERS
    # =========================================================

    @staticmethod
    def _number_prefix(
        text: str,
    ) -> str | None:
        """
        Extract a numbered structure prefix.

        Examples:

            12.5 Fleet Utilization
                -> "12.5"

            12 Performance metrics
                -> "12"

            8.1.2 Something
                -> "8.1.2"

            Normal text
                -> None
        """

        match = re.match(
            r"^(\d+(?:\.\d+)*)\s+",
            text.strip(),
        )

        if not match:
            return None

        return match.group(1)

    @staticmethod
    def _same_numbered_section(
        previous_number: str,
        current_number: str,
    ) -> bool:
        """
        Determine whether two numbered items belong to the
        same top-level section.

        Examples:

            12.4 + 12.5
                -> True

            8.1 + 8.2
                -> True

            8.5 + 9.1
                -> False

            12.1.1 + 12.1.2
                -> True

            12.1 + 13.1
                -> False
        """

        previous_root = (
            previous_number.split(".")[0]
        )

        current_root = (
            current_number.split(".")[0]
        )

        return (
            previous_root
            == current_root
        )

    # =========================================================
    # PAGE HELPERS
    # =========================================================

    @staticmethod
    def _page_width(
        page: Page,
    ) -> float:
        try:
            return float(
                page.metadata.get(
                    "page_width",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @staticmethod
    def _page_height(
        page: Page,
    ) -> float:
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

    # =========================================================
    # TEXT HELPERS
    # =========================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """
        Normalize whitespace for comparisons.
        """

        return re.sub(
            r"\s+",
            " ",
            text.strip(),
        ).casefold()

    @staticmethod
    def _is_footer(
        text: str,
    ) -> bool:
        """
        Detect common PDF footer formats.
        """

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
                text.strip(),
                re.IGNORECASE,
            )
            for pattern in patterns
        )

    @staticmethod
    def _looks_like_heading(
        text: str,
    ) -> bool:
        """
        Determine whether a row looks like a section heading.
        """

        text = text.strip()

        if not text:
            return False

        # Examples:
        #
        # 12 Performance metrics
        # 8 Order fulfillment procedure
        #

        if re.match(
            r"^\d+\s+[A-Za-z]",
            text,
        ):
            return True

        # Example:
        #
        # General note:
        #

        if text.endswith(":"):
            return True

        return False

    # =========================================================
    # FALLBACK
    # =========================================================

    def _fallback_page_region(
        self,
        document: Document,
        page: Page,
    ) -> list[Region]:
        """
        Create one region when layout information is unavailable.
        """

        if not page.text.strip():
            return []

        page_width = (
            self._page_width(page)
        )

        page_height = (
            self._page_height(page)
        )

        return [
            Region(
                region_id=(
                    f"{document.document_id}"
                    f"_page_{page.page_number}"
                    f"_region_0"
                ),

                document_id=(
                    document.document_id
                ),

                page_number=(
                    page.page_number
                ),

                region_type="unknown",

                confidence=0.30,

                bbox=BoundingBox(
                    x0=0.0,
                    top=0.0,
                    x1=page_width,
                    bottom=page_height,
                ),

                text=page.text,

                words=[],

                rows=[],

                metadata={
                    "source": document.source,
                    "fallback": True,
                    "reason": (
                        "layout_words_unavailable"
                    ),
                },
            )
        ]

def _detect_embedded_image_regions(
    self,
    document,
) -> list[Region]:
    """
    Detect embedded PDF images as explicit visual regions.

    This is intentionally based on the PDF image object's
    actual bounding box.

    We do NOT infer visual content from word count.

    This allows us to correctly detect diagrams that are
    stored inside the PDF as raster images.
    """

    regions: list[Region] = []

    for page in document.pages:

        image_objects = page.metadata.get(
            "image_objects",
            [],
        )

        for index, image in enumerate(
            image_objects
        ):

            x0 = float(
                image.get(
                    "x0",
                    0.0,
                )
            )

            x1 = float(
                image.get(
                    "x1",
                    0.0,
                )
            )

            top = float(
                image.get(
                    "top",
                    0.0,
                )
            )

            bottom = float(
                image.get(
                    "bottom",
                    0.0,
                )
            )

            if x1 <= x0:
                continue

            if bottom <= top:
                continue

            bbox = BoundingBox(
                x0=x0,
                top=top,
                x1=x1,
                bottom=bottom,
            )

            region_id = (
                f"{document.document_id}"
                f"_page_{page.page_number}"
                f"_image_{index}"
            )

            region = Region(
                region_id=region_id,
                document_id=document.document_id,
                page_number=page.page_number,
                region_type="image",
                confidence=0.98,
                bbox=bbox,
                text="",
                words=[],
                rows=[],
                metadata={
                    "source": "pdf_image_object",
                    "has_embedded_image": True,
                    "image_index": index,
                    "width": bbox.width,
                    "height": bbox.height,
                    "area": bbox.area,
                },
            )

            regions.append(
                region
            )

    return regions