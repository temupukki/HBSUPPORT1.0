from dataclasses import dataclass, field
from typing import Any

from app.models.document import Document
from app.models.region import BoundingBox, Region


@dataclass
class VisualElement:
    """
    Represents one detected element inside a visual region.

    Current element types:
    - text

    Future element types:
    - box
    - decision
    - arrow
    - connector
    - image
    - label
    """

    element_id: str
    element_type: str
    text: str = ""
    bbox: BoundingBox | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class VisualRegion:
    """
    Represents a meaningful visual region extracted
    from a PDF page.
    """

    visual_id: str
    document_id: str
    page_number: int

    bbox: BoundingBox

    elements: list[
        VisualElement
    ] = field(
        default_factory=list
    )

    text: str = ""

    confidence: float = 0.0

    metadata: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    @property
    def element_count(self) -> int:
        return len(
            self.elements
        )

    @property
    def has_text(self) -> bool:
        return bool(
            self.text.strip()
        )


class VisualExtractor:
    """
    Extracts visual content from PDF documents.

    Current stage:

    1. Detect significant embedded images.
    2. Preserve their coordinates.
    3. Associate nearby PDF text.
    4. Create a VisualRegion.
    5. Preserve text layout inside the visual area.

    IMPORTANT:

    This component does NOT yet understand the actual
    diagram structure.

    Future stages will detect:

    - workflow boxes
    - arrows
    - connectors
    - decisions
    - branch labels
    - diagram nodes
    - relationships
    """

    MIN_IMAGE_WIDTH = 120.0
    MIN_IMAGE_HEIGHT = 40.0
    MIN_IMAGE_AREA = 10_000.0

    def __init__(
        self,
        minimum_visual_confidence: float = 0.30,
    ):
        self.minimum_visual_confidence = (
            minimum_visual_confidence
        )

    # =============================================================
    # DOCUMENT
    # =============================================================

    def extract_from_document(
        self,
        document: Document,
        regions: list[Region] | None = None,
    ) -> list[VisualRegion]:
        """
        Extract visual regions from a document.

        Significant embedded PDF images are the primary source.

        `regions` is optional because visual content can exist
        independently from text regions.
        """

        visual_regions: list[
            VisualRegion
        ] = []

        for page in document.pages:

            image_objects = (
                page.metadata.get(
                    "significant_image_objects",
                    [],
                )
            )

            for image in image_objects:

                visual_region = (
                    self._extract_from_image(
                        document=document,
                        page=page,
                        image=image,
                    )
                )

                if visual_region is not None:
                    visual_regions.append(
                        visual_region
                    )

        return visual_regions

    # =============================================================
    # IMAGE
    # =============================================================

    def _extract_from_image(
        self,
        document: Document,
        page,
        image: dict[str, Any],
    ) -> VisualRegion | None:
        """
        Convert one significant PDF image object into
        a VisualRegion.
        """

        width = float(
            image.get(
                "width",
                0.0,
            )
        )

        height = float(
            image.get(
                "height",
                0.0,
            )
        )

        area = float(
            image.get(
                "area",
                width * height,
            )
        )

        if not self._is_significant_image(
            width=width,
            height=height,
            area=area,
        ):
            return None

        bbox = BoundingBox(
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
                    0.0,
                )
            ),
            bottom=float(
                image.get(
                    "bottom",
                    0.0,
                )
            ),
        )

        visual_id = (
            f"{document.document_id}"
            f"_visual_page_{page.page_number}"
            f"_image_{image.get('index', 0)}"
        )

        # ---------------------------------------------------------
        # Find PDF text that physically overlaps the image.
        #
        # This is useful because diagrams may have text extracted
        # separately from the image itself.
        # ---------------------------------------------------------

        words = self._words_inside_bbox(
            page.layout_words,
            bbox,
        )

        text = self._reconstruct_text(
            words
        )

        elements = (
            self._create_text_elements(
                visual_id=visual_id,
                words=words,
            )
        )

        confidence = (
            self._calculate_confidence(
                image=image,
                elements=elements,
            )
        )

        return VisualRegion(
            visual_id=visual_id,
            document_id=document.document_id,
            page_number=page.page_number,
            bbox=bbox,
            elements=elements,
            text=text,
            confidence=confidence,
            metadata={
                "source": "embedded_pdf_image",

                "image_index": image.get(
                    "index"
                ),

                "image_width": width,

                "image_height": height,

                "image_area": area,

                "image_object_type": image.get(
                    "object_type"
                ),

                "is_significant": True,

                "extraction_stage": (
                    "embedded_image_detection"
                ),
            },
        )

    # =============================================================
    # IMAGE FILTERING
    # =============================================================

    def _is_significant_image(
        self,
        width: float,
        height: float,
        area: float,
    ) -> bool:
        """
        Reject tiny logos, separators and decorative objects.
        """

        if (
            width
            < self.MIN_IMAGE_WIDTH
        ):
            return False

        if (
            height
            < self.MIN_IMAGE_HEIGHT
        ):
            return False

        if (
            area
            < self.MIN_IMAGE_AREA
        ):
            return False

        return True

    # =============================================================
    # WORDS
    # =============================================================

    @staticmethod
    def _words_inside_bbox(
        words: list[
            dict[str, Any]
        ],
        bbox: BoundingBox,
    ) -> list[
        dict[str, Any]
    ]:
        """
        Return words whose center point lies inside
        the visual bounding box.
        """

        result: list[
            dict[str, Any]
        ] = []

        for word in words:

            x0 = float(
                word.get(
                    "x0",
                    0.0,
                )
            )

            x1 = float(
                word.get(
                    "x1",
                    0.0,
                )
            )

            top = float(
                word.get(
                    "top",
                    0.0,
                )
            )

            bottom = float(
                word.get(
                    "bottom",
                    0.0,
                )
            )

            center_x = (
                x0 + x1
            ) / 2.0

            center_y = (
                top + bottom
            ) / 2.0

            if (
                bbox.x0
                <= center_x
                <= bbox.x1
                and
                bbox.top
                <= center_y
                <= bbox.bottom
            ):
                result.append(
                    word
                )

        return result

    # =============================================================
    # TEXT ELEMENTS
    # =============================================================

    def _create_text_elements(
        self,
        visual_id: str,
        words: list[
            dict[str, Any]
        ],
    ) -> list[
        VisualElement
    ]:
        """
        Convert layout words inside the visual area
        into initial visual text elements.
        """

        elements: list[
            VisualElement
        ] = []

        for index, word in enumerate(
            words
        ):

            text = str(
                word.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            bbox = BoundingBox(
                x0=float(
                    word.get(
                        "x0",
                        0.0,
                    )
                ),
                top=float(
                    word.get(
                        "top",
                        0.0,
                    )
                ),
                x1=float(
                    word.get(
                        "x1",
                        0.0,
                    )
                ),
                bottom=float(
                    word.get(
                        "bottom",
                        0.0,
                    )
                ),
            )

            elements.append(
                VisualElement(
                    element_id=(
                        f"{visual_id}"
                        f"_text_{index}"
                    ),
                    element_type="text",
                    text=text,
                    bbox=bbox,
                    confidence=0.95,
                    metadata={
                        "source": (
                            "pdf_layout_word"
                        ),
                    },
                )
            )

        return elements

    # =============================================================
    # TEXT RECONSTRUCTION
    # =============================================================

    @staticmethod
    def _reconstruct_text(
        words: list[
            dict[str, Any]
        ],
    ) -> str:
        """
        Reconstruct readable text from words
        using their PDF coordinates.
        """

        if not words:
            return ""

        rows: list[
            list[
                dict[str, Any]
            ]
        ] = []

        sorted_words = sorted(
            words,
            key=lambda word: (
                float(
                    word.get(
                        "top",
                        0.0,
                    )
                ),
                float(
                    word.get(
                        "x0",
                        0.0,
                    )
                ),
            ),
        )

        row_tolerance = 4.0

        for word in sorted_words:

            if not rows:

                rows.append(
                    [word]
                )

                continue

            previous_row = (
                rows[-1]
            )

            previous_top = float(
                previous_row[0].get(
                    "top",
                    0.0,
                )
            )

            current_top = float(
                word.get(
                    "top",
                    0.0,
                )
            )

            if abs(
                current_top
                - previous_top
            ) <= row_tolerance:

                previous_row.append(
                    word
                )

            else:

                rows.append(
                    [word]
                )

        text_lines: list[str] = []

        for row in rows:

            row.sort(
                key=lambda word: float(
                    word.get(
                        "x0",
                        0.0,
                    )
                )
            )

            line_parts: list[
                str
            ] = []

            for word in row:

                value = str(
                    word.get(
                        "text",
                        "",
                    )
                ).strip()

                if value:
                    line_parts.append(
                        value
                    )

            if line_parts:

                text_lines.append(
                    " ".join(
                        line_parts
                    )
                )

        return "\n".join(
            text_lines
        ).strip()

    # =============================================================
    # CONFIDENCE
    # =============================================================

    @staticmethod
    def _calculate_confidence(
        image: dict[str, Any],
        elements: list[
            VisualElement
        ],
    ) -> float:
        """
        Calculate extraction confidence based on the
        quality of the detected image object.

        This is NOT ML confidence.
        """

        score = 0.0

        width = float(
            image.get(
                "width",
                0.0,
            )
        )

        height = float(
            image.get(
                "height",
                0.0,
            )
        )

        # Large image gets stronger confidence.
        if width >= 500:
            score += 0.40
        elif width >= 250:
            score += 0.30
        else:
            score += 0.20

        if height >= 200:
            score += 0.30
        elif height >= 100:
            score += 0.20
        else:
            score += 0.10

        if elements:
            score += 0.20

        if image.get(
            "is_significant",
            False,
        ):
            score += 0.10

        return round(
            min(
                1.0,
                score,
            ),
            3,
        )