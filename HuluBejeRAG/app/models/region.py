from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    """
    Represents a rectangular area on a PDF page.
    """

    x0: float
    top: float
    x1: float
    bottom: float

    @property
    def width(self) -> float:
        return max(
            0.0,
            self.x1 - self.x0,
        )

    @property
    def height(self) -> float:
        return max(
            0.0,
            self.bottom - self.top,
        )

    @property
    def area(self) -> float:
        return (
            self.width
            * self.height
        )


@dataclass
class Region:
    """
    Represents a detected content region on a PDF page.

    Region types can include:

    - paragraph
    - heading
    - list
    - structured_section
    - workflow_definition
    - table_candidate
    - visual
    - image
    - diagram
    - workflow_diagram
    - unknown
    """

    region_id: str
    document_id: str
    page_number: int
    region_type: str
    confidence: float
    bbox: BoundingBox

    text: str = ""

    words: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    rows: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def word_count(self) -> int:
        return len(
            self.words
        )

    @property
    def row_count(self) -> int:
        return len(
            self.rows
        )

    @property
    def has_text(self) -> bool:
        return bool(
            self.text.strip()
        )

    @property
    def has_visual_content(self) -> bool:
        return (
            self.region_type
            in {
                "visual",
                "visual_content",
                "workflow_diagram",
                "diagram",
                "image",
            }
            or bool(
                self.metadata.get(
                    "has_embedded_image",
                    False,
                )
            )
        )