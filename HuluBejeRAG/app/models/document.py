from dataclasses import dataclass, field
from typing import Any


@dataclass
class Page:
    """
    Represents one PDF page.

    Keeps both extracted content and structural/layout
    information for downstream document analysis.
    """

    page_number: int

    # Cleaned page text.
    text: str = ""

    # Extracted tables.
    tables: list[dict[str, Any]] = field(
        default_factory=list
    )

    # Extracted/registered images.
    images: list[dict[str, Any]] = field(
        default_factory=list
    )

    # Whether the page may contain visual content.
    has_visual_content: bool = False

    # Page-level metadata.
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # Original text before structural cleanup.
    raw_text: str = ""

    # Repeated document header detected on this page.
    header_text: str | None = None

    # Page footer detected on this page.
    footer_text: str | None = None

    # Whether this page likely continues the previous page.
    continues_previous: bool = False

    # Whether this page likely continues onto the next page.
    continues_next: bool = False

    # Positional PDF words.
    #
    # Example:
    #
    # {
    #     "text": "Order",
    #     "x0": 72.0,
    #     "x1": 105.5,
    #     "top": 120.4,
    #     "bottom": 132.8
    # }
    #
    # Used later for:
    # - tables
    # - columns
    # - workflows
    # - diagrams
    # - structured regions
    layout_words: list[dict[str, Any]] = field(
        default_factory=list
    )

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    @property
    def has_tables(self) -> bool:
        return bool(self.tables)

    @property
    def has_images(self) -> bool:
        return bool(self.images)


@dataclass
class Document:
    """
    Represents a complete knowledge document.
    """

    document_id: str

    source: str

    pages: list[Page] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def content(self) -> str:
        """
        Return cleaned text from all pages.
        """

        return "\n\n".join(
            page.text
            for page in self.pages
            if page.text.strip()
        )