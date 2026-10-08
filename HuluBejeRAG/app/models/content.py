from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentBlock:
    """
    Base representation of a piece of document content.

    A content block is the normalized unit that later
    analysis, workflow extraction, and chunking stages
    will work with.
    """

    block_id: str

    page_number: int

    block_type: str

    content: Any

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class TextBlock(ContentBlock):
    """
    Normal textual content.
    """

    pass


@dataclass
class TableBlock(ContentBlock):
    """
    Structured table content.
    """

    pass


@dataclass
class ImageBlock(ContentBlock):
    """
    Visual content such as:

    - diagram
    - flowchart
    - screenshot
    - scanned page
    - image
    """

    pass