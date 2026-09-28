from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Page:
    """
    Represents one page extracted from a PDF.
    """

    page_number: int
    text: str


@dataclass
class Document:
    """
    Represents a complete source document.
    """

    document_id: str

    source: str

    pages: list[Page] = field(default_factory=list)

    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def content(self) -> str:
        """
        Return the complete document text.
        """

        return "\n\n".join(
            page.text
            for page in self.pages
            if page.text.strip()
        )