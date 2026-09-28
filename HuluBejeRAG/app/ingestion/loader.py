from pathlib import Path

from pypdf import PdfReader

from app.models.document import Document, Page


class DocumentLoader:
    """
    Loads PDF knowledge documents while preserving page boundaries.
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = Path(knowledge_dir)

    def load(self) -> list[Document]:
        """
        Load all supported PDF documents.
        """

        if not self.knowledge_dir.exists():
            raise FileNotFoundError(
                f"Knowledge directory does not exist: "
                f"{self.knowledge_dir}"
            )

        documents: list[Document] = []

        for file_path in sorted(self.knowledge_dir.iterdir()):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            documents.append(
                self._load_pdf(file_path)
            )

        return documents

    def _load_pdf(self, file_path: Path) -> Document:
        """
        Extract text from a PDF while preserving page numbers.
        """

        reader = PdfReader(file_path)

        pages: list[Page] = []

        for index, pdf_page in enumerate(reader.pages):

            text = pdf_page.extract_text() or ""

            text = text.strip()

            if not text:
                continue

            pages.append(
                Page(
                    page_number=index + 1,
                    text=text,
                )
            )

        if not pages:
            raise ValueError(
                f"No extractable text found in PDF: "
                f"{file_path}"
            )

        document_id = file_path.stem

        metadata = {
            "source": file_path.name,
            "language": "en",
            "version": "1.0",
            "file_type": "pdf",
        }

        return Document(
            document_id=document_id,
            source=file_path.name,
            pages=pages,
            metadata=metadata,
        )