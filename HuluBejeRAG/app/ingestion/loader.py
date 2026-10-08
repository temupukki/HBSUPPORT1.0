from pathlib import Path
from typing import Any

from pypdf import PdfReader
import pdfplumber

from app.models.document import Document, Page
from app.extraction.document_structure import DocumentStructureProcessor


class DocumentLoader:
    """
    Loads PDF knowledge documents and prepares them for
    downstream document structure analysis.

    Responsibilities:
    - Discover PDF files
    - Extract raw page text
    - Preserve page boundaries
    - Extract positional words
    - Detect embedded PDF images
    - Preserve image bounding boxes
    - Distinguish significant images from decorative objects
    - Preserve page geometry
    - Run document-level structural processing
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    # -------------------------------------------------------------
    # Image filtering
    # -------------------------------------------------------------
    #
    # Small images are commonly:
    # - logos
    # - header decorations
    # - separator lines
    # - tiny icons
    #
    # We preserve them in Page.images but do not treat them as
    # meaningful visual content.
    #
    MIN_SIGNIFICANT_IMAGE_WIDTH = 120.0
    MIN_SIGNIFICANT_IMAGE_HEIGHT = 40.0
    MIN_SIGNIFICANT_IMAGE_AREA = 10_000.0

    def __init__(
        self,
        knowledge_dir: str,
    ):
        self.knowledge_dir = Path(
            knowledge_dir
        )

        self.structure_processor = (
            DocumentStructureProcessor()
        )

    def load(
        self,
    ) -> list[Document]:

        if not self.knowledge_dir.exists():
            raise FileNotFoundError(
                f"Knowledge directory does not exist: "
                f"{self.knowledge_dir}"
            )

        if not self.knowledge_dir.is_dir():
            raise NotADirectoryError(
                f"Knowledge path is not a directory: "
                f"{self.knowledge_dir}"
            )

        documents: list[Document] = []

        for file_path in sorted(
            self.knowledge_dir.iterdir()
        ):

            if not file_path.is_file():
                continue

            if (
                file_path.suffix.lower()
                not in self.SUPPORTED_EXTENSIONS
            ):
                continue

            document = self._load_pdf(
                file_path
            )

            documents.append(
                document
            )

        return documents

    def _load_pdf(
        self,
        file_path: Path,
    ) -> Document:

        reader = PdfReader(
            file_path
        )

        pages: list[Page] = []

        with pdfplumber.open(
            file_path
        ) as layout_pdf:

            for index, pdf_page in enumerate(
                reader.pages
            ):

                page_number = (
                    index + 1
                )

                # -------------------------------------------------
                # Normal text
                # -------------------------------------------------

                raw_text = (
                    pdf_page.extract_text()
                    or ""
                )

                text = (
                    self._prepare_page_text(
                        raw_text
                    )
                )

                # -------------------------------------------------
                # Layout information
                # -------------------------------------------------

                layout_page = (
                    layout_pdf.pages[index]
                )

                layout_words = (
                    self._extract_layout_words(
                        layout_page
                    )
                )

                # -------------------------------------------------
                # Embedded PDF images
                # -------------------------------------------------

                image_objects = (
                    self._extract_image_objects(
                        layout_page
                    )
                )

                significant_images = [
                    image
                    for image in image_objects
                    if image.get(
                        "is_significant",
                        False,
                    )
                ]

                has_images = bool(
                    image_objects
                )

                has_significant_images = bool(
                    significant_images
                )

                # -------------------------------------------------
                # Page metadata
                # -------------------------------------------------

                metadata: dict[
                    str,
                    Any,
                ] = {

                    "has_embedded_images": (
                        has_images
                    ),

                    "has_significant_images": (
                        has_significant_images
                    ),

                    "has_visual_content": (
                        has_significant_images
                    ),

                    "image_count": len(
                        image_objects
                    ),

                    "significant_image_count": (
                        len(
                            significant_images
                        )
                    ),

                    "image_objects": (
                        image_objects
                    ),

                    "significant_image_objects": (
                        significant_images
                    ),

                    "text_length": len(
                        text
                    ),

                    "word_count": len(
                        layout_words
                    ),

                    "page_width": float(
                        layout_page.width
                    ),

                    "page_height": float(
                        layout_page.height
                    ),
                }

                page = Page(
                    page_number=page_number,
                    text=text,
                    tables=[],
                    images=image_objects,
                    has_visual_content=(
                        has_significant_images
                    ),
                    metadata=metadata,
                    raw_text=raw_text,
                    layout_words=layout_words,
                )

                pages.append(
                    page
                )

        if not pages:
            raise ValueError(
                f"No pages found in PDF: "
                f"{file_path}"
            )

        document_id = (
            file_path.stem
        )

        metadata: dict[
            str,
            Any,
        ] = {

            "source": file_path.name,

            "file_name": (
                file_path.name
            ),

            "file_path": str(
                file_path
            ),

            "language": "en",

            "version": "1.0",

            "file_type": "pdf",

            "page_count": len(
                pages
            ),
        }

        document = Document(
            document_id=document_id,
            source=file_path.name,
            pages=pages,
            metadata=metadata,
        )

        document = (
            self.structure_processor.process(
                document
            )
        )

        return document

    @staticmethod
    def _extract_layout_words(
        layout_page,
    ) -> list[
        dict[str, Any]
    ]:

        try:

            words = (
                layout_page.extract_words(
                    keep_blank_chars=False,
                    use_text_flow=False,
                )
            )

        except Exception:

            return []

        cleaned_words: list[
            dict[str, Any]
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

            cleaned_words.append(
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

                    "doctop": float(
                        word.get(
                            "doctop",
                            0.0,
                        )
                    ),

                    "upright": bool(
                        word.get(
                            "upright",
                            True,
                        )
                    ),
                }
            )

        return cleaned_words

    @classmethod
    def _extract_image_objects(
        cls,
        layout_page,
    ) -> list[
        dict[str, Any]
    ]:
        """
        Extract embedded PDF image objects.

        We preserve ALL images, but mark only meaningful-sized
        images as significant.

        This prevents things such as:

            73 x 73 logo

        or:

            429 x 3 decorative line

        from becoming workflow diagrams.
        """

        try:

            images = (
                layout_page.images
            )

        except Exception:

            return []

        image_objects: list[
            dict[str, Any]
        ] = []

        for index, image in enumerate(
            images
        ):

            try:

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

                width = float(
                    image.get(
                        "width",
                        max(
                            0.0,
                            x1 - x0,
                        ),
                    )
                )

                height = float(
                    image.get(
                        "height",
                        max(
                            0.0,
                            bottom - top,
                        ),
                    )
                )

                area = (
                    width
                    * height
                )

                is_significant = (
                    width
                    >= cls.MIN_SIGNIFICANT_IMAGE_WIDTH
                    and height
                    >= cls.MIN_SIGNIFICANT_IMAGE_HEIGHT
                    and area
                    >= cls.MIN_SIGNIFICANT_IMAGE_AREA
                )

                image_objects.append(
                    {
                        "index": index,

                        "x0": x0,

                        "x1": x1,

                        "top": top,

                        "bottom": bottom,

                        "width": width,

                        "height": height,

                        "area": area,

                        "object_type": "image",

                        "is_significant": (
                            is_significant
                        ),
                    }
                )

            except Exception:
                continue

        return image_objects

    @staticmethod
    def _prepare_page_text(
        text: str,
    ) -> str:

        if not text:
            return ""

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        text = text.replace(
            "\t",
            " ",
        )

        lines = text.split(
            "\n"
        )

        cleaned_lines: list[
            str
        ] = []

        previous_blank = False

        for line in lines:

            line = line.strip()

            if not line:

                if not previous_blank:
                    cleaned_lines.append(
                        ""
                    )

                previous_blank = True

                continue

            cleaned_lines.append(
                line
            )

            previous_blank = False

        return "\n".join(
            cleaned_lines
        ).strip()