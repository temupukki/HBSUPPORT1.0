from app.models.document import Document, Page
from app.models.content import (
    ContentBlock,
    TextBlock,
    TableBlock,
    ImageBlock,
)


class ContentAnalyzer:
    """
    Converts raw document pages into normalized content blocks.

    The analyzer does NOT yet perform:
    - workflow extraction
    - semantic understanding
    - LLM processing
    - OCR
    - vision reasoning

    Its current responsibility is to organize the content
    that has already been extracted by the document loader.
    """

    def analyze_document(
        self,
        document: Document,
    ) -> list[ContentBlock]:
        """
        Analyze every page in a document.

        Returns a flat list of content blocks.
        """

        blocks: list[ContentBlock] = []

        for page in document.pages:

            page_blocks = self.analyze_page(
                document=document,
                page=page,
            )

            blocks.extend(page_blocks)

        return blocks

    def analyze_page(
        self,
        document: Document,
        page: Page,
    ) -> list[ContentBlock]:
        """
        Convert one page into normalized content blocks.
        """

        blocks: list[ContentBlock] = []

        # --------------------------------------------------
        # TEXT
        # --------------------------------------------------

        if page.has_text:

            text_block = TextBlock(
                block_id=(
                    f"{document.document_id}"
                    f"_page_{page.page_number}"
                    f"_text"
                ),
                page_number=page.page_number,
                block_type="table",
                content=page.text,
                metadata={
                    "source": document.source,
                    "content_type": "text",
                },
            )

            blocks.append(text_block)

        # --------------------------------------------------
        # TABLES
        # --------------------------------------------------

        for index, table in enumerate(page.tables):

            table_block = TableBlock(
                block_id=(
                    f"{document.document_id}"
                    f"_page_{page.page_number}"
                    f"_table_{index}"
                ),
                page_number=page.page_number,
                block_type="table",
                content=table,
                metadata={
                    "source": document.source,
                    "content_type": "table",
                    "table_index": index,
                },
            )

            blocks.append(table_block)

        # --------------------------------------------------
        # IMAGES
        # --------------------------------------------------

        for index, image in enumerate(page.images):

            image_block = ImageBlock(
                block_id=(
                    f"{document.document_id}"
                    f"_page_{page.page_number}"
                    f"_image_{index}"
                ),
                page_number=page.page_number,
                block_type="table",
                content=image,
                metadata={
                    "source": document.source,
                    "content_type": "image",
                    "image_index": index,
                },
            )

            blocks.append(image_block)

        

        if (
            page.has_visual_content
            and not page.images
        ):

            image_block = ImageBlock(
                block_id=(
                    f"{document.document_id}"
                    f"_page_{page.page_number}"
                    f"_visual"
                ),
                page_number=page.page_number,
                block_type="table",
                content=None,
                metadata={
                    "source": document.source,
                    "content_type": "page_visual",
                     
                    "requires_rendering": True,
                },
            )

            blocks.append(image_block)

        return blocks