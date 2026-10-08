from pathlib import Path
from typing import Any

import pdfplumber

from app.models.document import Document


class ImageExtractor:
    """
    Extracts embedded PDF images as actual image files.

    Responsibilities:
    - locate embedded images
    - crop them from their PDF page
    - preserve page/image metadata
    - save them as PNG files

    This component does NOT perform OCR.

    Pipeline:

        PDF
         ↓
        embedded image
         ↓
        ImageExtractor
         ↓
        PNG
         ↓
        OCR / Vision
    """

    def __init__(
        self,
        output_dir: str = "artifacts/visuals",
    ):
        self.output_dir = Path(
            output_dir
        )

    def extract_document_images(
        self,
        document: Document,
    ) -> list[dict[str, Any]]:
        """
        Extract all significant visual images
        from a document.
        """

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        extracted: list[
            dict[str, Any]
        ] = []

        for page in document.pages:

            image_objects = (
                page.metadata.get(
                    "significant_image_objects",
                    [],
                )
            )

            if not image_objects:
                continue

            page_images = (
                self._extract_page_images(
                    document=document,
                    page=page,
                    image_objects=image_objects,
                )
            )

            extracted.extend(
                page_images
            )

        return extracted

    def _extract_page_images(
        self,
        document: Document,
        page,
        image_objects: list[
            dict[str, Any]
        ],
    ) -> list[dict[str, Any]]:
        """
        Extract images from one PDF page.
        """

        results: list[
            dict[str, Any]
        ] = []

        pdf_path = Path(
            document.metadata[
                "file_path"
            ]
        )

        with pdfplumber.open(
            pdf_path
        ) as pdf:

            pdf_page = pdf.pages[
                page.page_number - 1
            ]

            for image in image_objects:

                result = (
                    self._extract_single_image(
                        document=document,
                        page=page,
                        pdf_page=pdf_page,
                        image=image,
                    )
                )

                if result is not None:
                    results.append(
                        result
                    )

        return results

    def _extract_single_image(
        self,
        document: Document,
        page,
        pdf_page,
        image: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Crop one embedded image from the PDF page
        and save it as PNG.
        """

        image_index = int(
            image.get(
                "index",
                0,
            )
        )

        x0 = float(
            image.get(
                "x0",
                0.0,
            )
        )

        top = float(
            image.get(
                "top",
                0.0,
            )
        )

        x1 = float(
            image.get(
                "x1",
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
            return None

        if bottom <= top:
            return None

        # ---------------------------------------------------------
        # Crop the image region from the PDF page.
        #
        # pdfplumber uses:
        #
        #     (x0, top, x1, bottom)
        #
        # for crop coordinates.
        # ---------------------------------------------------------

        try:

            cropped = pdf_page.crop(
                (
                    x0,
                    top,
                    x1,
                    bottom,
                )
            )

            page_image = cropped.to_image(
                resolution=200
            )

        except Exception as exc:

            print(
                f"Failed to render image "
                f"on page {page.page_number}: "
                f"{exc}"
            )

            return None

        filename = (
            f"{document.document_id}"
            f"_page_{page.page_number}"
            f"_image_{image_index}.png"
        )

        output_path = (
            self.output_dir
            / filename
        )

        try:

            page_image.save(
                output_path,
                format="PNG",
            )

        except Exception as exc:

            print(
                f"Failed to save image "
                f"{output_path}: {exc}"
            )

            return None

        return {
            "document_id": (
                document.document_id
            ),

            "document_source": (
                document.source
            ),

            "page_number": (
                page.page_number
            ),

            "image_index": (
                image_index
            ),

            "x0": x0,

            "top": top,

            "x1": x1,

            "bottom": bottom,

            "width": float(
                image.get(
                    "width",
                    x1 - x0,
                )
            ),

            "height": float(
                image.get(
                    "height",
                    bottom - top,
                )
            ),

            "path": str(
                output_path
            ),

            "source": (
                "embedded_pdf_image"
            ),
        }