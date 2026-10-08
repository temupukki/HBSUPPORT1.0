from pathlib import Path

from app.ingestion.loader import DocumentLoader
from app.extraction.image_extractor import ImageExtractor
from app.extraction.diagram_extractor import DiagramExtractor


def print_separator(char="=", length=100):
    print()
    print(char * length)
    print()


def main():

    loader = DocumentLoader(
        "knowledge"
    )

    image_extractor = ImageExtractor(
        "artifacts/visuals"
    )

    diagram_extractor = DiagramExtractor()

    documents = loader.load()

    for document in documents:

        print_separator()

        print(
            "DOCUMENT:",
            document.source
        )

        print_separator()

        images = (
            image_extractor.extract_document_images(
                document
            )
        )

        print(
            "EXTRACTED IMAGES:",
            len(images)
        )

        for image in images:

            print_separator("-")

            print(
                "PAGE:",
                image["page_number"]
            )

            print(
                "IMAGE:",
                image["image_index"]
            )

            print(
                "SIZE:",
                image["width"],
                "x",
                image["height"]
            )

            print(
                "OUTPUT:",
                image["path"]
            )

            # ----------------------------------------------------------
            # Only analyze the workflow diagram for now.
            # ----------------------------------------------------------

            if (
                "Standard Operating Procedure"
                not in document.source
            ):
                continue

            if image["page_number"] != 7:
                continue

            analysis = (
                diagram_extractor.analyze(
                    image["path"]
                )
            )

            # ----------------------------------------------------------
            # SUMMARY
            # ----------------------------------------------------------

            print_separator()

            print(
                "DIAGRAM SUMMARY"
            )

            print_separator()

            print(
                "IMAGE SIZE:",
                analysis.image_width,
                "x",
                analysis.image_height
            )

            print(
                "OCR TEXT:",
                len(
                    analysis.ocr_text
                )
            )

            print(
                "TOTAL ELEMENTS:",
                len(
                    analysis.elements
                )
            )

            print(
                "RECTANGLES:",
                len(
                    analysis.rectangles
                )
            )

            print(
                "DIAMONDS:",
                len(
                    analysis.diamonds
                )
            )

            print(
                "ELLIPSES:",
                len(
                    analysis.ellipses
                )
            )

            print(
                "TEXT REGIONS:",
                len(
                    analysis.text_regions
                )
            )

            # ----------------------------------------------------------
            # ALL ELEMENTS
            # ----------------------------------------------------------

            print_separator()

            print(
                "ALL DETECTED ELEMENTS"
            )

            print_separator()

            for index, element in enumerate(
                analysis.elements,
                start=1,
            ):

                print(
                    f"[{index}]"
                )

                print(
                    "  ID:",
                    element.element_id
                )

                print(
                    "  TYPE:",
                    element.element_type
                )

                print(
                    "  BBOX:",
                    (
                        element.x,
                        element.y,
                        element.width,
                        element.height,
                    )
                )

                print(
                    "  CENTER:",
                    (
                        round(
                            element.center_x,
                            1,
                        ),
                        round(
                            element.center_y,
                            1,
                        ),
                    )
                )

                print(
                    "  TEXT:",
                    repr(
                        element.text
                    )
                )

                print(
                    "  CONFIDENCE:",
                    element.confidence
                )

                print(
                    "  METADATA:",
                    element.metadata
                )

                print()

            # ----------------------------------------------------------
            # OCR-BASED REGIONS ONLY
            # ----------------------------------------------------------

            print_separator()

            print(
                "TEXT REGIONS ONLY"
            )

            print_separator()

            for index, region in enumerate(
                analysis.text_regions,
                start=1,
            ):

                print(
                    f"[{index}] "
                    f"{region.element_id}"
                )

                print(
                    "  BBOX:",
                    (
                        region.x,
                        region.y,
                        region.width,
                        region.height,
                    )
                )

                print(
                    "  TEXT:",
                    repr(
                        region.text
                    )
                )

                print(
                    "  CONFIDENCE:",
                    region.confidence
                )

                print()

            print_separator()

            print(
                "DONE"
            )

            print_separator()


if __name__ == "__main__":
    main()