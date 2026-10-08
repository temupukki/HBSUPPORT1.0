from dataclasses import dataclass
from typing import Any


@dataclass
class TextWord:
    """
    A single positioned word extracted from a PDF page.
    """

    text: str

    x0: float
    x1: float

    top: float
    bottom: float


class LayoutTableExtractor:
    """
    Detects table-like structures from positioned PDF text.

    This is different from normal PDF table extraction.

    Some PDFs visually contain tables but internally store
    them only as positioned text objects.

    Example:

        11.1    Motorcycles parking
        11.2    Charging stations
        11.3    Personal lockers

    There may be no actual PDF table/cell objects.

    This class reconstructs the structure from:
    - X coordinates
    - Y coordinates
    - repeated row positions
    """

    def __init__(
        self,
        row_tolerance: float = 3.0,
    ):
        self.row_tolerance = row_tolerance

    def group_words_into_rows(
        self,
        words: list[dict[str, Any]],
    ) -> list[list[TextWord]]:
        """
        Group positioned PDF words into visual rows.
        """

        text_words = [
            TextWord(
                text=str(word["text"]).strip(),
                x0=float(word["x0"]),
                x1=float(word["x1"]),
                top=float(word["top"]),
                bottom=float(word["bottom"]),
            )
            for word in words
            if str(word.get("text", "")).strip()
        ]

        # PDF words normally arrive in reading order,
        # but we explicitly sort by vertical position.
        text_words.sort(
            key=lambda word: (
                word.top,
                word.x0,
            )
        )

        rows: list[list[TextWord]] = []

        for word in text_words:

            if not rows:
                rows.append([word])
                continue

            current_row = rows[-1]

            reference_top = current_row[0].top

            if abs(
                word.top - reference_top
            ) <= self.row_tolerance:

                current_row.append(word)

            else:
                rows.append([word])

        # Ensure words inside each row are left-to-right.
        for row in rows:
            row.sort(
                key=lambda word: word.x0
            )

        return rows

    @staticmethod
    def row_to_text(
        row: list[TextWord],
    ) -> str:
        """
        Convert one visual row back into readable text.
        """

        return " ".join(
            word.text
            for word in row
        )

    def extract_rows(
        self,
        words: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Return positioned rows.

        This is the first layout-normalization step.

        We deliberately do not guess columns yet.
        """

        rows = self.group_words_into_rows(
            words
        )

        result: list[dict[str, Any]] = []

        for index, row in enumerate(rows):

            if not row:
                continue

            result.append(
                {
                    "row_index": index,
                    "top": row[0].top,
                    "bottom": max(
                        word.bottom
                        for word in row
                    ),
                    "x_start": min(
                        word.x0
                        for word in row
                    ),
                    "x_end": max(
                        word.x1
                        for word in row
                    ),
                    "text": self.row_to_text(row),
                    "words": [
                        {
                            "text": word.text,
                            "x0": word.x0,
                            "x1": word.x1,
                            "top": word.top,
                            "bottom": word.bottom,
                        }
                        for word in row
                    ],
                }
            )

        return result