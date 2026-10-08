from pathlib import Path
from typing import Any

import pdfplumber


class TableExtractor:
    """
    Extracts structured tables from PDF pages.

    This class is responsible ONLY for table extraction.

    It does not:
    - understand workflows
    - interpret business meaning
    - create chunks
    - generate embeddings
    - call an LLM
    """

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF does not exist: {self.pdf_path}"
            )

    def extract_page_tables(
        self,
        page_number: int,
    ) -> list[dict[str, Any]]:
        """
        Extract all detected tables from one PDF page.

        page_number is 1-based.
        """

        tables: list[dict[str, Any]] = []

        with pdfplumber.open(self.pdf_path) as pdf:

            if page_number < 1 or page_number > len(pdf.pages):
                raise ValueError(
                    f"Invalid page number {page_number}. "
                    f"PDF has {len(pdf.pages)} pages."
                )

            page = pdf.pages[page_number - 1]

            extracted_tables = page.extract_tables()

            for table_index, table in enumerate(
                extracted_tables
            ):

                cleaned_rows = self._clean_table(
                    table
                )

                if not cleaned_rows:
                    continue

                tables.append(
                    {
                        "table_index": table_index,
                        "page_number": page_number,
                        "rows": cleaned_rows,
                    }
                )

        return tables

    @staticmethod
    def _clean_table(
        table: list[list[Any] | None],
    ) -> list[list[str]]:

        cleaned_rows: list[list[str]] = []

        for row in table:

            if not row:
                continue

            cleaned_row: list[str] = []

            for cell in row:

                if cell is None:
                    cell = ""

                cell = str(cell)

                cell = (
                    cell
                    .replace("\r\n", " ")
                    .replace("\n", " ")
                    .strip()
                )

                cleaned_row.append(cell)

            # Ignore completely empty rows
            if any(cell for cell in cleaned_row):
                cleaned_rows.append(
                    cleaned_row
                )

        return cleaned_rows