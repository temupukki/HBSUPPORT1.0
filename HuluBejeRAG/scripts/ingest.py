from __future__ import annotations

from typing import Any

from app.ingestion.loader import DocumentLoader
from app.analysis.region_detector import RegionDetector
from app.extraction.workflow_extractor import WorkflowExtractor

from app.chunking.document_chunker import (
    DocumentChunker,
)

from app.retrieval.embedder import BGEEmbedder
from app.retrieval.chroma_store import ChromaStore


KNOWLEDGE_DIR = "knowledge"
CHROMA_DIR = "data/chroma"


def print_separator() -> None:

    print(
        "=" * 70
    )


def print_chunk_statistics(
    chunks: list[Any],
) -> None:

    if not chunks:
        return

    statistics = (
        DocumentChunker.summarize(
            chunks
        )
    )

    print()
    print(
        "Chunk statistics:"
    )

    print(
        f"  Total chunks       : "
        f"{statistics['total_chunks']}"
    )

    print(
        f"  Workflow chunks    : "
        f"{statistics['workflow_chunks']}"
    )

    print(
        f"  Total characters   : "
        f"{statistics['total_characters']}"
    )

    print(
        f"  Total words        : "
        f"{statistics['total_words']}"
    )

    print(
        f"  Total tokens       : "
        f"{statistics['total_tokens']}"
    )

    print()
    print(
        "Chunks by type:"
    )

    for content_type, count in sorted(
        statistics[
            "by_type"
        ].items()
    ):

        print(
            f"  {content_type:<24}: "
            f"{count}"
        )


def print_document_statistics(
    document: Any,
    chunks: list[Any],
) -> None:

    print()
    print(
        f"Document: "
        f"{document.source}"
    )

    print(
        f"  Pages: "
        f"{document.page_count}"
    )

    print(
        f"  Chunks: "
        f"{len(chunks)}"
    )

    by_type: dict[
        str,
        int,
    ] = {}

    for chunk in chunks:

        content_type = (
            chunk.content_type
        )

        by_type[
            content_type
        ] = (
            by_type.get(
                content_type,
                0,
            )
            + 1
        )

    for content_type, count in sorted(
        by_type.items()
    ):

        print(
            f"    {content_type:<22}: "
            f"{count}"
        )


def print_chunk_token_distribution(
    chunks: list[Any],
) -> None:

    text_chunks = [
        chunk
        for chunk in chunks
        if chunk.content_type == "text"
    ]

    if not text_chunks:
        return

    print()
    print(
        "Normal-text token distribution:"
    )

    counts = []

    for chunk in text_chunks:

        value = chunk.metadata.get(
            "token_count"
        )

        if value is not None:

            try:
                counts.append(
                    int(value)
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    if not counts:
        return

    print(
        f"  Minimum : {min(counts)}"
    )

    print(
        f"  Maximum : {max(counts)}"
    )

    print(
        f"  Average : "
        f"{sum(counts) / len(counts):.1f}"
    )


def print_sample_chunks(
    chunks: list[Any],
    limit: int = 10,
) -> None:

    print()
    print(
        "-" * 70
    )

    print(
        "SAMPLE GENERATED CHUNKS"
    )

    print(
        "-" * 70
    )

    for index, chunk in enumerate(
        chunks[:limit],
        start=1,
    ):

        print()
        print(
            f"[{index}] "
            f"{chunk.chunk_id}"
        )

        print(
            f"Type       : "
            f"{chunk.content_type}"
        )

        print(
            f"Pages      : "
            f"{chunk.page_numbers}"
        )

        print(
            f"Tokens     : "
            f"{chunk.metadata.get('token_count', 'N/A')}"
        )

        if chunk.section:

            print(
                f"Section    : "
                f"{chunk.section}"
            )

        if chunk.workflow_name:

            print(
                f"Workflow   : "
                f"{chunk.workflow_name}"
            )

        if chunk.step_ids:

            print(
                f"Steps      : "
                f"{len(chunk.step_ids)}"
            )

        if chunk.decision_ids:

            print(
                f"Decisions  : "
                f"{len(chunk.decision_ids)}"
            )

        print(
            f"Characters : "
            f"{chunk.character_count}"
        )

        print()
        print(
            chunk.text[:500]
        )

        if len(
            chunk.text
        ) > 500:

            print(
                "..."
            )


def detect_regions(
    detector: RegionDetector,
    document: Any,
) -> list[Any]:

    if hasattr(
        detector,
        "detect",
    ):

        result = detector.detect(
            document
        )

    elif hasattr(
        detector,
        "process",
    ):

        result = detector.process(
            document
        )

    elif hasattr(
        detector,
        "detect_document",
    ):

        result = (
            detector.detect_document(
                document
            )
        )

    else:

        raise AttributeError(
            "RegionDetector does not expose "
            "detect(), process(), or "
            "detect_document()."
        )

    if result is None:
        return []

    if isinstance(
        result,
        dict,
    ):

        regions = result.get(
            "regions"
        )

        if regions is not None:

            return list(
                regions
            )

    return list(
        result
    )


def build_workflow_provider():

    workflow_extractor = (
        WorkflowExtractor()
    )

    def provider(
        document: Any,
        region: Any,
    ) -> list[Any]:

        try:

            return (
                DocumentChunker
                .call_workflow_extractor(
                    extractor=(
                        workflow_extractor
                    ),
                    region=region,
                    document_id=(
                        document.document_id
                    ),
                )
            )

        except Exception as exc:

            print(
                "[Ingestion] Workflow "
                f"extraction skipped on "
                f"page "
                f"{getattr(region, 'page_number', '?')}: "
                f"{exc}"
            )

            return []

    return provider


def main():

    print_separator()

    print(
        "HULUBEJE RAG — "
        "KNOWLEDGE BASE INGESTION"
    )

    print_separator()

    # =========================================================
    # 1. LOAD
    # =========================================================

    print()
    print(
        "[1/5] Loading PDF documents..."
    )

    loader = DocumentLoader(
        KNOWLEDGE_DIR
    )

    documents = loader.load()

    print(
        f"Documents loaded: "
        f"{len(documents)}"
    )

    if not documents:

        raise RuntimeError(
            "No PDF documents were found "
            "in the knowledge directory."
        )

    total_pages = sum(
        document.page_count
        for document in documents
    )

    print(
        f"Total pages: "
        f"{total_pages}"
    )

    # =========================================================
    # 2. REGIONS
    # =========================================================

    print()
    print(
        "[2/5] Detecting document regions..."
    )

    region_detector = (
        RegionDetector()
    )

    document_regions: dict[
        str,
        list[Any],
    ] = {}

    total_regions = 0

    for document in documents:

        regions = detect_regions(
            region_detector,
            document,
        )

        document_regions[
            document.document_id
        ] = regions

        total_regions += len(
            regions
        )

        print()
        print(
            f"Document: "
            f"{document.source}"
        )

        print(
            f"  Regions: "
            f"{len(regions)}"
        )

        region_types: dict[
            str,
            int,
        ] = {}

        for region in regions:

            region_type = str(
                getattr(
                    region,
                    "region_type",
                    "unknown",
                )
            )

            region_types[
                region_type
            ] = (
                region_types.get(
                    region_type,
                    0,
                )
                + 1
            )

        for region_type, count in sorted(
            region_types.items()
        ):

            print(
                f"    {region_type:<24}: "
                f"{count}"
            )

    print()
    print(
        f"Total regions detected: "
        f"{total_regions}"
    )

    # =========================================================
    # 3. CHUNKING
    # =========================================================

    print()
    print(
        "[3/5] Creating knowledge chunks..."
    )

    document_chunker = (
        DocumentChunker()
    )

    workflow_provider = (
        build_workflow_provider()
    )

    all_chunks: list[Any] = []

    for document in documents:

        regions = (
            document_regions.get(
                document.document_id,
                [],
            )
        )

        chunks = (
            document_chunker.chunk_document(
                document=document,
                regions=regions,
                workflow_provider=(
                    workflow_provider
                ),
            )
        )

        all_chunks.extend(
            chunks
        )

        print_document_statistics(
            document,
            chunks,
        )

    print()
    print(
        f"Knowledge chunks created: "
        f"{len(all_chunks)}"
    )

    if not all_chunks:

        raise RuntimeError(
            "No knowledge chunks were created."
        )

    print_chunk_statistics(
        all_chunks
    )

    print_chunk_token_distribution(
        all_chunks
    )

    print_sample_chunks(
        all_chunks,
        limit=10,
    )

    # =========================================================
    # 4. EMBEDDINGS
    # =========================================================

    print()
    print(
        "[4/5] Creating BGE-large embeddings..."
    )

    embedder = BGEEmbedder()

    texts = [
        chunk.text
        for chunk in all_chunks
    ]

    embeddings = (
        embedder.embed_documents(
            texts,
            batch_size=16,
        )
    )

    if len(embeddings) != len(
        all_chunks
    ):

        raise RuntimeError(
            "Number of embeddings does not "
            "match number of knowledge chunks."
        )

    print(
        f"Embeddings created: "
        f"{len(embeddings)}"
    )

    print(
        f"Embedding dimension: "
        f"{embedder.dimension}"
    )

    # =========================================================
    # 5. CHROMA
    # =========================================================

    print()
    print(
        "[5/5] Storing data in ChromaDB..."
    )

    store = ChromaStore(
        persist_directory=CHROMA_DIR
    )

    ids = [
        chunk.chunk_id
        for chunk in all_chunks
    ]

    metadata = [
        chunk.metadata
        for chunk in all_chunks
    ]

    store.add_chunks(
        ids=ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadata,
    )

    print(
        f"Chunks stored: "
        f"{store.count}"
    )

    # =========================================================
    # COMPLETE
    # =========================================================

    print()
    print_separator()

    print(
        "INGESTION COMPLETE"
    )

    print_separator()

    print()

    print(
        f"Documents : "
        f"{len(documents)}"
    )

    print(
        f"Pages     : "
        f"{total_pages}"
    )

    print(
        f"Regions   : "
        f"{total_regions}"
    )

    print(
        f"Chunks    : "
        f"{len(all_chunks)}"
    )

    print(
        f"Embeddings: "
        f"{len(embeddings)}"
    )

    print(
        f"Dimension : "
        f"{embedder.dimension}"
    )

    print(
        f"Chroma    : "
        f"{store.count}"
    )

    print(
        f"Location  : "
        f"{CHROMA_DIR}"
    )

    print()

    print(
        "Knowledge base is ready "
        "for retrieval."
    )


if __name__ == "__main__":
    main()