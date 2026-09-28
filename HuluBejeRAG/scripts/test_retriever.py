from app.retrieval.embedder import BGEEmbedder
from app.retrieval.chroma_store import ChromaStore
from app.retrieval.retriever import Retriever


def main():

    print("=" * 70)
    print("HULUBEJE RETRIEVER TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Initialize components
    # ---------------------------------------------------------

    print()
    print("Loading BGE-large...")

    embedder = BGEEmbedder()

    print("Connecting to ChromaDB...")

    store = ChromaStore(
        persist_directory="data/chroma"
    )

    print(
        f"Knowledge chunks: {store.count}"
    )

    if store.count == 0:

        raise RuntimeError(
            "ChromaDB is empty. "
            "Run the ingestion pipeline first."
        )

    retriever = Retriever(
        embedder=embedder,
        store=store,
    )

    # ---------------------------------------------------------
    # Test query
    # ---------------------------------------------------------

    query = "How long does standard delivery take?"

    print()
    print(f"QUERY:")
    print(query)

    print()
    print("Searching knowledge base...")

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"RETRIEVED CHUNKS: {len(results)}"
    )
    print("=" * 70)

    for index, result in enumerate(results):

        print()
        print(
            f"RESULT {index + 1}"
        )

        print("-" * 70)

        print(
            f"Chunk ID : {result.chunk_id}"
        )

        print(
            f"Distance : "
            f"{result.distance:.6f}"
        )

        print(
            f"Source   : "
            f"{result.metadata.get('source')}"
        )

        print(
            f"Page     : "
            f"{result.metadata.get('page')}"
        )

        print(
            f"Section  : "
            f"{result.metadata.get('section')}"
        )

        print(
            f"Subsection : "
            f"{result.metadata.get('subsection')}"
        )

        print()

        print(result.text)

    print()
    print("=" * 70)
    print("RETRIEVER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()