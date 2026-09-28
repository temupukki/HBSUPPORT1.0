from app.retrieval.chroma_store import ChromaStore


def main():

    print("=" * 60)
    print("CHROMADB STORE TEST")
    print("=" * 60)

    print()
    print("Creating ChromaDB store...")

    store = ChromaStore(
        persist_directory="data/chroma_test"
    )

    print(
        f"Collection: "
        f"{store.COLLECTION_NAME}"
    )

    print(
        f"Initial records: "
        f"{store.count}"
    )

    # ---------------------------------------------------------
    # Fake embeddings for storage testing
    #
    # We are NOT testing BGE here.
    # We are only testing ChromaDB storage.
    # ---------------------------------------------------------

    ids = [
        "test_chunk_001",
        "test_chunk_002",
        "test_chunk_003",
    ]

    texts = [
        "Standard delivery takes 3 to 7 business days.",
        "Express delivery takes 1 to 3 business days.",
        "Customers can return eligible products.",
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
    ]

    metadatas = [
        {
            "source": "shipping.pdf",
            "page": "1",
            "section": "Delivery",
        },
        {
            "source": "shipping.pdf",
            "page": "2",
            "section": "Express Delivery",
        },
        {
            "source": "returns.pdf",
            "page": "1",
            "section": "Returns",
        },
    ]

    # ---------------------------------------------------------
    # Insert
    # ---------------------------------------------------------

    print()
    print("Adding test chunks...")

    store.add_chunks(
        ids=ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        f"Records after insert: "
        f"{store.count}"
    )

    # ---------------------------------------------------------
    # Read
    # ---------------------------------------------------------

    print()
    print("Reading stored chunks...")

    result = store.get()

    print(
        f"Retrieved records: "
        f"{len(result['ids'])}"
    )

    for index, chunk_id in enumerate(
        result["ids"]
    ):

        print()
        print(f"ID       : {chunk_id}")
        print(
            f"Document : "
            f"{result['documents'][index]}"
        )
        print(
            f"Metadata : "
            f"{result['metadatas'][index]}"
        )

    # ---------------------------------------------------------
    # Query
    # ---------------------------------------------------------

    print()
    print("Testing vector query...")

    query_embedding = [
        0.1,
        0.2,
        0.3,
    ]

    results = store.query(
        query_embedding=query_embedding,
        top_k=2,
    )

    print(
        f"Results returned: "
        f"{len(results['ids'][0])}"
    )

    for index, chunk_id in enumerate(
        results["ids"][0]
    ):

        print()
        print(f"Result {index + 1}")
        print(f"ID       : {chunk_id}")
        print(
            f"Distance : "
            f"{results['distances'][0][index]}"
        )
        print(
            f"Document : "
            f"{results['documents'][0][index]}"
        )

    print()
    print("=" * 60)
    print("CHROMADB STORE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()