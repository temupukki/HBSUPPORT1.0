from app.retrieval.embedder import BGEEmbedder


def main():

    print("=" * 60)
    print("BGE-LARGE EMBEDDER TEST")
    print("=" * 60)

    print()
    print("Loading model...")

    embedder = BGEEmbedder()

    print()
    print(f"Model     : {embedder.MODEL_NAME}")
    print(f"Dimension : {embedder.dimension}")

    # ---------------------------------------------------------
    # Test document embedding
    # ---------------------------------------------------------

    documents = [
        "Standard delivery takes 3 to 7 business days.",
        "Express delivery takes 1 to 3 business days.",
        "Customers can return eligible products according to "
        "the HuluBeje return policy.",
    ]

    print()
    print("Embedding documents...")

    document_embeddings = embedder.embed_documents(
        documents
    )

    print(
        f"Documents embedded : "
        f"{len(document_embeddings)}"
    )

    for index, embedding in enumerate(
        document_embeddings
    ):

        print(
            f"Document {index + 1}: "
            f"{len(embedding)} dimensions"
        )

        embedder.validate_embedding(
            embedding
        )

    # ---------------------------------------------------------
    # Test query embedding
    # ---------------------------------------------------------

    query = "How long does delivery take?"

    print()
    print(f"Query: {query}")

    query_embedding = embedder.embed_query(
        query
    )

    print(
        f"Query embedding dimension: "
        f"{len(query_embedding)}"
    )

    embedder.validate_embedding(
        query_embedding
    )

    # ---------------------------------------------------------
    # Check normalization
    # ---------------------------------------------------------

    magnitude = sum(
        value * value
        for value in query_embedding
    ) ** 0.5

    print()
    print(
        f"Query vector magnitude: "
        f"{magnitude:.6f}"
    )

    print()
    print("=" * 60)
    print("EMBEDDER TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()