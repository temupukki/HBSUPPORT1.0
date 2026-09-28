from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import Chunker
from app.retrieval.embedder import BGEEmbedder
from app.retrieval.chroma_store import ChromaStore


KNOWLEDGE_DIR = "knowledge"
CHROMA_DIR = "data/chroma"


def main():

    print("=" * 70)
    print("HULUBEJE RAG — KNOWLEDGE BASE INGESTION")
    print("=" * 70)

    # =========================================================
    # 1. LOAD PDF DOCUMENTS
    # =========================================================

    print()
    print("[1/4] Loading PDF documents...")

    loader = DocumentLoader(
        KNOWLEDGE_DIR
    )

    documents = loader.load()

    print(
        f"Documents loaded: {len(documents)}"
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
        f"Total pages: {total_pages}"
    )

    # =========================================================
    # 2. CREATE CHUNKS
    # =========================================================

    print()
    print("[2/4] Creating chunks...")

    chunker = Chunker()

    chunks = chunker.chunk_documents(
        documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    if not chunks:
        raise RuntimeError(
            "No chunks were created."
        )

    # ---------------------------------------------------------
    # Chunk statistics
    # ---------------------------------------------------------

    token_counts = [
        chunk.token_count
        for chunk in chunks
    ]

    minimum = min(token_counts)
    maximum = max(token_counts)
    average = (
        sum(token_counts)
        / len(token_counts)
    )

    print()
    print("Chunk statistics:")
    print(
        f"  Minimum tokens : {minimum}"
    )
    print(
        f"  Maximum tokens : {maximum}"
    )
    print(
        f"  Average tokens : {average:.1f}"
    )

    # ---------------------------------------------------------
    # Hard validation
    # ---------------------------------------------------------

    if minimum < Chunker.MIN_TOKENS:

        raise RuntimeError(
            f"Chunk validation failed. "
            f"Minimum chunk is {minimum} tokens, "
            f"but minimum allowed is "
            f"{Chunker.MIN_TOKENS}."
        )

    if maximum > Chunker.MAX_TOKENS:

        raise RuntimeError(
            f"Chunk validation failed. "
            f"Maximum chunk is {maximum} tokens, "
            f"but maximum allowed is "
            f"{Chunker.MAX_TOKENS}."
        )

    print(
        "  Chunk validation: PASSED"
    )

    # =========================================================
    # 3. CREATE BGE EMBEDDINGS
    # =========================================================

    print()
    print("[3/4] Creating BGE-large embeddings...")

    embedder = BGEEmbedder()

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = embedder.embed_documents(
        texts,
        batch_size=16,
    )

    if len(embeddings) != len(chunks):

        raise RuntimeError(
            "Number of embeddings does not "
            "match number of chunks."
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
    # 4. STORE IN CHROMADB
    # =========================================================

    print()
    print("[4/4] Storing data in ChromaDB...")

    store = ChromaStore(
        persist_directory=CHROMA_DIR
    )

    ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    metadata = [
        chunk.metadata
        for chunk in chunks
    ]

    store.add_chunks(
        ids=ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadata,
    )

    print(
        f"Chunks stored: {store.count}"
    )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print()
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)

    print()
    print(f"Documents : {len(documents)}")
    print(f"Pages     : {total_pages}")
    print(f"Chunks    : {len(chunks)}")
    print(
        f"Embeddings: {len(embeddings)}"
    )
    print(
        f"Dimension : {embedder.dimension}"
    )
    print(
        f"Chroma    : {store.count}"
    )
    print(
        f"Location  : {CHROMA_DIR}"
    )

    print()
    print("Knowledge base is ready for retrieval.")


if __name__ == "__main__":
    main()