from app.llm.client import LLMClient
from app.llm.query_generator import QueryGenerator
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.chroma_store import ChromaStore
from app.retrieval.embedder import BGEEmbedder
from app.retrieval.retriever import Retriever


def main():

    print("=" * 70)
    print("HULUBEJE RAG PIPELINE TEST")
    print("=" * 70)

    # -------------------------------------------------
    # LLM 1
    # -------------------------------------------------

    print()
    print("[1] Creating LLM client...")

    client = LLMClient()

    query_generator = QueryGenerator(
        client=client
    )

    # -------------------------------------------------
    # BGE
    # -------------------------------------------------

    print()
    print("[2] Loading BGE-large...")

    embedder = BGEEmbedder()

    # -------------------------------------------------
    # Chroma
    # -------------------------------------------------

    print()
    print("[3] Connecting to ChromaDB...")

    store = ChromaStore(
        persist_directory="data/chroma"
    )

    print(
        f"Knowledge chunks: {store.count}"
    )

    if store.count == 0:
        raise RuntimeError(
            "ChromaDB is empty. "
            "Run ingestion first."
        )

    # -------------------------------------------------
    # Retriever
    # -------------------------------------------------

    retriever = Retriever(
        embedder=embedder,
        store=store,
    )

    # -------------------------------------------------
    # Context Builder
    # -------------------------------------------------

    context_builder = ContextBuilder()

    # -------------------------------------------------
    # Pipeline
    # -------------------------------------------------

    pipeline = RAGPipeline(
        query_generator=query_generator,
        retriever=retriever,
        context_builder=context_builder,
    )

    # -------------------------------------------------
    # Test question
    # -------------------------------------------------

    user_message = (
        "How long does standard delivery take?"
    )

    print()
    print("=" * 70)
    print("CUSTOMER MESSAGE")
    print("=" * 70)

    print(user_message)

    # -------------------------------------------------
    # Run pipeline
    # -------------------------------------------------

    result = pipeline.run(
        user_message=user_message,
        top_k=3,
    )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print()
    print("=" * 70)
    print("ORIGINAL LANGUAGE")
    print("=" * 70)

    print(result.original_language)

    print()
    print("=" * 70)
    print("RETRIEVAL QUERY")
    print("=" * 70)

    print(result.retrieval_query)

    print()
    print("=" * 70)
    print("RETRIEVED CHUNKS")
    print("=" * 70)

    print(
        f"Count: {len(result.chunks)}"
    )

    for index, chunk in enumerate(
        result.chunks,
        start=1,
    ):

        print()
        print(f"RESULT {index}")
        print("-" * 70)

        print(
            f"Distance: "
            f"{chunk.distance:.6f}"
        )

        print(
            f"Source: "
            f"{chunk.metadata.get('source')}"
        )

        print(
            f"Page: "
            f"{chunk.metadata.get('page')}"
        )

        print(
            f"Section: "
            f"{chunk.metadata.get('section')}"
        )

    print()
    print("=" * 70)
    print("FINAL CONTEXT")
    print("=" * 70)

    print(result.context)

    print()
    print("=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()