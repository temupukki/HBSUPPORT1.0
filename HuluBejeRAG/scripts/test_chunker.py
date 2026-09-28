from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import Chunker


def main():
    print("Loading documents...")

    loader = DocumentLoader("knowledge")
    documents = loader.load()

    print(f"Documents loaded: {len(documents)}")

    print()
    print("Chunking documents...")

    chunker = Chunker()
    chunks = chunker.chunk_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        print("No chunks were created.")
        return

    token_counts = [
        chunker._token_count(chunk.text)
        for chunk in chunks
    ]

    print()
    print("=" * 60)
    print("CHUNK STATISTICS")
    print("=" * 60)

    print(f"Minimum tokens : {min(token_counts)}")
    print(f"Maximum tokens : {max(token_counts)}")
    print(
        f"Average tokens : "
        f"{sum(token_counts) / len(token_counts):.1f}"
    )

    print(
        f"Chunks > 450  : "
        f"{sum(1 for x in token_counts if x > 450)}"
    )

    print(
        f"Chunks > 512  : "
        f"{sum(1 for x in token_counts if x > 512)}"
    )

    print()

    print("=" * 60)
    print("FIRST CHUNK")
    print("=" * 60)

    first = chunks[0]

    print(f"Chunk ID     : {first.chunk_id}")
    print(f"Document ID  : {first.document_id}")
    print(f"Token count  : {token_counts[0]}")
    print(f"Metadata     : {first.metadata}")

    print()
    print(first.text)

    print()

    print("=" * 60)
    print("LAST CHUNK")
    print("=" * 60)

    last = chunks[-1]

    print(f"Chunk ID     : {last.chunk_id}")
    print(f"Document ID  : {last.document_id}")
    print(f"Token count  : {token_counts[-1]}")
    print(f"Metadata     : {last.metadata}")

    print()
    print(last.text)


if __name__ == "__main__":
    main()