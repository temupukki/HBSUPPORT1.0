from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.retrieval.embedder import BGEEmbedder
from app.retrieval.chroma_store import ChromaStore


@dataclass
class RetrievedChunk:
    """
    One chunk returned by vector search.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class Retriever:
    """
    Retrieves the most relevant knowledge chunks
    from ChromaDB using BGE-large embeddings.
    """

    DEFAULT_TOP_K = 5

    def __init__(
        self,
        embedder: BGEEmbedder,
        store: ChromaStore,
    ) -> None:

        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RetrievedChunk]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = self.embedder.embed_query(
            query
        )

        results = self.store.query(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        ids = results.get("ids", [[]])[0]

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        retrieved: list[RetrievedChunk] = []

        for index, chunk_id in enumerate(ids):

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=documents[index],
                    metadata=metadatas[index],
                    distance=distances[index],
                )
            )

        return retrieved[:top_k]