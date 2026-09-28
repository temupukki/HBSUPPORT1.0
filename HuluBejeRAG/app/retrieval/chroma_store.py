from __future__ import annotations

from typing import Any

import chromadb


class ChromaStore:
    """
    Persistent ChromaDB vector store for HuluBeje RAG.
    """

    COLLECTION_NAME = "hulubeje_knowledge"

    def __init__(
        self,
        persist_directory: str = "data/chroma",
    ) -> None:

        self.persist_directory = persist_directory

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            },
        )

    # =========================================================
    # COLLECTION INFORMATION
    # =========================================================

    @property
    def count(self) -> int:
        """
        Return the number of stored chunks.
        """

        return self.collection.count()

    # =========================================================
    # ADD CHUNKS
    # =========================================================

    def add_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:

        if not ids:
            return

        if not (
            len(ids)
            == len(texts)
            == len(embeddings)
            == len(metadatas)
        ):

            raise ValueError(
                "ids, texts, embeddings, and metadatas "
                "must contain the same number of items."
            )

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # =========================================================
    # GET CHUNKS
    # =========================================================

    def get(
        self,
        ids: list[str] | None = None,
        limit: int | None = None,
    ):

        return self.collection.get(
            ids=ids,
            limit=limit,
        )

    # =========================================================
    # QUERY
    # =========================================================

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ):

        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    # =========================================================
    # DELETE COLLECTION
    # =========================================================

    def clear(self) -> None:
        """
        Delete all records from the collection.
        """

        if self.collection.count() == 0:
            return

        ids = self.collection.get(
            include=[]
        )["ids"]

        if ids:
            self.collection.delete(
                ids=ids
            )