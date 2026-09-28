from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    """
    BGE-large embedding service for HuluBeje RAG.

    Model:
        BAAI/bge-large-en-v1.5

    Embedding dimension:
        1024

    Embeddings are normalized so cosine similarity
    can be used effectively during retrieval.
    """

    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(
        self,
        device: str | None = None,
    ) -> None:

        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device=device,
        )

    # =========================================================
    # DOCUMENT EMBEDDINGS
    # =========================================================

    def embed_documents(
        self,
        texts: Iterable[str],
        batch_size: int = 16,
    ) -> list[list[float]]:

        texts = list(texts)

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    # =========================================================
    # QUERY EMBEDDING
    # =========================================================

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    # =========================================================
    # EMBEDDING DIMENSION
    # =========================================================

    @property
    def dimension(self) -> int:

        return self.model.get_sentence_embedding_dimension()

    # =========================================================
    # VALIDATION
    # =========================================================

    def validate_embedding(
        self,
        embedding: list[float],
    ) -> None:

        if len(embedding) != 1024:

            raise ValueError(
                "Unexpected embedding dimension: "
                f"{len(embedding)}. "
                "Expected 1024."
            )

        if not np.all(
            np.isfinite(embedding)
        ):

            raise ValueError(
                "Embedding contains invalid "
                "numeric values."
            )