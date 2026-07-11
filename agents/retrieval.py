"""Retrieval wrapper — reuses the RAG project's OWN embeddings + FAISS index.

We use only the RAG project's retrieval capability (its bge-small embeddings
and the FAISS index built by rag/rag/index_corpus.py). We do NOT touch its
frontend/API. The public surface here is deliberately tiny:

    retriever = get_retriever()
    passages = retriever.retrieve("Tomato Late Blight treatment", k=5)

Each Passage carries the chunk text, similarity score, and source document so
the reasoning agent can cite evidence and the coverage report can tell which
of the 38 classes have retrievable documents.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from agents import config

# Make the RAG project's `app` package importable without disturbing its layout.
if str(config.RAG_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(config.RAG_PROJECT_ROOT))


@dataclass
class Passage:
    """One retrieved evidence chunk.

    `score` is cosine similarity in [-1, 1] (higher = more relevant), derived
    from the store's raw L2 distance over normalized embeddings. `distance` is
    that raw L2 distance (lower = more relevant), kept for transparency.
    """

    text: str
    score: float         # cosine similarity, higher is better
    source: str          # source document filename / path
    doc_id: str          # chunk id from the vector store
    metadata: dict
    distance: float = 0.0  # raw L2 distance, lower is better


class Retriever:
    """Thin adapter over the RAG project's FAISS vector store."""

    def __init__(self, index_dir: Path | None = None):
        from app.config.settings import get_settings
        from app.embeddings.factory import get_embeddings_model
        from app.vectorstore.faiss_store import FAISSVectorStore

        self._index_dir = Path(index_dir or config.FAISS_INDEX_DIR)
        settings = get_settings()
        # Reuse the RAG project's configured embedding model (bge-small) so the
        # query embeddings match what was indexed.
        embeddings = get_embeddings_model(settings)
        # Point directly at our built index (absolute path), independent of the
        # RAG project's cwd-relative default. The store auto-loads on init.
        self._store = FAISSVectorStore(
            embedding_function=embeddings,
            persist_directory=str(self._index_dir),
        )

    @property
    def document_count(self) -> int:
        return self._store.get_document_count()

    def retrieve(self, query: str, k: int | None = None) -> list[Passage]:
        """Return the top-k evidence passages for a query."""
        k = k or config.RETRIEVAL_TOP_K
        results = self._store.similarity_search_with_score(query, k=k)
        passages: list[Passage] = []
        for doc, distance in results:
            meta = dict(doc.metadata or {})
            source = (
                meta.get("source")
                or meta.get("filename")
                or meta.get("file_name")
                or "unknown"
            )
            # Normalized embeddings -> cosine sim = 1 - (L2_distance^2)/2.
            dist = float(distance)
            similarity = max(-1.0, min(1.0, 1.0 - (dist * dist) / 2.0))
            passages.append(
                Passage(
                    text=doc.page_content,
                    score=similarity,
                    source=str(source),
                    doc_id=str(meta.get("id", meta.get("chunk_id", ""))),
                    metadata=meta,
                    distance=dist,
                )
            )
        return passages


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Process-wide cached retriever (loads the index once)."""
    return Retriever()
