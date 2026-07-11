"""Phase 2 sanity check — run a handful of queries and SHOW the passages.

Usage:
    python agents/check_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.retrieval import get_retriever

TEST_QUERIES = [
    "Tomato Late Blight treatment",
    "Apple Scab management",
    "Corn Common Rust control",
    "Potato Early Blight symptoms and fungicide",
    "Grape Black Rot management",
    "Citrus greening Huanglongbing management",
]


def main() -> None:
    retriever = get_retriever()
    print("=" * 74)
    print("  RETRIEVAL SANITY CHECK")
    print("=" * 74)
    print(f"  Index documents (chunks): {retriever.document_count}")

    for q in TEST_QUERIES:
        print("\n" + "-" * 74)
        print(f"QUERY: {q}")
        print("-" * 74)
        passages = retriever.retrieve(q, k=3)
        if not passages:
            print("  (no passages retrieved)")
            continue
        for i, p in enumerate(passages, 1):
            snippet = " ".join(p.text.strip().split())
            if len(snippet) > 300:
                snippet = snippet[:300].rsplit(" ", 1)[0] + " …"
            print(f"\n  [{i}] score={p.score:.4f}  source={p.source}")
            print(f"      {snippet}")


if __name__ == "__main__":
    main()
