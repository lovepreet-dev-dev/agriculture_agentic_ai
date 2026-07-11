"""Phase 2 coverage check — which of the 38 classes have retrievable docs?

For each class we build a query from its readable name, retrieve top-k passages,
and record how many pass a similarity-score threshold. Writes coverage_report.json
and prints a covered / uncovered summary.

Usage:
    python agents/coverage_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import config
from agents.retrieval import get_retriever
from agents.vision_agent import _clean_class_name, _load_class_names

# A retrieved passage counts as "covering" a class if its cosine similarity is
# at least this. On-topic ag passages score ~0.83-0.95; off-topic / absent
# diseases fall below this band.
COVERAGE_SCORE_THRESHOLD = 0.75
TOP_K = 5


def main() -> None:
    class_names = _load_class_names()
    retriever = get_retriever()

    print("=" * 74)
    print("  CLASS COVERAGE CHECK (38 classes)")
    print("=" * 74)
    print(f"  Index chunks: {retriever.document_count}")
    print(f"  Score threshold for 'covered': {COVERAGE_SCORE_THRESHOLD}")

    per_class = []
    covered, uncovered = [], []

    for cn in class_names:
        readable = _clean_class_name(cn)
        # Skip disease query for healthy classes — query the crop's general health.
        query = readable
        passages = retriever.retrieve(query, k=TOP_K)
        strong = [p for p in passages if p.score >= COVERAGE_SCORE_THRESHOLD]
        best = max((p.score for p in passages), default=0.0)
        entry = {
            "class": cn,
            "query": query,
            "num_hits": len(passages),
            "num_strong_hits": len(strong),
            "best_score": round(best, 4),
            "top_sources": list(dict.fromkeys(p.source for p in strong))[:3],
        }
        per_class.append(entry)
        if strong:
            covered.append(cn)
        else:
            uncovered.append(cn)

    report = {
        "total_classes": len(class_names),
        "covered_classes": len(covered),
        "uncovered_classes": len(uncovered),
        "coverage_ratio": round(len(covered) / max(1, len(class_names)), 4),
        "score_threshold": COVERAGE_SCORE_THRESHOLD,
        "top_k": TOP_K,
        "covered": covered,
        "uncovered": uncovered,
        "per_class": per_class,
    }

    config.COVERAGE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.COVERAGE_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\n  Covered:   {len(covered)}/{len(class_names)} "
          f"({report['coverage_ratio']:.1%})")
    print(f"  Uncovered: {len(uncovered)}")
    print("\n  COVERED classes (best score shown):")
    for e in sorted(per_class, key=lambda x: -x["best_score"]):
        if e["class"] in covered:
            print(f"    [{e['best_score']:.3f}] {e['class']}")
    print("\n  UNCOVERED classes (best score below threshold):")
    for e in sorted(per_class, key=lambda x: -x["best_score"]):
        if e["class"] in uncovered:
            print(f"    [{e['best_score']:.3f}] {e['class']}")
    print(f"\n  Report written to: {config.COVERAGE_REPORT_PATH}")


if __name__ == "__main__":
    main()
