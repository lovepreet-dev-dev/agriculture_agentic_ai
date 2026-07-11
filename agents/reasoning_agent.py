"""Reasoning agent — turns retrieved evidence into a grounded recommendation.

Runs only on the MODERATE (retrieve_and_reason) tier. Given the vision
prediction plus passages from the real retriever, it asks the LLM to explain
the diagnosis and recommend a treatment *grounded in the retrieved text*, with
citations back to source documents.

Graceful degradation:
  - No passages retrieved            -> signal escalation (no evidence to reason on).
  - No LLM key configured            -> return an extractive evidence summary
                                        (deterministic, no network) so the whole
                                        pipeline still runs end-to-end offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents import config
from agents.retrieval import Passage
from agents.vision_agent import _clean_class_name


SYSTEM_PROMPT = (
    "You are a plant pathology assistant. You are given a tentative diagnosis "
    "from an image classifier and a set of retrieved passages from scientific "
    "and agricultural-extension documents. Using ONLY the retrieved passages, "
    "explain whether they support the diagnosis and give a concise, practical "
    "management/treatment recommendation. Cite passages as [1], [2], etc. If "
    "the passages do not actually cover this disease, say so explicitly and "
    "recommend consulting a human expert. Never invent chemical names, "
    "dosages, or facts that are not in the passages."
)


@dataclass
class ReasoningResult:
    """Output of the reasoning agent."""

    recommendation: str
    used_llm: bool
    escalate: bool = False
    doc_ids: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    evidence_summary: str = ""


def _format_evidence(passages: list[Passage]) -> str:
    """Number the passages for citation in the prompt."""
    lines = []
    for i, p in enumerate(passages, 1):
        snippet = p.text.strip().replace("\n", " ")
        lines.append(f"[{i}] (source: {p.source}, score={p.score:.3f})\n{snippet}")
    return "\n\n".join(lines)


def _extractive_summary(readable: str, passages: list[Passage]) -> str:
    """Deterministic fallback when no LLM is configured.

    Presents the top evidence snippets verbatim with citations, so the output
    is still grounded and useful without a network call.
    """
    header = (
        f"Evidence-grounded summary for '{readable}' "
        f"(LLM not configured — showing top retrieved passages):"
    )
    bullets = []
    for i, p in enumerate(passages[:3], 1):
        snippet = " ".join(p.text.strip().split())
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + " …"
        bullets.append(f"  [{i}] ({p.source}) {snippet}")
    footer = (
        "Set OPENROUTER_API_KEY in .env to enable full LLM reasoning over this "
        "evidence."
    )
    return "\n".join([header, *bullets, footer])


class ReasoningAgent:
    """Grounds a diagnosis in retrieved evidence via the LLM."""

    def reason(
        self,
        label: str,
        confidence: float,
        passages: list[Passage],
    ) -> ReasoningResult:
        readable = _clean_class_name(label)

        # No evidence -> cannot ground a recommendation -> escalate.
        if not passages:
            return ReasoningResult(
                recommendation=(
                    f"No supporting documents were retrieved for '{readable}'. "
                    f"Cannot produce a grounded recommendation — escalating to a "
                    f"human expert."
                ),
                used_llm=False,
                escalate=True,
            )

        doc_ids = [p.doc_id for p in passages]
        sources = list(dict.fromkeys(p.source for p in passages))  # unique, ordered
        evidence = _format_evidence(passages)

        # No LLM key -> deterministic extractive fallback (still grounded).
        if not config.llm_is_configured():
            return ReasoningResult(
                recommendation=_extractive_summary(readable, passages),
                used_llm=False,
                doc_ids=doc_ids,
                sources=sources,
                evidence_summary=evidence,
            )

        from agents.llm import call_llm

        prompt = (
            f"Tentative diagnosis from image classifier: {readable} "
            f"(confidence {confidence:.1%}).\n\n"
            f"Retrieved passages:\n{evidence}\n\n"
            f"Task: State whether the passages support this diagnosis, then give "
            f"a concise management/treatment recommendation grounded in and cited "
            f"to the passages above."
        )
        try:
            answer = call_llm(prompt, system=SYSTEM_PROMPT)
        except Exception as exc:  # network/provider failure -> fall back, don't crash
            fallback = _extractive_summary(readable, passages)
            return ReasoningResult(
                recommendation=(
                    f"[LLM call failed: {exc}. Falling back to evidence summary.]\n"
                    f"{fallback}"
                ),
                used_llm=False,
                doc_ids=doc_ids,
                sources=sources,
                evidence_summary=evidence,
            )

        return ReasoningResult(
            recommendation=answer,
            used_llm=True,
            doc_ids=doc_ids,
            sources=sources,
            evidence_summary=evidence,
        )
