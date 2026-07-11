"""Orchestrator — central routing agent for the agentic diagnosis pipeline.

Implements the exact 3-tier routing policy:
  1. HIGH CONFIDENCE: agreement==1.0 AND confidence>=0.85 AND margin>=0.15
     AND good quality -> skip retrieval, return direct treatment lookup
  2. MODERATE: confidence>=0.55 OR low margin OR partial disagreement
     (agreement>=0.5) with good quality -> full pipeline (retrieve+reason)
  3. ESCALATE: confidence<0.55 OR heavy disagreement (agreement<0.5) OR
     bad quality -> human escalation, NO automated recommendation

Also handles:
  - Healthy-class shortcut (14 classes ending in ___healthy)
  - Empty-retrieval fallback
  - Decision logging to JSON
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.vision_agent import VisionAgent, VisionPrediction, _clean_class_name, _load_class_names
from agents.confidence_quality_agent import ConfidenceQualityAgent, QualityAssessment

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class OrchestratorDecision:
    """Complete output of the orchestrator for one image."""

    image_id: str
    prediction: str
    confidence: float
    margin: float
    agreement_ratio: float
    quality_ok: bool
    action: str  # "direct_diagnosis", "retrieve_and_reason", "escalate_to_human"
    output: str
    retrieved_doc_ids: list[str] = field(default_factory=list)
    reasoning_output: str = ""
    is_healthy: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────


class Orchestrator:
    """Routes diagnosis through the appropriate pipeline tier.

    Args:
        vision_agent: Backend implementing VisionAgent contract.
        quality_agent: Confidence/quality assessment agent.
        decision_log_path: Where to write per-image decision logs.
    """

    # Thresholds (exact values from spec)
    CONFIDENCE_HIGH = 0.85
    CONFIDENCE_LOW = 0.55
    MARGIN_THRESHOLD = 0.15
    AGREEMENT_DISAGREEMENT = 0.5

    def __init__(
        self,
        vision_agent: VisionAgent,
        quality_agent: ConfidenceQualityAgent | None = None,
        decision_log_path: str | Path | None = None,
    ):
        self.vision_agent = vision_agent
        self.quality_agent = quality_agent or ConfidenceQualityAgent()
        self.class_names = _load_class_names()
        self.healthy_classes = {
            cn for cn in self.class_names if cn.endswith("___healthy")
        }
        self.decision_log_path = (
            Path(decision_log_path) if decision_log_path else None
        )
        self._decisions: list[dict[str, Any]] = []

    def process_image(
        self,
        image: Any,
        image_id: str = "unknown",
        image_array: Any = None,
    ) -> OrchestratorDecision:
        """Run full orchestration on one image.

        Args:
            image: Image input to the vision agent (PIL, tensor, path).
            image_id: Identifier for logging.
            image_array: Optional numpy array for quality checks.

        Returns:
            OrchestratorDecision with action taken and output.
        """
        # ── Step 1: Vision prediction ────────────────────────────────────
        pred = self.vision_agent.predict(image)

        # ── Step 2: Quality assessment ───────────────────────────────────
        quality = self.quality_agent.assess(
            confidence=pred.confidence,
            margin=pred.confidence - sorted(pred.probs.values())[-2]
            if len(pred.probs) > 1
            else 0.0,
            agreement_ratio=pred.agreement_ratio,
            image=image_array,
        )

        # Recompute margin properly from probs
        sorted_probs = sorted(pred.probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.0

        quality = self.quality_agent.assess(
            confidence=pred.confidence,
            margin=margin,
            agreement_ratio=pred.agreement_ratio,
            image=image_array,
        )

        # ── Step 3: Check if healthy ─────────────────────────────────────
        is_healthy = pred.label in self.healthy_classes

        if is_healthy:
            decision = self._handle_healthy(pred, quality, image_id)
            self._log_decision(decision)
            return decision

        # ── Step 4: Route through 3-tier policy ──────────────────────────
        action = self._determine_action(pred, quality)

        if action == "direct_diagnosis":
            decision = self._handle_direct(pred, quality, image_id)
        elif action == "retrieve_and_reason":
            decision = self._handle_retrieve_and_reason(pred, quality, image_id)
        else:
            decision = self._handle_escalate(pred, quality, image_id)

        self._log_decision(decision)
        return decision

    def _determine_action(
        self, pred: VisionPrediction, quality: QualityAssessment
    ) -> str:
        """Apply the exact 3-tier routing policy.

        Returns one of: "direct_diagnosis", "retrieve_and_reason",
        "escalate_to_human".
        """
        # Tier 3: ESCALATE — low confidence, heavy disagreement, or bad quality
        if pred.confidence < self.CONFIDENCE_LOW:
            return "escalate_to_human"
        if pred.agreement_ratio < self.AGREEMENT_DISAGREEMENT:
            return "escalate_to_human"
        if not quality.quality_ok:
            return "escalate_to_human"

        # Tier 1: DIRECT — high confidence, full agreement, good margin, good quality
        sorted_probs = sorted(pred.probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.0

        if (
            pred.agreement_ratio == 1.0
            and pred.confidence >= self.CONFIDENCE_HIGH
            and margin >= self.MARGIN_THRESHOLD
            and quality.quality_ok
        ):
            return "direct_diagnosis"

        # Tier 2: RETRIEVE AND REASON — moderate confidence or low margin or partial disagreement
        return "retrieve_and_reason"

    def _handle_healthy(
        self,
        pred: VisionPrediction,
        quality: QualityAssessment,
        image_id: str,
    ) -> OrchestratorDecision:
        """Handle healthy plant classification — skip retrieval entirely."""
        crop = pred.label.split("___")[0].replace("_", " ").replace("(", "").replace(")", "").strip()
        output = (
            f"Plant appears healthy ({crop}). Continue regular monitoring. "
            f"No disease treatment needed."
        )
        return OrchestratorDecision(
            image_id=image_id,
            prediction=pred.label,
            confidence=pred.confidence,
            margin=0.0,
            agreement_ratio=pred.agreement_ratio,
            quality_ok=quality.quality_ok,
            action="direct_diagnosis",
            output=output,
            is_healthy=True,
        )

    def _handle_direct(
        self,
        pred: VisionPrediction,
        quality: QualityAssessment,
        image_id: str,
    ) -> OrchestratorDecision:
        """Handle high-confidence direct diagnosis (no retrieval needed)."""
        readable = _clean_class_name(pred.label)
        output = (
            f"Diagnosis: {readable} "
            f"(confidence={pred.confidence:.1%}, "
            f"margin={pred.margin:.3f}, "
            f"agreement={pred.agreement_ratio:.0%}). "
            f"Recommended treatment: Consult local agricultural extension "
            f"for {readable} management guidelines."
        )
        return OrchestratorDecision(
            image_id=image_id,
            prediction=pred.label,
            confidence=pred.confidence,
            margin=pred.margin,
            agreement_ratio=pred.agreement_ratio,
            quality_ok=quality.quality_ok,
            action="direct_diagnosis",
            output=output,
        )

    def _handle_retrieve_and_reason(
        self,
        pred: VisionPrediction,
        quality: QualityAssessment,
        image_id: str,
    ) -> OrchestratorDecision:
        """Handle moderate-confidence cases — retrieve evidence and reason.

        STUB: This calls retrieve_and_reason() which is a placeholder.
        In Phase 3 this will be replaced with real retrieval + reasoning.
        """
        result = self.retrieve_and_reason(pred, image_id)
        result.quality_ok = quality.quality_ok
        self._log_decision(result)
        return result

    def _handle_escalate(
        self,
        pred: VisionPrediction,
        quality: QualityAssessment,
        image_id: str,
    ) -> OrchestratorDecision:
        """Handle low-confidence / disagreement / bad quality — escalate."""
        reasons = []
        if pred.confidence < self.CONFIDENCE_LOW:
            reasons.append(f"low confidence ({pred.confidence:.1%})")
        if pred.agreement_ratio < self.AGREEMENT_DISAGREEMENT:
            reasons.append(f"heavy disagreement ({pred.agreement_ratio:.0%})")
        if not quality.quality_ok:
            reasons.append("poor image quality")
            reasons.extend(quality.quality_reasons)

        output = (
            f"ESCALATE TO HUMAN EXPERT — cannot provide automated "
            f"recommendation. Reasons: {'; '.join(reasons)}. "
            f"Classifier suggested: {_clean_class_name(pred.label)}. "
            f"Please consult a plant pathology expert."
        )
        return OrchestratorDecision(
            image_id=image_id,
            prediction=pred.label,
            confidence=pred.confidence,
            margin=pred.margin,
            agreement_ratio=pred.agreement_ratio,
            quality_ok=quality.quality_ok,
            action="escalate_to_human",
            output=output,
        )

    # ── STUB: to be replaced in Phase 3 ──────────────────────────────────

    def retrieve_and_reason(
        self, pred: VisionPrediction, image_id: str
    ) -> OrchestratorDecision:
        """Placeholder for retrieval + reasoning pipeline.

        Phase 3 will replace this with:
          1. Clean class name -> query string
          2. Call real RAG retrieval
          3. If no results -> escalate to human
          4. Pass to reasoning agent -> grounded recommendation
        """
        readable = _clean_class_name(pred.label)
        output = (
            f"[STUB] Retrieve and reason for: {readable}. "
            f"Confidence={pred.confidence:.1%}, margin={pred.margin:.3f}. "
            f"Retrieval pipeline not yet connected."
        )
        return OrchestratorDecision(
            image_id=image_id,
            prediction=pred.label,
            confidence=pred.confidence,
            margin=pred.margin,
            agreement_ratio=pred.agreement_ratio,
            quality_ok=True,
            action="retrieve_and_reason",
            output=output,
        )

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_decision(self, decision: OrchestratorDecision) -> None:
        """Append decision to in-memory log and optionally to disk."""
        entry = asdict(decision)
        self._decisions.append(entry)
        logger.info(
            "orchestrator_decision",
            image_id=decision.image_id,
            action=decision.action,
            prediction=decision.prediction,
            confidence=round(decision.confidence, 4),
        )

    def flush_log(self) -> None:
        """Write all accumulated decisions to disk."""
        if self.decision_log_path is None:
            return
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.decision_log_path, "w") as f:
            json.dump(self._decisions, f, indent=2)
        logger.info(
            "decision_log_written",
            path=str(self.decision_log_path),
            count=len(self._decisions),
        )

    def get_decisions(self) -> list[dict[str, Any]]:
        """Return all accumulated decisions."""
        return list(self._decisions)
