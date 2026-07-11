"""Evaluation metrics for the agentic diagnosis pipeline.

Computes:
  - Classifier accuracy (when ground truth is available)
  - Correct-action rate (did the system do the right thing?)
  - Harmful-action rate (did it give a wrong treatment recommendation?)
  - Naive-baseline harmful rate (if we always trusted the classifier)
  - Human-intervention rate (how often it escalated)
  - Ablation: errors caught by ensemble disagreement that confidence
    thresholding alone would have missed (N/A with N=1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationReport:
    """Complete evaluation results."""

    total_images: int = 0
    classifier_accuracy: float = 0.0
    classifier_macro_f1: float = 0.0
    correct_action_rate: float = 0.0
    harmful_action_rate: float = 0.0
    naive_baseline_harmful_rate: float = 0.0
    human_intervention_rate: float = 0.0
    direct_diagnosis_rate: float = 0.0
    retrieve_reason_rate: float = 0.0
    healthy_correct_rate: float = 0.0
    ablation_disagreement_caught: int = 0
    ablation_note: str = ""
    action_breakdown: dict[str, int] = field(default_factory=dict)
    per_class_accuracy: dict[str, dict[str, Any]] = field(default_factory=dict)


class Evaluator:
    """Evaluates orchestrator decisions against ground truth.

    Args:
        ground_truth: Dict mapping image_id -> true_class_name.
        class_names: Full list of 38 class names.
    """

    def __init__(
        self,
        ground_truth: dict[str, str] | None = None,
        class_names: list[str] | None = None,
        n_models: int = 1,
    ):
        self.ground_truth = ground_truth or {}
        self.class_names = class_names or []
        self.n_models = n_models

    def evaluate(
        self,
        decisions: list[dict[str, Any]],
    ) -> EvaluationReport:
        """Compute all metrics from a list of OrchestratorDecision dicts.

        Args:
            decisions: List of dicts from orchestrator.get_decisions().

        Returns:
            EvaluationReport with all metrics.
        """
        if not decisions:
            return EvaluationReport(ablation_note="No decisions to evaluate.")

        report = EvaluationReport(total_images=len(decisions))

        correct_actions = 0
        harmful_actions = 0
        naive_harmful = 0
        human_escalations = 0
        direct_count = 0
        retrieve_count = 0
        healthy_correct = 0
        total_with_gt = 0
        disagreements_caught = 0
        n_disagreements = 0

        action_counts: dict[str, int] = {}
        per_class: dict[str, dict[str, Any]] = {}

        for d in decisions:
            action = d.get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1

            if action == "direct_diagnosis":
                direct_count += 1
            elif action == "retrieve_and_reason":
                retrieve_count += 1
            elif action == "escalate_to_human":
                human_escalations += 1

            # If we have ground truth, compute accuracy metrics
            image_id = d.get("image_id", "")
            if image_id in self.ground_truth:
                gt = self.ground_truth[image_id]
                pred = d.get("prediction", "")
                total_with_gt += 1

                # Track per-class
                if gt not in per_class:
                    per_class[gt] = {"total": 0, "correct": 0, "actions": {}}
                per_class[gt]["total"] += 1
                per_class[gt]["actions"][action] = (
                    per_class[gt]["actions"].get(action, 0) + 1
                )

                is_correct_pred = pred == gt

                if is_correct_pred:
                    per_class[gt]["correct"] += 1

                # Correct action: either correctly predicted + direct,
                # or escalated when uncertain, or retrieved when moderate
                if action == "direct_diagnosis" and is_correct_pred:
                    correct_actions += 1
                elif action == "retrieve_and_reason":
                    # Retrieve is always "correct" (not harmful, just cautious)
                    correct_actions += 1
                elif action == "escalate_to_human":
                    # Escalation is correct if the prediction was wrong
                    if not is_correct_pred:
                        correct_actions += 1
                    # Also correct if prediction was right but confidence was low
                    # (better safe than sorry)

                # Harmful action: gave a direct diagnosis that was WRONG
                if action == "direct_diagnosis" and not is_correct_pred:
                    harmful_actions += 1

                # Naive baseline: if we always trusted the classifier directly
                if not is_correct_pred:
                    naive_harmful += 1

                # Healthy correct
                is_healthy = d.get("is_healthy", False)
                if is_healthy and is_correct_pred:
                    healthy_correct += 1

                # Ablation: disagreements caught by ensemble
                # (With N=1, agreement is always 1.0, so this is N/A)
                agr = d.get("agreement_ratio", 1.0)
                conf = d.get("confidence", 0.0)
                if agr < 1.0:
                    n_disagreements += 1
                if (
                    agr < 1.0
                    and conf >= 0.55
                    and action == "escalate_to_human"
                    and not is_correct_pred
                ):
                    disagreements_caught += 1

        # ── Compute rates ────────────────────────────────────────────────
        if total_with_gt > 0:
            report.classifier_accuracy = sum(
                1 for d in decisions
                if d.get("image_id") in self.ground_truth
                and d.get("prediction") == self.ground_truth[d["image_id"]]
            ) / total_with_gt

            report.classifier_macro_f1 = self._macro_f1(decisions)

            report.correct_action_rate = correct_actions / total_with_gt
            report.harmful_action_rate = harmful_actions / total_with_gt
            report.naive_baseline_harmful_rate = naive_harmful / total_with_gt
            report.healthy_correct_rate = (
                healthy_correct / max(1, sum(
                    1 for d in decisions if d.get("is_healthy")
                ))
            )

        report.human_intervention_rate = human_escalations / len(decisions)
        report.direct_diagnosis_rate = direct_count / len(decisions)
        report.retrieve_reason_rate = retrieve_count / len(decisions)
        report.action_breakdown = action_counts
        report.ablation_disagreement_caught = disagreements_caught

        if self.n_models < 2:
            report.ablation_note = (
                f"Ablation N/A: with N={self.n_models} model, agreement_ratio is "
                "always 1.0 and cannot provide an independent uncertainty signal. "
                "This metric becomes meaningful when N>=2 models are used."
            )
        elif disagreements_caught > 0:
            report.ablation_note = (
                f"Ensemble of {self.n_models} models disagreed on "
                f"{n_disagreements} image(s); disagreement caught "
                f"{disagreements_caught} confident error(s) that "
                f"confidence-thresholding alone would have missed."
            )
        elif n_disagreements > 0:
            report.ablation_note = (
                f"Ensemble of {self.n_models} models disagreed on "
                f"{n_disagreements} image(s), all of which were routed correctly "
                f"(escalated or still predicted right), so disagreement caught no "
                f"additional missed errors this run."
            )
        else:
            report.ablation_note = (
                f"Ensemble of {self.n_models} models was unanimous on all "
                f"{len(decisions)} images (agreement_ratio 1.0 throughout), so the "
                f"disagreement signal was never triggered on this test set."
            )

        report.per_class_accuracy = per_class
        return report

    def _macro_f1(self, decisions: list[dict[str, Any]]) -> float:
        """Macro-averaged F1 of the Vision Agent (Table III in the paper).

        Computed over the classifier's raw predictions vs ground truth,
        independent of routing. Averaged over every class that appears in the
        ground truth (a class with no support contributes 0 to recall).
        """
        tp: dict[str, int] = {}
        fp: dict[str, int] = {}
        fn: dict[str, int] = {}
        gt_classes: set[str] = set()

        for d in decisions:
            image_id = d.get("image_id", "")
            if image_id not in self.ground_truth:
                continue
            y = self.ground_truth[image_id]
            yhat = d.get("prediction", "")
            gt_classes.add(y)
            if yhat == y:
                tp[y] = tp.get(y, 0) + 1
            else:
                fp[yhat] = fp.get(yhat, 0) + 1
                fn[y] = fn.get(y, 0) + 1

        if not gt_classes:
            return 0.0

        f1s = []
        for c in gt_classes:
            t = tp.get(c, 0)
            p_den = t + fp.get(c, 0)
            r_den = t + fn.get(c, 0)
            precision = t / p_den if p_den else 0.0
            recall = t / r_den if r_den else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            f1s.append(f1)
        return sum(f1s) / len(f1s)

    def generate_coverage_report(
        self,
        retrieved_class_counts: dict[str, int],
    ) -> dict[str, Any]:
        """Generate class coverage report for the paper.

        Args:
            retrieved_class_counts: Dict mapping class_name -> number of
                   retrieved documents for that class.

        Returns:
            Coverage report dict.
        """
        covered = []
        uncovered = []
        for cn in self.class_names:
            count = retrieved_class_counts.get(cn, 0)
            entry = {
                "class": cn,
                "readable_name": cn.replace("___", " - ").replace("_", " "),
                "retrieved_docs": count,
            }
            if count > 0:
                covered.append(entry)
            else:
                uncovered.append(entry)

        return {
            "total_classes": len(self.class_names),
            "covered_classes": len(covered),
            "uncovered_classes": len(uncovered),
            "coverage_ratio": len(covered) / max(1, len(self.class_names)),
            "covered": covered,
            "uncovered": uncovered,
        }
