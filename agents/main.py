"""Main entry point — run the agentic diagnosis pipeline end-to-end.

Phase 1: Uses MockVisionAgent with stubbed retrieval.
Phase 3: Switches to real vision + real retrieval.

Usage:
    python main.py                    # Mock mode (default)
    python main.py --mode ensemble    # Real checkpoint
    python main.py --num-samples 100  # More test images
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import config
from agents.vision_agent import create_vision_agent, _load_class_names, _clean_class_name
from agents.confidence_quality_agent import ConfidenceQualityAgent
from agents.orchestrator import Orchestrator
from agents.evaluate import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def create_synthetic_test_set(
    class_names: list[str],
    num_samples: int = 50,
    seed: int = 42,
) -> dict[str, str]:
    """Create a synthetic test set with known ground truth.

    Returns dict mapping image_id -> true_class_name.
    Distributes samples across classes, with extra weight on healthy
    and common diseases.
    """
    import random
    rng = random.Random(seed)

    # Ensure at least one sample from each class, then fill randomly
    test_set: dict[str, str] = {}
    for i, cn in enumerate(class_names):
        test_set[f"img_{i:04d}"] = cn

    # Fill remaining with random classes (biased toward common ones)
    common_diseases = [
        "Tomato___Late_blight",
        "Tomato___Early_blight",
        "Potato___Late_blight",
        "Apple___Apple_scab",
        "Corn_(maize)___Common_rust_",
        "Tomato___healthy",
        "Potato___healthy",
        "Apple___healthy",
    ]
    while len(test_set) < num_samples:
        idx = len(test_set)
        # 40% chance of common disease, 60% random
        if rng.random() < 0.4:
            cn = rng.choice(common_diseases)
        else:
            cn = rng.choice(class_names)
        test_set[f"img_{idx:04d}"] = cn

    return test_set


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_test_images(
    test_dir: Path, class_names: list[str]
) -> list[tuple[str, Path, str]]:
    """Scan TEST_IMAGE_DIR/<class>/*.img into (image_id, path, true_label).

    The subfolder name is the ground-truth label and must match a class in
    class_names.json exactly. Classes with no images are simply skipped.
    """
    test_dir = Path(test_dir)
    class_set = set(class_names)
    samples: list[tuple[str, Path, str]] = []
    if not test_dir.exists():
        return samples
    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in class_set:
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                samples.append((f"{cls_dir.name}/{p.name}", p, cls_dir.name))
    return samples


def print_section(title: str, width: int = 70) -> None:
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def main():
    parser = argparse.ArgumentParser(description="Agentic Plant Disease Diagnosis")
    parser.add_argument(
        "--mode", choices=["mock", "ensemble"], default="mock",
        help="Vision backend: 'mock' for synthetic, 'ensemble' for real models"
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Number of synthetic test images"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log-path", type=str, default=None,
        help="Path to write decision log JSON"
    )
    parser.add_argument(
        "--retrieval", choices=["auto", "on", "off"], default="auto",
        help="Wire real RAG retrieval + reasoning into the moderate tier. "
             "'auto' enables it when the FAISS index exists."
    )
    parser.add_argument(
        "--test-image-dir", type=str, default=config.TEST_IMAGE_DIR,
        help="Directory of real test images (per-class subfolders). Used in "
             "ensemble mode. Defaults to TEST_IMAGE_DIR from .env."
    )
    parser.add_argument(
        "--models", default="ensemble",
        choices=["ensemble", "fedavg", "fedopt", "fedma"],
        help="Vision configuration: the full 3-model ensemble (default) or a "
             "single named checkpoint (for single-vs-ensemble comparison)."
    )
    args = parser.parse_args()

    class_names = _load_class_names()

    # Decide whether we run on REAL images (ensemble mode + images present).
    real_samples: list[tuple] = []
    if args.mode == "ensemble" and args.test_image_dir:
        real_samples = scan_test_images(Path(args.test_image_dir), class_names)
    use_real_images = args.mode == "ensemble" and len(real_samples) > 0

    print_section("AGENTIC PLANT DISEASE DIAGNOSIS")
    print(f"Mode: {args.mode}" + ("  (real images)" if use_real_images else ""))
    print(f"Classes: {len(class_names)}")
    print(f"Healthy classes: {sum(1 for c in class_names if c.endswith('___healthy'))}")

    # ── Step 1: Create vision agent ──────────────────────────────────────
    print_section("STEP 1: Vision Agent")
    if args.mode == "ensemble" and not use_real_images:
        print(f"  Ensemble mode requested but no images found in "
              f"'{args.test_image_dir}'.")
        print(f"  Falling back to MockVisionAgent so the run still produces a "
              f"report. Add images to the per-class subfolders to evaluate the "
              f"real checkpoint.")
        args.mode = "mock"

    if args.mode == "ensemble":
        all_checkpoints = {
            "fedma": PROJECT_ROOT / "model" / "fedma_global_model_plantvillage.pth",
            "fedopt": PROJECT_ROOT / "model" / "fedopt_plantvillage.pth",
            "fedavg": PROJECT_ROOT / "model" / "fedavg_round10.pt",
        }
        if args.models == "ensemble":
            checkpoint_paths = list(all_checkpoints.values())
        else:
            checkpoint_paths = [all_checkpoints[args.models]]
        vision_agent = create_vision_agent(
            mode="ensemble",
            checkpoint_paths=checkpoint_paths,
        )
        print(f"Loaded {len(checkpoint_paths)} model(s)")
        print(f"Real test images: {len(real_samples)}")
    else:
        vision_agent = create_vision_agent(mode="mock", seed=args.seed)
        print("MockVisionAgent created (synthetic probability distributions)")

    # ── Step 1b: Wire retrieval + reasoning (Phase 3) ────────────────────
    print_section("STEP 1b: Retrieval + Reasoning")
    index_exists = (config.FAISS_INDEX_DIR / "index.faiss").exists()
    if args.retrieval == "on":
        enable_retrieval = True
    elif args.retrieval == "off":
        enable_retrieval = False
    else:  # auto
        enable_retrieval = index_exists

    retriever = None
    reasoning_agent = None
    if enable_retrieval:
        if not index_exists:
            print(f"  Index not found at {config.FAISS_INDEX_DIR} — run "
                  f"rag/rag/index_corpus.py first. Falling back to stub.")
            enable_retrieval = False
        else:
            from agents.retrieval import get_retriever
            from agents.reasoning_agent import ReasoningAgent
            retriever = get_retriever()
            reasoning_agent = ReasoningAgent()
            print(f"  Retrieval: ON  (FAISS index: {config.FAISS_INDEX_DIR})")
            print(f"  Index chunks: {retriever.document_count}")
            print(f"  Retrieval top-k: {config.RETRIEVAL_TOP_K}")
            print(f"  LLM reasoning: "
                  f"{'ON (OpenRouter: ' + config.OPENROUTER_MODEL + ')' if config.llm_is_configured() else 'OFF — evidence-summary fallback (set OPENROUTER_API_KEY to enable)'}")
    if not enable_retrieval:
        print("  Retrieval: OFF (moderate tier uses the Phase-1 stub)")

    # ── Step 2: Create orchestrator ──────────────────────────────────────
    print_section("STEP 2: Orchestrator")
    log_path = args.log_path or str(
        PROJECT_ROOT / "agents" / "decision_log.json"
    )
    orchestrator = Orchestrator(
        vision_agent=vision_agent,
        quality_agent=ConfidenceQualityAgent(),
        decision_log_path=log_path,
        retriever=retriever,
        reasoning_agent=reasoning_agent,
        retrieval_top_k=config.RETRIEVAL_TOP_K,
    )
    print(f"Decision log path: {log_path}")
    print(f"Routing thresholds:")
    print(f"  HIGH confidence:  >= {orchestrator.CONFIDENCE_HIGH}")
    print(f"  LOW confidence:   <  {orchestrator.CONFIDENCE_LOW}")
    print(f"  MARGIN threshold: >= {orchestrator.MARGIN_THRESHOLD}")
    print(f"  DISAGREEMENT:     <  {orchestrator.AGREEMENT_DISAGREEMENT}")

    # ── Step 3: Build test set ───────────────────────────────────────────
    print_section("STEP 3: Test Set")
    if use_real_images:
        # Real images: (image_id, path, label). Ground truth = folder name.
        test_items = [(sid, path, label) for (sid, path, label) in real_samples]
        test_set = {sid: label for (sid, _path, label) in real_samples}
        n_classes = len(set(test_set.values()))
        print(f"Loaded {len(test_items)} real test images across {n_classes} classes")
        for sid, _p, label in test_items[:5]:
            print(f"  {sid}  ->  {label}")
        if len(test_items) > 5:
            print(f"  ... ({len(test_items) - 5} more)")
    else:
        test_set = create_synthetic_test_set(
            class_names, num_samples=args.num_samples, seed=args.seed
        )
        test_items = [(sid, None, label) for sid, label in test_set.items()]
        print(f"Generated {len(test_set)} synthetic test images")
        print(f"Sample entries:")
        for img_id, label in list(test_set.items())[:5]:
            print(f"  {img_id}: {label}")
        print(f"  ... ({len(test_set) - 5} more)")

    # ── Step 4: Run pipeline ─────────────────────────────────────────────
    print_section("STEP 4: Running Pipeline")
    start_time = time.time()

    decisions = []
    for i, (img_id, path, true_label) in enumerate(test_items):
        if use_real_images:
            # Load the real image; pass a numpy array too for quality checks.
            from PIL import Image
            import numpy as _np
            pil = Image.open(path).convert("RGB")
            image = pil
            image_array = _np.array(pil)
        else:
            # Mock mode: pass the ground-truth label as the "image" so the
            # MockVisionAgent acts as a realistic (~82% accurate) classifier
            # whose confidence correlates with correctness.
            image = true_label
            image_array = None

        decision = orchestrator.process_image(
            image=image,
            image_id=img_id,
            image_array=image_array,
        )
        decisions.append(decision)

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(test_items)}] "
                f"{img_id[:30]}: {decision.action:25s} "
                f"conf={decision.confidence:.3f} "
                f"pred={decision.prediction[:30]}"
            )

    elapsed = time.time() - start_time
    print(f"\nCompleted {len(decisions)} images in {elapsed:.2f}s "
          f"({len(decisions)/max(elapsed, 0.001):.0f} img/s)")

    # ── Step 5: Sample outputs (one of each action, fuller text) ─────────
    print_section("STEP 5: Sample Outputs")
    # Show up to two examples per action so every branch is visible, and print
    # the full retrieve_and_reason output (that's the grounded recommendation).
    shown: dict[str, int] = {}
    samples = []
    for d in decisions:
        if shown.get(d.action, 0) < 2:
            samples.append(d)
            shown[d.action] = shown.get(d.action, 0) + 1
    for d in samples:
        full = d.action == "retrieve_and_reason"
        print(f"\n--- {d.image_id}  [{d.action}] ---")
        print(f"  Prediction:  {d.prediction}")
        print(f"  Confidence:  {d.confidence:.4f}   Margin: {d.margin:.4f}   "
              f"Agreement: {d.agreement_ratio:.0%}   Healthy: {d.is_healthy}")
        if d.retrieved_doc_ids:
            print(f"  Retrieved:   {len(d.retrieved_doc_ids)} chunk(s)")
        out = d.output if full else (d.output[:200] + ("…" if len(d.output) > 200 else ""))
        print(f"  Output:\n    " + out.replace("\n", "\n    "))

    # ── Step 6: Evaluation Report ────────────────────────────────────────
    print_section("STEP 6: Evaluation Report")

    n_models = len(checkpoint_paths) if args.mode == "ensemble" else 1
    evaluator = Evaluator(
        ground_truth=test_set,
        class_names=class_names,
        n_models=n_models,
    )
    report = evaluator.evaluate(
        [d.__dict__ if hasattr(d, '__dict__') else d for d in decisions]
    )

    print(f"  Total images evaluated:     {report.total_images}")
    print(f"  Classifier accuracy:        {report.classifier_accuracy:.1%}")
    print(f"  Classifier macro-F1:        {report.classifier_macro_f1:.1%}")
    print(f"  Correct action rate:        {report.correct_action_rate:.1%}")
    print(f"  Harmful action rate:        {report.harmful_action_rate:.1%}")
    print(f"  Naive baseline harmful:     {report.naive_baseline_harmful_rate:.1%}")
    print(f"  Human intervention rate:    {report.human_intervention_rate:.1%}")
    print(f"  Direct diagnosis rate:      {report.direct_diagnosis_rate:.1%}")
    print(f"  Retrieve & reason rate:     {report.retrieve_reason_rate:.1%}")
    print(f"  Healthy correct rate:       {report.healthy_correct_rate:.1%}")
    print(f"\n  Action breakdown:")
    for action, count in sorted(report.action_breakdown.items()):
        print(f"    {action:30s}: {count:4d} ({count/report.total_images:.1%})")
    print(f"\n  Ablation (ensemble disagreement):")
    print(f"    {report.ablation_note}")

    # ── Step 7: Per-class accuracy (top/bottom 5) ───────────────────────
    if report.per_class_accuracy:
        print_section("STEP 7: Per-Class Accuracy")
        per_class = report.per_class_accuracy
        sorted_classes = sorted(
            per_class.items(),
            key=lambda x: x[1]["correct"] / max(1, x[1]["total"]),
            reverse=True,
        )
        print("  Top 5:")
        for cn, stats in sorted_classes[:5]:
            acc = stats["correct"] / max(1, stats["total"])
            print(f"    {cn[:45]:45s}: {acc:.0%} ({stats['correct']}/{stats['total']})")
        print("  Bottom 5:")
        for cn, stats in sorted_classes[-5:]:
            acc = stats["correct"] / max(1, stats["total"])
            print(f"    {cn[:45]:45s}: {acc:.0%} ({stats['correct']}/{stats['total']})")

    # ── Step 8: Class coverage report ────────────────────────────────────
    print_section("STEP 8: Class Coverage Report (for paper)")
    if enable_retrieval and retriever is not None:
        # Real coverage: query each class and count strong (>=0.75 cosine) hits.
        COVERAGE_SIM = 0.75
        retrieved_class_counts: dict[str, int] = {}
        for cn in class_names:
            passages = retriever.retrieve(_clean_class_name(cn), k=config.RETRIEVAL_TOP_K)
            retrieved_class_counts[cn] = sum(1 for p in passages if p.score >= COVERAGE_SIM)
        coverage = evaluator.generate_coverage_report(retrieved_class_counts)
        print(f"  Total classes:   {coverage['total_classes']}")
        print(f"  Covered:         {coverage['covered_classes']}")
        print(f"  Uncovered:       {coverage['uncovered_classes']}")
        print(f"  Coverage ratio:  {coverage['coverage_ratio']:.1%}")
        if coverage["uncovered"]:
            print("  Uncovered classes:")
            for e in coverage["uncovered"]:
                print(f"    - {e['class']}")
        else:
            print("  All 38 classes have retrievable documents (>=0.75 cosine).")
    else:
        coverage = evaluator.generate_coverage_report(retrieved_class_counts={})
        print(f"  Total classes:   {coverage['total_classes']}")
        print(f"  Covered:         {coverage['covered_classes']}")
        print(f"  Uncovered:       {coverage['uncovered_classes']}")
        print(f"  Coverage ratio:  {coverage['coverage_ratio']:.1%}")
        print(f"\n  Retrieval stubbed — run with --retrieval on (needs the FAISS "
              f"index) to populate coverage.")

    # ── Save results ─────────────────────────────────────────────────────
    results_path = PROJECT_ROOT / "agents" / "evaluation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "retrieval_enabled": bool(enable_retrieval),
                "llm_reasoning_enabled": bool(enable_retrieval and config.llm_is_configured()),
                "num_samples": len(decisions),
                "report": {
                    "total_images": report.total_images,
                    "classifier_accuracy": report.classifier_accuracy,
                    "classifier_macro_f1": report.classifier_macro_f1,
                    "correct_action_rate": report.correct_action_rate,
                    "harmful_action_rate": report.harmful_action_rate,
                    "naive_baseline_harmful_rate": report.naive_baseline_harmful_rate,
                    "human_intervention_rate": report.human_intervention_rate,
                    "direct_diagnosis_rate": report.direct_diagnosis_rate,
                    "retrieve_reason_rate": report.retrieve_reason_rate,
                    "action_breakdown": report.action_breakdown,
                    "ablation_note": report.ablation_note,
                },
                "coverage_report": coverage,
            },
            f, indent=2,
        )
    print(f"\n  Results saved to: {results_path}")
    orchestrator.flush_log()
    print(f"  Decision log:    {log_path}")

    if enable_retrieval:
        print_section("RUN COMPLETE — full pipeline (vision -> route -> retrieve -> reason)")
        print("  Real retrieval + reasoning wired in over the FAISS index.")
        if not config.llm_is_configured():
            print("  LLM reasoning used the offline evidence-summary fallback.")
            print("  Set OPENROUTER_API_KEY in .env for full LLM-grounded answers.")
    else:
        print_section("RUN COMPLETE — agentic skeleton (retrieval stubbed)")
        print("  Routing, escalation, healthy-class handling, and metrics all work.")
        print("  Run with --retrieval on (after indexing) for the full pipeline.")


if __name__ == "__main__":
    main()
