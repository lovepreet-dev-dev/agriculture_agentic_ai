"""Single-image demo: run ONE leaf photo through the full pipeline and print
everything a human would see -- prediction, confidence, routing decision,
retrieved evidence, and the LLM's grounded disease/treatment recommendation.

Usage:
    python3 agents/demo_one.py <path-to-image>
    python3 agents/demo_one.py                # uses a default sample image
"""
from __future__ import annotations
import os
# Quiet the environment BEFORE the heavy imports load:
#  - use the cached embedding model instead of re-checking HuggingFace online
#  - avoid the macOS FAISS/PyTorch clash
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
logging.disable(logging.INFO)  # hide JSON logs, httpx request lines, etc.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import config
from agents.vision_agent import create_vision_agent
from agents.confidence_quality_agent import ConfidenceQualityAgent
from agents.orchestrator import Orchestrator
from agents.retrieval import get_retriever
from agents.reasoning_agent import ReasoningAgent


def main():
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        # a moderate-confidence sample that tends to route through reasoning
        img_path = (PROJECT_ROOT / "data" / "test_images_web" /
                    "Cherry_(including_sour)___Powdery_mildew" / "web_02.jpg")

    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    print("=" * 68)
    print(f"  IMAGE: {img_path.name}   (true class folder: {img_path.parent.name})")
    print("=" * 68)

    checkpoints = [
        PROJECT_ROOT / "model" / "fedma_global_model_plantvillage.pth",
        PROJECT_ROOT / "model" / "fedopt_plantvillage.pth",
        PROJECT_ROOT / "model" / "fedavg_round10.pt",
    ]

    print("\n  Loading models and knowledge base ...")
    # Mute the RAG/FAISS startup chatter (it prints JSON to stdout) so the
    # demo shows only the diagnosis.
    import io, contextlib
    from PIL import Image
    import numpy as np
    with contextlib.redirect_stdout(io.StringIO()):
        vision = create_vision_agent(mode="ensemble", checkpoint_paths=checkpoints)
        retriever = get_retriever()
        reasoning = ReasoningAgent()
        orch = Orchestrator(
            vision_agent=vision,
            quality_agent=ConfidenceQualityAgent(),
            retriever=retriever,
            reasoning_agent=reasoning,
            retrieval_top_k=config.RETRIEVAL_TOP_K,
        )
        pil = Image.open(img_path).convert("RGB")
        d = orch.process_image(image=pil, image_id=img_path.name,
                               image_array=np.array(pil))

    print(f"\n  Prediction : {d.prediction}")
    print(f"  Confidence : {d.confidence:.1%}   Margin: {d.margin:.3f}   "
          f"Agreement: {d.agreement_ratio:.0%}")
    print(f"  ROUTING    : {d.action}")
    if d.retrieved_doc_ids:
        print(f"  Retrieved  : {len(d.retrieved_doc_ids)} evidence passage(s)")
    print("\n  ---------------- SYSTEM OUTPUT ----------------")
    print("  " + (d.output or "(no text)").replace("\n", "\n  "))
    print("  -----------------------------------------------")


if __name__ == "__main__":
    main()
