#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Agentic Plant Disease Diagnosis — one-command demo launcher
# Usage:
#   bash run_demo.sh          # FAST: single vs 3-model ensemble comparison (~1 min)
#   bash run_demo.sh full     # FULL: ensemble + real RAG retrieval + LLM reasoning
# ─────────────────────────────────────────────────────────────────────────────

cd "/Users/lovepreet/Desktop/vibe projects/agriculture_agentic_ai" || {
  echo "ERROR: project folder not found"; exit 1; }

# Prevent the macOS FAISS + PyTorch library clash (harmless if not needed)
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

IMAGES="data/test_images_web"   # out-of-distribution web test images (190, 38 classes)

if [ "$1" == "full" ]; then
  echo "=============================================================="
  echo " FULL PIPELINE: 3-model ensemble + RAG retrieval + LLM reasoning"
  echo " (uses the internet for the language model; takes a few minutes)"
  echo "=============================================================="
  python3 agents/main.py --mode ensemble --retrieval on --test-image-dir "$IMAGES"
else
  echo "=============================================================="
  echo " COMPARISON: single model vs 3-model ensemble (fast, no LLM)"
  echo "=============================================================="
  for m in fedavg ensemble; do
    echo ""
    echo "########## CONFIG: $m ##########"
    python3 agents/main.py --mode ensemble --models "$m" --retrieval off \
      --test-image-dir "$IMAGES" 2>/dev/null \
      | sed -n '/STEP 6: Evaluation Report/,/Ablation/p'
  done
fi
