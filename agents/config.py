"""Central configuration for the agentic diagnosis layer.

All tunable values live here so a rerun only needs env / flag changes.
Reads secrets and paths from a project-root `.env` via python-dotenv.
Never hardcodes an API key; the key is read from the environment only.
"""

from __future__ import annotations

import os
from pathlib import Path

# macOS ships multiple OpenMP runtimes; PyTorch and FAISS each bundle one, and
# loading both in-process segfaults on query. This is the documented workaround
# and must be set before torch/faiss import. config.py is imported first by
# every agent entry point, so setting it here fixes all of them.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Single-threaded OpenMP additionally avoids a FAISS+PyTorch query-time segfault
# on macOS. Retrieval is not thread-bound here; override by exporting your own.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Load .env from the project root (one level up from agents/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv optional; env vars still work without it
    pass


def _resolve(path_str: str) -> Path:
    """Resolve a path that may be absolute or project-root-relative."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# ── Paths ────────────────────────────────────────────────────────────────────

MODEL_DIR = PROJECT_ROOT / "model"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# Ensemble checkpoints. N=1 today; append paths to grow the ensemble.
CHECKPOINT_PATHS = [
    MODEL_DIR / "fedma_global_model_plantvillage.pth",
    # MODEL_DIR / "fedopt_plantvillage.pth",
]

# RAG project root (its own package lives at rag/rag).
RAG_PROJECT_ROOT = PROJECT_ROOT / "rag" / "rag"

# FAISS index built by rag/rag/index_corpus.py.
FAISS_INDEX_DIR = _resolve(os.environ.get("FAISS_INDEX_DIR", "rag/rag/data/faiss_index"))

# Where per-run artifacts are written.
DECISION_LOG_PATH = PROJECT_ROOT / "agents" / "decision_log.json"
EVAL_RESULTS_PATH = PROJECT_ROOT / "agents" / "evaluation_results.json"
COVERAGE_REPORT_PATH = PROJECT_ROOT / "agents" / "coverage_report.json"

# ── Vision ───────────────────────────────────────────────────────────────────

VISION_MODE = os.environ.get("VISION_MODE", "mock")  # "mock" | "ensemble"
# Real test images (populated later). One flag reruns the same evaluation.
TEST_IMAGE_DIR = os.environ.get("TEST_IMAGE_DIR", "").strip() or None

# ── Retrieval ────────────────────────────────────────────────────────────────

RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))

# ── LLM (OpenRouter, OpenAI-API-compatible) ──────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).strip()
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()


def llm_is_configured() -> bool:
    """True when an OpenRouter key is present so call_llm() can run."""
    return bool(OPENROUTER_API_KEY)


# ── Routing thresholds (mirrored on Orchestrator for reference) ──────────────

CONFIDENCE_HIGH = 0.85
CONFIDENCE_LOW = 0.55
MARGIN_THRESHOLD = 0.15
AGREEMENT_DISAGREEMENT = 0.5
