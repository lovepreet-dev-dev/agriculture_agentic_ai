# Agentic AI for Plant Disease Diagnosis

An **agentic layer** wrapped around a PlantVillage ResNet18 classifier. Instead
of emitting a bare label, the system reasons about its own confidence, retrieves
supporting evidence from a scientific/extension corpus, explains the diagnosis,
and **defers to a human when uncertain**.

```
image ─▶ Vision Agent ─▶ Confidence/Quality Agent ─▶ Orchestrator (3-tier routing)
                                                        ├─ direct_diagnosis   (high confidence, healthy shortcut)
                                                        ├─ retrieve_and_reason (moderate → RAG + LLM, grounded + cited)
                                                        └─ escalate_to_human   (low confidence / disagreement / bad image)
```

## Components (`agents/`)

| File | Role |
|------|------|
| `vision_agent.py` | Decoupled `VisionAgent` contract. `EnsembleVisionAgent` (N ResNet18 checkpoints, softmax-averaged) and `MockVisionAgent` (realistic synthetic classifier for image-free evaluation). |
| `confidence_quality_agent.py` | Confidence, margin, ensemble agreement, and image-quality checks (Laplacian blur + brightness). |
| `orchestrator.py` | The 3-tier routing policy, healthy-class shortcut, empty-retrieval fallback, and JSON decision logging. |
| `retrieval.py` | Thin adapter over the **RAG project's own** bge-small embeddings + FAISS index. Returns cited `Passage`s (cosine similarity). |
| `reasoning_agent.py` | Grounds the diagnosis in retrieved evidence via the LLM; falls back to a deterministic evidence summary when no API key is set. |
| `llm.py` | Single `call_llm(prompt)` wrapper (OpenRouter, OpenAI-API-compatible). Swap providers here only. |
| `evaluate.py` | Accuracy, correct-action rate, harmful-action rate, naive-baseline harmful rate, human-intervention rate, class coverage. |
| `config.py` | All paths, thresholds, and env-driven settings in one place. |
| `main.py` | End-to-end run + evaluation report. |

## Setup

```bash
pip install -r requirements.txt          # agents layer
pip install -r rag/rag/requirements.txt  # RAG retrieval stack

cp .env.example .env                      # then paste your OpenRouter key into .env
```

`.env` is gitignored. `OPENROUTER_API_KEY` is read from the environment only —
never commit a key. Without a key the pipeline still runs; the reasoning step
uses an offline evidence-summary fallback.

## Running

```bash
# 1) Build the retrieval index (RAG project's OWN pipeline) — once.
cd rag/rag && python index_corpus.py && cd ../..

# 2) Sanity-check retrieval and class coverage.
python agents/check_retrieval.py
python agents/coverage_check.py           # writes agents/coverage_report.json

# 3) Full pipeline end-to-end (mock vision + real retrieval/reasoning).
python agents/main.py --num-samples 60 --retrieval on

# Agentic skeleton only (retrieval stubbed) — proves routing without the index.
python agents/main.py --num-samples 60 --retrieval off
```

Key flags: `--mode {mock,ensemble}`, `--retrieval {auto,on,off}`,
`--num-samples N`, `--seed N`. `auto` enables retrieval when the FAISS index
exists. Real test images go in `TEST_IMAGE_DIR` (config/`.env`) for later
`--mode ensemble` runs.

## Design notes

- **Vision is mocked; retrieval is real from day one.** `MockVisionAgent`
  simulates an ~82%-accurate classifier whose confidence correlates with
  correctness, so the evaluation demonstrates the agentic layer reducing harmful
  actions well below the naive "always trust the classifier" baseline.
- The real checkpoint is treated as a **loads-and-predicts smoke test**
  (`--mode ensemble`) until labelled test images are added.
- `class_names.json` is in exact training index order and must not be reordered.
