# Project Code Analysis — Agentic Pipeline

**Scope:** `agents/` + `rag/` + `model/` + `data/` — to ground the revision tasks in what the codebase already does.

## 1. Architecture → Paper Mapping

| Paper Agent | Code File | Class / Function | I/O Contract |
|---|---|---|---|
| Vision Agent | `agents/vision_agent.py:103-201` | `EnsembleVisionAgent`, `MockVisionAgent` → `VisionPrediction:34-42` | Input: PIL/Path/tensor 224×224. Output: `label, confidence, probs[38], agreement_ratio, per_model_preds`. Averaged softmax, margin `top1-top2`, agreement `count/N`. |
| Confidence & Image-Quality Agent | `agents/confidence_quality_agent.py:37-135` | `ConfidenceQualityAgent.assess:56-113` | Inputs: `confidence, margin, agreement_ratio, image (np.ndarray)`. Checks: Laplacian variance `blur_threshold=50.0` (`115-130`), brightness `[40, 220]` (`132-135`). Output: `QualityAssessment` (`quality_ok, blur_score, brightness_ok, quality_reasons`). If `image is None` → `quality_ok=True` fallback (`102-103`). |
| Retrieval Agent | `agents/retrieval.py:46-102` | `Retriever.retrieve:70-96` | Query=`_clean_class_name(label)` (`vision_agent.py:64-78`), `k=5` via `config.RETRIEVAL_TOP_K`. FAISS cosine sim derived from L2 `1-(d²)/2` (`85`), over `bge-small-en-v1.5` embeddings. `Passage` dataclass (`29-43`) with `text, score, source, doc_id, metadata, distance`. |
| Reasoning Agent | `agents/reasoning_agent.py:80-148` | `ReasoningAgent.reason` | Inputs: `label, confidence, passages`. Prompt: `SYSTEM_PROMPT:24-33` (cite [1]..[k], never invent chemicals). Fallback: extractive summary `57-77` if no LLM key (`config.llm_is_configured`). Empty passages → escalate (`92-101`). Graceful LLM failure → fallback (`128-140`). |
| Orchestrator | `agents/orchestrator.py:57-430` | `Orchestrator.process_image:98-157`, `_determine_action:159-188` | Thresholds `CONFIDENCE_HIGH=0.85, CONFIDENCE_LOW=0.55, MARGIN_THRESHOLD=0.15, AGREEMENT_DISAGREEMENT=0.5` (`68-71`). Tiers: Tier 3 escalate `c<0.55 ∨ a<0.5 ∨ ¬quality_ok` (`168-173`), Tier 1 direct `a==1 ∧ c≥0.85 ∧ margin≥0.15 ∧ quality_ok` (`179-185`), Tier 2 retrieve+reason else (`188`). Healthy shortcut: `label.endswith("___healthy")` (`85-87`, `138-144`). Decision logging `OrchestratorDecision:36-52` → `_log_decision:403-413` → `flush_log:415-426`. |
| Evaluator | `agents/evaluate.py:39-288` | `Evaluator.evaluate:57-213` | Metrics `EvaluationReport:19-36`: `classifier_accuracy, macro_f1, correct_action_rate, harmful_action_rate, naive_baseline_harmful_rate, human_intervention_rate, direct/retrieve rates, ablation_disagreement_caught`. Correct-action logic: `direct+correct ∨ retrieve ∨ escalate+wrong` (`121-131`); harmful = `direct ∧ wrong` (`134-135`); naive = `¬correct` (`138-139`). Per-class + macro-F1 `_macro_f1:215-252`. Coverage `generate_coverage_report:254-288`. |

**Notes on original vs revised routing:** Original submission (`research_paper/template_final.tex:183-204` Alg.1) used `DirectLookup` for high-confidence (skip retrieval) and `a` not present. Revised (`review/final_template_orignal.tex:183-202` Alg.1) changes to grounded retrieval for every disease diagnosis (both tiers run `RetrievalAgent → ReasoningAgent`, with caveats). Code at `orchestrator.py:214-268` already implements grounded direct (`_handle_direct:228-232` calls `retrieve_and_reason`), matching revised paper.

## 2. Implementation Details — Evidence for Revision Claims

- **Dataset split:** described in `agents/main.py:??` (not shown) but referenced in `research_paper/template_final.tex:212-213` as prior FL split. Test sizes 114 (PlantVillage) and 190 (OOD) match `agents/evaluate.py` expectations.
- **Vision backbone:** `vision_agent.py:_ResNet18Wrapper:84-101` loads 38-class ResNet-18, ImageNet norm `129-130`, preprocess 224×224 `132-158`. Supports single or ensemble (N=1 or 3).
- **Knowledge base:** `data/output/raw_docs/` + FAISS index at `config.FAISS_INDEX_DIR`; reported 125 docs / 16,706 chunks in revised tex — verify against `rag/rag/index_corpus.py` indexing.
- **Retrieval index:** `bge-small-en-v1.5` (`retrieval.py:58`), FAISS `similarity_search_with_score`.
- **Episodic logging:** `orchestrator.py:96`, `403-426` writes `decision_log*.json` used for offline eval.

## 3. Gaps Explicitly Called Out by Reviewers (Code Lens)

| Gap | Code Location | Why it triggers reviewer comment |
|---|---|---|
| Thresholds empirically chosen, no sensitivity | `orchestrator.py:68-71` hard-coded, `evaluate.py:150-158` only counts disagreements | R3-2/R4-3 |
| No Wilson CIs | `evaluate.py:161-177` computes rates without intervals | R3-1/R4-8/R4-11 |
| OOD N=190 small | `evaluate.py:69` total_images, per-class dict sparse | R3-3 |
| Retrieval only coverage, not precision | `evaluate.py:254-288` coverage only; `retrieval.py:85` scores unused for eval | R4-4 |
| No expert validation | `reasoning_agent.py:24-33` system prompt + `orchestrator.py:374-386` escalate flag not evaluated | R4-5 |
| No latency numbers | `orchestrator.py:98-157` no timers | R4-14 |
| No failure/hallucination example | `reasoning_agent.py:57-77` fallback but not showcased | R4-15 |
| No ablation | `evaluate.py:183-210` ablation note but only disagreement catch; no disable-agent comparison | R4-9 |

## 4. Directories & Data

- `model/class_names.json` — 38 classes, 14 healthy (`___healthy`).
- `data/output/raw_docs/` — curated docs for RAG.
- `rag/` — embedding/index code (`rag/rag/index_corpus.py`, `app/embeddings/factory.py`, `app/vectorstore/faiss_store.py`).
- `agents/config.py` — `FAISS_INDEX_DIR`, `RETRIEVAL_TOP_K`, `llm_is_configured()`, `RAG_PROJECT_ROOT`.

## 5. Quick Runbook (for experiment tasks)

```bash
# Evaluate existing decision logs (no LLM needed)
python3 -m agents.main --mode mock --seed 0 --out muse_review/logs_mock.json

# Coverage check
python3 -m agents.coverage_check  # or agents.coverage_check:main

# Threshold sweep (proposed script to create: muse_review/threshold_sweep.py)
python3 muse_review/threshold_sweep.py --log agents/decision_log.json --vary high low margin agree
```

All paths are relative to repo root `/Users/lovepreet/Desktop/vibe_projects/agriculture_agentic_ai`.

