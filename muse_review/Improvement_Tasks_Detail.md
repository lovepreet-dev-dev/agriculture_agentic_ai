# Improvement Tasks Detail — Paper ID 982
## Agentic AI for Plant Disease Diagnosis: Autonomous Multi-Agent Framework Integrating Vision, Retrieval, and Reasoning

**Directory:** `muse_review/` (isolated; no original files modified)  
**Source audit:** `review/Reviewer_Issues_Tracker.docx` (21 issues), `review/final_template_orignal.tex:1-362` (revised draft), `research_paper/template_final.tex:1-413` (original submission), `research_paper/agentic_AI_review (2).pdf` (original PDF), `review/agentic_AI_review_orignal.pdf` (revised PDF), `review/reviews paper (1).pdf` (scanned — text extraction returned 0 chars, needs OCR), `agents/*.py` (pipeline code)

**Legend:** ✓ Done = implemented in `review/final_template_orignal.tex` | ◐ Partial = partially addressed | ✗ Not Yet = planned, not yet done | Tier 3 = requires new experiment or data collection

**Overall tally from tracker:** 8 ✓ Done, 5 ◐ Partial, 8 ✗ Not Yet (tracker labels as Tier 3)

---

## How to use this document

Each task lists: **Reviewer verbatim** → **Current state** (file:line evidence) → **What “Done” means** → **Exact next steps** (what we have to do) → **Acceptance criteria / evidence for rebuttal** → **Effort & dependencies**. For ✓ Done tasks, “next steps” is verification/rewording polish. For ✗/◐ tasks, it is concrete experiment or manuscript edit.

---

## Reviewer #1 — Recommendation: Accept | Technical: High, Novelty: High, Documentation: High

### R1-1 ✓ Done — Cite Sounkhla et al. 2026 ResNet-Transformer Hybrid

- **Issue:** *Consider citing recent related work on transformer-based plant disease classification: Sounkhla, I., Dhir, R., Gupta, S., & Chouhan, S. A. (2026). ResNet-Transformer Hybrid Model for Classifying Apple Leaf Diseases. Procedia Computer Science, 283, 5329-5335.*
- **Current state:** Added to `references.bib` (check that bib entry exists — in `review/final_template_orignal.tex` it has not yet been added as a `\bibitem`; the Table I row is currently *missing* in `review/final_template_orignal.tex:78-104` — Table I there lists only Mohanty, He, Lewis, Acharya, Bandi, Hosseini with 6 rows; the Sounkhla row described in tracker is NOT present in either `review/final_template_orignal.tex` or `research_paper/template_final.tex`). This is a discrepancy: tracker says “new row in Table I” but the actual `.tex` does not contain it.
- **What Done should be:**
  1. `\bibitem{r_sounkhla2026}` entry in `thebibliography` or `references.bib` with exact metadata.
  2. New row in `Table I` `tab:related_work_summary` (now at `review/final_template_orignal.tex:82-85`) with columns: Study=Sounkhla et al. 2026, Method=ResNet-Transformer hybrid for apple leaf diseases, Dataset=AppleLeaf (?) / setting, Reported result=accuracy, Limitation=“Vision backbone only; no retrieval/reasoning/deferral; substitutable backbone for Vision Agent”.
  3. Sentence in Related Work summary paragraph `review/final_template_orignal.tex:106` (“The comparison highlights...”) referencing Sounkhla.
- **What we have to do:**
  - Add bib entry; insert table row (7th row before `\bottomrule`); add one sentence: “Sounkhla et al. [r_sounkhla2026] represents a complementary vision-backbone improvement that could replace the ResNet-18 in the Vision Agent without altering the agentic layer.” Verify table still fits `table*` width (adjust `\tabcolsep` if needed).
- **Acceptance:** `grep -n sounkhla muse_review/final_template_revised.tex` returns 2 hits (bib + table).
- **Effort:** 15 min, no experiment.

### R1-2 ◐ Partial — Minor Proofreading for Language Consistency

- **Issue:** *Minor proofreading is recommended to improve language consistency and presentation quality.*
- **Current state:** `review/final_template_orignal.tex:52` abstract, `60-68` introduction were reworded while fixing other comments. A full pass not done per tracker note.
- **What we have to do:**
  1. Consistent terminology: “Orchestrator” vs “orchestrator”, “RAG” expanded at first use, “OOD” defined at first use (`research_paper/template_final.tex:46` already defines but check), “human-in-the-loop” hyphenation, “bounded autonomy” phrasing (`review/final_template_orignal.tex:52-53`).
  2. Fix line-length/overfull hbox warnings (long abstract line `review/final_template_orignal.tex:53` is 2000+ chars).
  3. Run `aspell`/`LanguageTool` over `muse_review/final_template_revised.tex`; keep `IEEEtran` style (no Oxford commas inconsistency).
  4. Ensure British vs American English consistent (e.g., “behaviour” vs “behavior” — paper uses American).
- **Acceptance:** No spelling warnings, consistent acronym expansion on first appearance.
- **Effort:** 1–2 hours.

---

## Reviewer #3 — Recommendation: Accept | Technical: Average, Novelty: Above Average, Documentation: Above Average

### R3-1 ✓ Done (with caveat) — Statistical Significance / Confidence Intervals for Agentic Metrics

- **Issue:** *Evaluation of the agentic layer is relatively limited. While classification accuracy and proposed agentic metrics are reported, there is no statistical significance analysis or confidence intervals to demonstrate robustness of observed improvements.*
- **Current state:** Tracker says Wilson 95% CIs added to all 4 results tables (accuracy Tables III/V, harmful-action Tables IV/VI) and a “Statistical robustness” paragraph interpreting non-overlapping CIs. **But** `review/final_template_orignal.tex:243-321` tables currently have **no CI columns** — they show only `Accuracy | Macro-F1` and `Correct | Harmful | Intervention | Naive`. The paragraph described is also absent (results end with “Grounding and coverage” at `review/final_template_orignal.tex:347`). Code `agents/evaluate.py:161-177` computes rates without CIs.
- **What Done should be:**
  1. Wilson interval: for proportion `p = k/N`, CI = `(p + z²/2N ± z*√(p(1-p)/N + z²/4N²)) / (1+z²/N)`, `z=1.96`.
  2. Tables: add column `95% CI` or show as `99.1 [95.5–99.9]` for each accuracy/harmful rate. For PlantVillage N=114, OOD N=190.
  3. Paragraph after `Table VI` (now `review/final_template_orignal.tex:328`) : “Statistical robustness — OOD accuracy differences between FedAvg/FedOpt/FedMA/Ensemble have overlapping CIs at this N, so not statistically distinguishable; harmful-action reduction vs naive baseline has non-overlapping CIs, confirming safety improvement is robust.”
- **What we have to do:**
  - Edit `agents/evaluate.py` to compute Wilson CIs for each rate; update `muse_review/final_template_revised.tex` tables III–VI to include CI brackets.
  - Add robustness paragraph (4–5 sentences) with exact CI values computed from actual data (`data/output` not needed; use table values).
- **Acceptance:** Each table row shows CI; paragraph explicitly states overlapping vs non-overlapping interpretation.
- **Effort:** 30–45 min (no new data collection; compute from existing counts).

### R3-2 ✗ Not Yet — Threshold Sensitivity Analysis (0.85, 0.55, 0.15, 0.5)

- **Issue:** *Threshold values used by Orchestrator appear empirically selected, but manuscript does not provide sensitivity analysis or justification.*
- **Current state:** Hard-coded in `agents/orchestrator.py:68-71` (`CONFIDENCE_HIGH=0.85`, `CONFIDENCE_LOW=0.55`, `MARGIN_THRESHOLD=0.15`, `AGREEMENT_DISAGREEMENT=0.5`) and in `review/final_template_orignal.tex:184` `Algorithm 1`. Text at `review/final_template_orignal.tex:211` says “These thresholds bias the system toward caution” but gives no justification or sweep.
- **What we have to do:**
  1. Sweep each threshold independently `±0.05, ±0.10` while holding others fixed, re-running Orchestrator routing on existing `agents/decision_log*.json` predictions (no new inference needed — reuse `agents/main.py` logic with modified thresholds).
  2. For each sweep point compute Correct-action / Harmful-action / Human-intervention rates (via `agents/evaluate.py`).
  3. Plot 4 small line plots or one table: threshold value vs 3 rates; identify plateau region where current values lie.
  4. Manuscript: add sentence in `Implementation Details` (“Thresholds selected from validation-set operating point maximizing harmful-action reduction at ≤30% intervention”) + add figure or supplementary table with sensitivity.
- **Dependencies:** Existing decision logs sufficient; no LLM calls. Can be mocked via `python3 -c` loop over logs.
- **Acceptance:** Table/figure showing that varying thresholds trades off safety vs intervention, and current values sit at conservative knee.
- **Effort:** 2–3 hours (code + figure).

### R3-3 ✗ Not Yet — OOD Dataset Size (190 images, 5/class)

- **Issue:** *OOD dataset contains only 190 images (5 per class), relatively small for assessing generalization. Larger and more diverse field dataset would strengthen conclusions.*
- **Current state:** `review/final_template_orignal.tex:224` describes OOD set 190 images; `research_paper/template_final.tex:229-230` same. No larger dataset collected.
- **What we have to do (Tier 3 fallback if no time to collect data):**
  1. Add explicit **Limitations** subsection before Conclusion (`review/final_template_orignal.tex:350` → new Section `VI-C Limitations` or paragraph in Results): state `N=5 per class` limits per-class precision, Wilson CI width ~±X%, and that conclusions are aggregate not per-class.
  2. If time allows: collect 10–20 additional field images per class from smartphone field conditions (cluttered background, variable lighting) and re-evaluate; else propose as future work with protocol (data source, verification, annotation).
- **Acceptance:** Manuscript acknowledges sample-size limitation and its impact on CI width / generalizability claim.
- **Effort:** 30 min for text; days if new data collection.

### R3-4 ✓ Done — Acronym Consistency (OOD, RAG, FedAvg, FedOpt, FedMA)

- **Issue:** *Acronyms should be consistently defined at first appearance and checked for formatting consistency.*
- **Current state:** `review/final_template_orignal.tex:152-153` defines RAG via `\cite{r_lewis2020}`; OOD defined at `review/final_template_orignal.tex:224` (first use); FedAvg/FedOpt/FedMA defined at `review/final_template_orignal.tex:147`. Originally RAG appeared in Fig.1 diagram before prose definition — now fixed (diagram now at `review/final_template_orignal.tex:113-144` after Related Work).
- **What we have to do:** Verify in `muse_review/final_template_revised.tex` that `\acronyms` are defined on first occurrence and not re-defined later; spot-check `OOD Dataset`, `retrieval-augmented generation (RAG)`, `Federated Averaging (FedAvg)` etc.
- **Effort:** 10 min.

---

## Reviewer #4 — Recommendation: Accept | Technical: Above Average, Novelty: Above Average, Documentation: Average — 15 points

### R4-1 ◐ Partial — Novelty Distinction vs Existing Agentic & Plant-Disease Approaches

- **Issue:** *Novelty of multi-agent framework should be clearly distinguished from existing agentic AI and plant disease diagnosis approaches.*
- **Current state:** Table I `review/final_template_orignal.tex:78-104` includes limitation column but tracker notes it only adds Sounkhla row; broader agentic novelty argument not expanded.
- **What we have to do:**
  1. Expand Related Work summary paragraph `review/final_template_orignal.tex:106` to 4–5 sentences contrasting: classifier-only (Mohanty), backbone-only (He), NLP RAG (Lewis), agentic surveys (Acharya/Bandi/Hosseini) vs this work’s integration (safety-aware routing + grounded treatment + deferral).
  2. In rebuttal letter (not tex): explicitly list 3 novelty axes — perception→action loop, safety metrics, evidence grounding — with table row references.
- **Effort:** 30 min.

### R4-2 ✗ Not Yet — Precise Agent Roles, Interactions, Decision Boundaries

- **Issue:** *Roles, interactions, and decision boundaries of the four agents should be described more precisely.*
- **Current state:** Section III (`review/final_template_orignal.tex:108-202`) describes each agent in its own subsection but lacks explicit I/O contracts and decision boundaries. Code has precise contracts: `agents/vision_agent.py:VisionPrediction` (label, probs, agreement), `agents/confidence_quality_agent.py:QualityAssessment`, `agents/orchestrator.py:_determine_action:159-188`.
- **What we have to do:**
  1. Add a small interaction diagram or table (besides Fig.1) listing per agent: Input → Output → Called when → Thresholds applied.
  2. In text, add one sentence per agent subsection stating decision boundary: e.g., “Vision Agent never escalates; Orchestrator escalates if `c < 0.55 ∨ a < 0.5 ∨ q≠good` (`Algorithm 1:6`).”
  3. Reference `Algorithm 1` explicitly from each agent subsection.
- **Effort:** 45 min.

### R4-3 ✗ Not Yet — Quantitative Justification for Confidence/Quality Escalation Criteria

- **Issue:** *Criteria used by Confidence and Image-Quality Agent to trigger human intervention should be quantitatively justified.*
- **Current state:** Same as R3-2; thresholds not justified.
- **What we have to do:** Same sweep as R3-2, but additionally report per-trigger attribution: fraction escalated due to `c < 0.55` vs `a < 0.5` vs `q≠good` vs margin (`agents/orchestrator.py:168-173`). Add stacked bar or table.
- **Effort:** Bundled with R3-2 (+30 min).

### R4-4 ✗ Not Yet — Retrieval Agent Accuracy/Reliability/Quality

- **Issue:** *Retrieval Agent should be evaluated for accuracy, reliability, quality of evidence retrieved.*
- **Current state:** Only coverage reported (`review/final_template_orignal.tex:347`: “all 38 classes return strongly relevant passages” with threshold `cos ≥0.75` defined at `review/final_template_orignal.tex:238` / `research_paper/template_final.tex:243`). No precision@k, no relevance annotation. Code `agents/retrieval.py:70-96` returns `Passage.score` (cosine sim) but no eval.
- **What we have to do:**
  1. Sample 30–50 retrieved passages (5 classes × ~8 queries), manually label relevance (relevant / partially / irrelevant) against query disease.
  2. Compute `Precision@5` and `Relevance rate @ cos≥0.75`; report distribution of scores.
  3. Manuscript: add Table or sentence in Results: “Manual rating of N=XX passages gave Precision@5 = YY%”.
  4. Code: use `agents/coverage_check.py` as template; log scores to `muse_review/retrieval_quality_sample.json`.
- **Effort:** 3–4 hours (manual labeling dominates).

### R4-5 ✗ Not Yet — Validate Treatment Recommendations Against Expert Agreement (Hardest)

- **Issue:** *Reasoning process to generate treatment recommendations should be validated against expert-agreed recommendations.*
- **Current state:** Grounding check at `review/final_template_orignal.tex:236` is qualitative “whether each factual claim is supported by cited passage” without inter-annotator agreement; no external plant-pathology expert involved. `agents/reasoning_agent.py` prompts LLM with fixed template requiring citations.
- **What we have to do:**
  1. Define protocol: sample 20–30 Reasoning outputs (from `agents/reasoning_agent.py` + `agents/orchestrator.py:retrieve_and_reason:346-399`), give to domain expert (plant pathologist) blinded to retrieved passages, ask to rate: correct / minor error / major error / unsafe.
  2. Report % expert agreement, Cohen’s κ if 2 raters.
  3. Manuscript fallback if no expert available before deadline: state as limitation, propose protocol in Future Work (`review/final_template_orignal.tex:352`), and add structured logging of reasoning outputs for future audit (`agents/orchestrator.py:_log_decision:403-413` already logs).
- **Effort:** Days (expert recruitment); text fallback 30 min.

### R4-6 ✓ Done — Moderate “Autonomous” Claim

- **Issue:** *Claim of “autonomous” diagnosis should be moderated given low-confidence cases deferred to human.*
- **Current state:** Abstract `review/final_template_orignal.tex:52-53` now says “autonomous, explainable, and safety-aware diagnostic advisory system” with “bounded autonomy within human-in-the-loop envelope” implied; tracker claims reworded Abstract/Intro/Conclusion to “bounded autonomy”. Check `review/final_template_orignal.tex:52`, `65`, `355` for exact phrasing.
- **What we have to do in `muse_review/final_template_revised.tex`:** Ensure Abstract line 53, Introduction paragraph 3 (`60-68`), and Conclusion `355` use “bounded autonomy” or “human-in-the-loop envelope” not “fully autonomous”. Keep title as is (acceptable).
- **Effort:** 15 min.

### R4-7 ◐ Partial — PlantVillage Alone Not Field-Robust

- **Issue:** *Use of PlantVillage alone may not adequately demonstrate robustness under real-world field conditions (lighting, background, severity, quality).*
- **Current state:** OOD web set 190 images provides some generalization (`review/final_template_orignal.tex:224`, `323`, `336`) but not true field conditions (occlusion, severe disease, smartphone capture). Tracker marks partial.
- **What we have to do:** Same as R3-3: add Limitations paragraph acknowledging controlled backgrounds vs cluttered field images, cite `agents/confidence_quality_agent.py` blur/brightness checks as partial mitigation at inference, but not evaluated on field data.
- **Effort:** 20 min.

### R4-8 ✓ Done — Support 99.1% Accuracy with More Metrics & Statistical Significance

- **Issue:** *Reported 99.1% accuracy should be supported with additional metrics and statistical significance.*
- **Current state:** Tracker says Wilson CIs added to 99.1% figure; but again tables lack CI display. Macro-F1 already in tables (`review/final_template_orignal.tex:249-256`). Need CI.
- **What we have to do:** Same as R3-1: add CI to accuracy values `99.1 [X–Y]` in Tables III/V; ensure abstract 99.1% also has CI if space.
- **Effort:** Bundled with R3-1.

### R4-9 ✗ Not Yet — Comprehensive Ablation Study (Highest-Value Remaining)

- **Issue:** *Ablation study needed to quantify individual contribution of vision, retrieval, reasoning, orchestration.*
- **Current state:** Code supports disabling agents: `agents/orchestrator.py:83-94` has optional `retriever` and `reasoning_agent` (Phase 1 stub vs Phase 3 real), and `_determine_action` can be bypassed. But no ablation results reported. `agents/evaluate.py:33-34` already computes `ablation_disagreement_caught` for ensemble.
- **What we have to do:**
  1. Run 4 configurations on same test sets (PlantVillage 114 + OOD 190): Full system | No Confidence/Quality (always act) | No Retrieval (direct lookup) | No Reasoning (retrieval but no LLM) .
  2. Report harmful-action rate and grounding quality per config; show that removing Confidence/Quality raises harmful rate to naive baseline, removing Retrieval hurts grounding.
  3. Manuscript: add Table `Ablation Results` in Results (`review/final_template_orignal.tex:330-347` area) and one paragraph interpretation.
- **Effort:** 4–6 hours (re-running pipeline).

### R4-10 ✓ Done — Formally Define Agentic Metrics & Baselines

- **Issue:** *Agentic metrics (correct-action, harmful-action, human-intervention) should be formally defined and evaluated against meaningful baselines.*
- **Current state:** Defined in Equations 1–3 `review/final_template_orignal.tex:228-232`, evaluated vs naive baseline in Tables IV/VI `review/final_template_orignal.tex:261-319`. Tracker says added explicit sentence pointing back to Eqs when introducing tables.
- **What we have to do:** Ensure sentence exists in Results intro `review/final_template_orignal.tex:241` (“The safety tables (Tables IV/VI) are scored per Eqs. (1)–(3) against the naive classifier-only baseline”); if missing, add it in `muse_review/final_template_revised.tex`.
- **Effort:** 10 min.

### R4-11 ✓ Done — Careful Harmful-Action Rate Evaluation (Safety)

- **Issue:** *Harmful-action rate requires particularly careful evaluation because incorrect treatment may have agricultural/economic consequences.*
- **Current state:** Same fix as R3-1/R4-8 but focused on harmful-action CI; tracker says added Wilson CIs to Tables IV/VI and robustness paragraph confirms non-overlapping vs baseline.
- **What we have to do:** Same as R3-1; emphasize in robustness paragraph that harmful-action reduction is the primary safety claim.
- **Effort:** Bundled.

### R4-12 ✓ Done — Federated Experiments Connection to Multi-Agent Framework

- **Issue:** *Federated learning experiments (FedAvg/FedOpt/FedMA) should be better connected to multi-agent framework and practical motivation clarified.*
- **Current state:** New paragraph in `review/final_template_orignal.tex:147` (Vision Agent) + `review/final_template_orignal.tex:330` in Results (“agreement among three member models provides more confidence...”) explains privacy/data-ownership motivation and links diversity to agreement-ratio signal. Original `research_paper/template_final.tex:148-150` was terse, single-model focused.
- **What we have to do:** Verify paragraph exists in `muse_review/final_template_revised.tex:147-150`; keep it; ensure `FedAvg [13]`, `FedOpt [14]`, `FedMA [15]` citations appear.
- **Effort:** 10 min (verify).

### R4-13 ◐ Partial — External/Cross-Dataset Evaluation

- **Issue:** *System should be evaluated on external or cross-dataset data to establish generalization beyond PlantVillage.*
- **Current state:** OOD web set 190 images serves this purpose (`review/final_template_orignal.tex:224`), discussed in Results (`323`, Fig.2/3). Tracker notes additional external datasets not added.
- **What we have to do:** Add Limitations note; if possible evaluate on second small external set (e.g., PlantDoc or field-collected) even 20 images to show trend; else state as future work.
- **Effort:** 30 min text; hours if new dataset.

### R4-14 ✗ Not Yet — Computational Overhead / Latency / Resources

- **Issue:** *Computational overhead, latency, resource requirements introduced by retrieval and multi-agent reasoning should be reported.*
- **Current state:** Text at `review/final_template_orignal.tex:347` claims “lower average inference cost than design that reasons over every input” but no numbers. No timers in code (`agents/orchestrator.py`, `agents/retrieval.py`, `agents/reasoning_agent.py` lack timing).
- **What we have to do:**
  1. Instrument `agents/orchestrator.py:98-157` `process_image` with `time.perf_counter()` per stage: Vision, Confidence/Quality, Retrieval (FAISS `k=5`), Reasoning (LLM call). Log to decision record.
  2. Report table: `Mean ± SD latency (ms)` per stage for direct vs retrieve-and-reason paths; note FAISS index size (16,706 chunks, embedding dim), LLM provider/tokens.
  3. Manuscript: add Table in Implementation Details or Results (e.g., after `review/final_template_orignal.tex:347`).
- **Effort:** 1–2 hours (instrument + 20-image benchmark).

### R4-15 ◐ Partial — Failure Cases, Hallucination, Outdated Evidence, Safeguards

- **Issue:** *Paper should discuss potential failure cases, hallucinated recommendations, outdated retrieved evidence, safeguards against unsafe treatment.*
- **Current state:** Grounding paragraph `review/final_template_orignal.tex:347` describes hallucination-averse behavior (“declined to fabricate treatment when evidence lacked treatment guidance”). No worked example with case ID/image or genuine failure case.
- **What we have to do:**
  1. Select one successful grounding example (with cited passages) and one failure/abstention example from `agents/decision_log*.json` (e.g., escalated due to empty retrieval `agents/orchestrator.py:350-366`).
  2. Add figure or boxed example in Results: query label, retrieved passage snippet, reasoning output, safeguard triggered.
  3. Discuss outdated evidence risk (static KB 125 docs) and mitigation (periodic KB refresh, date-filtering, citation requirement in prompt `agents/reasoning_agent.py:??`).
- **Effort:** 1–2 hours.

---

## Summary Table

| # | Reviewer | Issue | Status | Effort | Requires Experiment? |
|---|---|---|---|---|---|
| R1-1 | #1 | Sounkhla 2026 citation | ✓ Done (discrepancy: row not in tex) | 15m | No |
| R1-2 | #1 | Proofreading | ◐ Partial | 1-2h | No |
| R3-1 | #3 | Statistical CIs | ✓ Done (but not in tex) | 45m | No (compute) |
| R3-2 | #3 | Threshold sensitivity | ✗ | 2-3h | Yes (sweep) |
| R3-3 | #3 | OOD size 190 | ✗ | 30m / days | Optional data |
| R3-4 | #3 | Acronyms | ✓ Done | 10m | No |
| R4-1 | #4 | Novelty distinction | ◐ Partial | 30m | No |
| R4-2 | #4 | Agent roles/boundaries | ✗ | 45m | No (text) |
| R4-3 | #4 | Quality trigger quant | ✗ | +30m | Yes (sweep) |
| R4-4 | #4 | Retrieval quality | ✗ | 3-4h | Yes (annotate) |
| R4-5 | #4 | Expert validation | ✗ | days | Yes (expert) |
| R4-6 | #4 | “Autonomous” moderation | ✓ Done | 15m | No |
| R4-7 | #4 | Field robustness | ◐ Partial | 20m | No (text) |
| R4-8 | #4 | 99.1% with CIs | ✓ Done | — | — |
| R4-9 | #4 | Ablation study | ✗ | 4-6h | Yes (re-run) |
| R4-10 | #4 | Agentic metrics defined | ✓ Done | 10m | No |
| R4-11 | #4 | Harmful-action careful | ✓ Done | — | — |
| R4-12 | #4 | Federated motivation | ✓ Done | 10m | No |
| R4-13 | #4 | Cross-dataset | ◐ Partial | 30m | Optional |
| R4-14 | #4 | Latency/resources | ✗ | 1-2h | Yes (bench) |
| R4-15 | #4 | Failure/hallucination | ◐ Partial | 1-2h | Yes (example) |

**Priority for camera-ready if time-limited:** R1-1, R3-1/R4-8/R4-11 (CIs), R1-2, R4-12, R4-1 (text-only, high reviewer visibility) → then R3-2/R4-3, R4-2, R4-14 (small experiments) → then R4-4, R4-9 (larger experiments) → R4-5 (longest).

---

## Evidence Gaps vs Tracker Claims

- Tracker claims Sounkhla row and Wilson CIs are “✓ Done” in `final_template.tex`, but **actual tex files contain neither** — the revised copy in `muse_review/` will finally implement them, making tracker accurate.
- `reviews paper (1).pdf` is scanned; its reviewer verbatim may differ slightly from tracker paraphrases — recommend OCR before final rebuttal to avoid missing nuance.

---

## What will be in `muse_review/final_template_revised.tex`

See separate file. It starts as copy of `research_paper/template_final.tex:1-413` (original submission, with XX.X% placeholders) and applies all ✓ Done edits plus correctness fixes, without touching originals. Placeholder values replaced with actual results from `review/final_template_orignal.tex:243-321` (99.1%, 86-99.1% etc.) and CI scaffolding added.

