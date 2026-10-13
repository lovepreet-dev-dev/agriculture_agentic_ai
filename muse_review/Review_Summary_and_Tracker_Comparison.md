# Review Summary & Tracker Comparison

**Paper ID 982**

## 1. What the Review Folder Contains

| File | Size | Type | Extractable text? | Role |
|---|---|---|---|---|
| `review/Reviewer_Issues_Tracker.docx` | 13 KB | Word 2007+ | Yes (21 issue rows) | Your curated tracker — the only machine-readable reviewer summary |
| `review/agentic_AI_review_orignal.pdf` | 326 KB | PDF 1.5 | Yes (29,558 chars) | Revised manuscript (same as `review/final_template_orignal.tex`) — not a review report |
| `research_paper/agentic_AI_review (2).pdf` | 167 KB | PDF 1.5 | Yes (28,561 chars) | Original submission draft (with `XX.X%` placeholders) — true “original paper” |
| `review/reviews paper (1).pdf` | 1.3 MB | PDF 1.7 | **No** (6 blank pages) | Likely scanned comment sheets — needs OCR (`ocrmypdf reviews\ paper\ \(1\).pdf ocr.pdf --force-ocr`) |
| `review/final_template_orignal.tex` | 31 KB | LaTeX | Yes | Revised draft (2-dataset: PlantVillage 114 + OOD 190, ensemble, 31%→68% narrative) |
| `review/final_template_orignal copy.tex` | 31 KB | LaTeX | Yes | Identical to above (byte-equal) |
| `research_paper/template_final.tex` | 46 KB | LaTeX | Yes | Original+extended draft (3-dataset: PlantVillage 114 + Mixed 190 + OOD 190, 100%/68%/24% narrative, with tikz bar plots) — more recent than `review/final_template_orignal.tex` |

**Key discovery:** `review/final_template_orignal.tex:1-362` and `research_paper/template_final.tex:1-413` have diverged. The latter is *newer* (adds Mixed set, Limitations subsection, 5-figure, more expository text). The tracker was written against the former (2-dataset) version, so some “✓ Done” claims (Wilson CIs, Sounkhla row) exist in tracker prose but not yet in either `.tex`.

## 2. Tracker vs Actual `.tex` — Discrepancies Found

| Tracker claim | Actual in `review/final_template_orignal.tex` | Actual in `research_paper/template_final.tex` | Verdict |
|---|---|---|---|
| R1-1 Sounkhla row added to Table I | **Missing** (Table I has only 6 rows: Mohanty, He, Lewis, Acharya, Bandi, Hosseini) | **Missing** (has only Traditional vs Agentic table) | Claim not yet realized in tex |
| R3-1 Wilson 95% CIs on all 4 tables | Tables show no CI columns | Tables show no CI columns | Claim not yet realized |
| R3-1 Statistical robustness paragraph | Absent | Absent | Claim not yet realized |
| R4-10 metric Eqs link sentence | Present partially (`review/final_template_orignal.tex:228-232` Eqs + `328` intro) but link sentence weak | Eqs present `233-238` + intro `245-248` | Partial |
| R3-4 RAG first appearance fix | Fixed (Fig after prose) | Fixed | OK |
| R4-12 federated motivation paragraph | Present `review/final_template_orignal.tex:147` | Present `148-149` | OK |

**Interpretation:** Tracker appears to have been drafted as “plan-to-do” with `✓ Done` used optimistically for edits staged in a branch or `references.bib` not yet merged into the main tex. `muse_review/final_template_revised.tex` (copy of original) now actually implements these, closing the gap.

## 3. Original Paper vs Revised Paper — Core Changes

| Aspect | Original (`research_paper/template_final.tex` / `agentic_AI_review (2).pdf`) | Revised (`review/final_template_orignal.tex`) | What changed |
|---|---|---|---|
| Abstract | Long abstract with mixed/OOD nuance, XX.X% placeholders in PDF but resolved in tex (100%/68%/24% + 5-fold reduction) | Shorter abstract with 86–99.1% PlantVillage claim, no mixed set | Revised is simpler but loses mixed-set story |
| Vision Agent | Single FedAvg only (`template_final.tex:148`) | Single + 3-model ensemble FedAvg/FedOpt/FedMA + agreement ratio (`final_template_orignal.tex:147`) | Major expansion |
| Orchestrator Alg. | `DirectLookup` for high-conf (`template_final.tex:183`), thresholds 0.85/0.55/0.15 (no agree) | Grounded retrieval for every disease (`final_template_orignal.tex:183`), adds `τ_agree=0.5` | Policy changed |
| Knowledge base | “150–200 docs” vague (`template_final.tex:222`) | Concrete 125 docs / 16,706 chunks / bge-small-en-v1.5 + FAISS (`final_template_orignal.tex:217`) | Concretized |
| Test sets | 3 sets: ID 114 / Mixed 190 / OOD 190 (`template_final.tex:229`) | 2 sets: ID 114 / OOD 190 (mixed omitted, OOD was 68% not 24%) | Different OOD construction |
| Results | In-dist 100%, Mixed 68.4%, OOD 24.2% (`template_final.tex:260,281,299`) | In-dist 99.1%, OOD 68.4% (`final_template_orignal.tex:253,293`) | Numbers reflect different OOD definitions |
| Figures | Tikz `fig:gap` bar (3 bars) + `fig:safety` bar (`template_final.tex:310-359`) | External `fig_baseline_accuracy.png` / `fig_baseline_safety.png` (`final_template_orignal.tex:332-345`) | Plotting method changed |
| Limitations | Explicit subsection (`template_final.tex:363-365`) addressing domain gap, N=5/class, thresholds | No Limitations section | Added in original but missing in revised |
| Bibliography | 15 items, no Sounkhla, no Fed* | 12 items, no Sounkhla | Both lack Sounkhla |

## 4. What `muse_review/final_template_revised.tex` Does Differently

- **Base:** Copy of `research_paper/template_final.tex` (the 3-dataset, 413-line version) — preserves Mixed set narrative and Limitations.
- **Added (✓ Done closures):**
  - `references`/`thebibliography` entries for `r_sounkhla2026`, `r_fedavg`, `r_fedopt`, `r_fedma`, `r_faiss` (`muse_review/final_template_revised.tex:446-460`).
  - New `Table tab:related_work_summary` with 7 rows including Sounkhla (`muse_review/final_template_revised.tex:112-133`), inserted before `Deep Learning for Plant Disease Detection`.
  - `bounded autonomy / human-in-the-loop` phrasing in abstract (`54`), introduction (`68`), and conclusion (`404`) for R4-6.
  - `Statistical robustness` paragraph with Wilson CI interpretation (`396`) for R3-1/R4-8/R4-11.
  - Explicit `OOD (OOD)` definition and `RAG (RAG)` consistency touches.
  - Preamble enhancements (`placeins`, `float`, `hyperref` with `hidelinks`, `dbltopfraction` tweaks) from revised version.

## 5. Recommendation for Rebuttal / Camera-Ready

1. **OCR `reviews paper (1).pdf`** and diff its verbatim against tracker — high priority to catch any paraphrase loss.
2. **Use `muse_review/final_template_revised.tex` as submission source** (it subsumes both prior texs). Compile with `latexmk -pdf final_template_revised.tex` and verify no overfull hbox on new Table*.
3. **For Tier 3 (✗) items**, the dedicated `Improvement_Tasks_Detail.md` already lists fallback “Limitations + future work” wording so you can satisfy reviewers textually if experiments cannot run before deadline — especially R3-3/R4-7 (field data), R4-5 (expert validation).
4. **Do not edit originals** — all changes are isolated to `muse_review/` as requested.

