# Supervisor-Review Audit — Lightweight PC-STE

Prepared for review by Dr Erick Giovani Sperandio Nascimento and Marco Aurélio da Silva Cruz.

Date: 2026-09-21
Scope: three editorial passes, in order — scientific-consistency polish, generation and
integration of the five remaining main-paper figures, and a bibliography/front-matter
release pass. No training, no TEST rerun, no inference, no regenerated results, no frozen
artifact modified.

- Sections A–G and J record the editorial polish pass; its textual changes are in
  `manuscript/SUPERVISOR_POLISH.diff` (4 files, 29 changed lines).
- Section H records the now-complete six-figure main-paper set.
- Section I records the bibliography and front-matter release pass.

---

## A. Major scientific consistency checks

| # | Check | Result | Notes |
|---|---|---|---|
| A1 | Dataset-level Macro-F1 table vs frozen `PER_DATASET_SUMMARY.csv` | **PASS** | All 4 datasets × (mean, SD, Δ) match to the printed 6 decimals. |
| A2 | Dataset-level Macro-AUC table vs frozen summary | **PASS** | All 4 datasets match exactly. |
| A3 | Macro-4 headline (F1 and AUC) vs `FINAL_SEALED_TEST_REPORT.md` | **PASS** | 0.934644±0.023258 → 0.955334±0.018393; Δ +0.020691±0.013509. |
| A4 | Nine matched cells table vs frozen `matched_cells.csv` | **PASS** | All 9 rows and Δ values match. |
| A5 | Wins/ties/losses 8/0/1 | **PASS** | Matches frozen aggregate. |
| A6 | Non-inferiority margin −0.02, exact one-sided *p* = 0.001953125 | **PASS** | Correctly framed as the primary confirmatory result. |
| A7 | Gated two-sided superiority *p* = 0.0078125 | **PASS** | Explicitly secondary in both Methods and Results; never the headline. |
| A8 | No AUC significance test invented | **PASS** | AUC is described as a descriptive secondary endpoint only. |
| A9 | Efficiency values vs `CORRECTED_EFFICIENCY_SUMMARY.json` | **PASS** | Parameters, FP32 size, FLOPs, all four latencies, both throughputs, GPU memory all verified. |
| A10 | Superseded `efficiency_v1` timings absent from claims | **PASS** | 2.213×/2.179×/91.850/93.327/11.811/13.409 ms appear **nowhere**. Only a qualitative "roughly 2.2× to roughly 1.9×" in the audit-trail paragraph, which is explicitly framed as a correction. |
| A11 | Q8 storage vs `Q8_FINAL_REPORT.md` §4 | **PASS** | 5,541,832 B → 1,600,387 B; −71.12% (3.46×); vs Full-S1 −83.29% (5.99×). |
| A12 | Q8 predictive result (`sim`) | **PASS** | 0.955334±0.018393 → 0.955334±0.018405; margin −0.01; *p* = 0.001953125. |
| A13 | **Q8 `sim` vs `cpu_dynamic` caveat** | **PASS (strengthened)** | Manuscript states the TEST result applies to `sim`, not the deployed artifact, in Methods, Results, Table 8 caption, Limitations, and Conclusion. |
| A14 | No GPU INT8 acceleration claim | **PASS** | Explicitly disclaimed in Methods, Results and Conclusion. |
| A15 | CWRU = 3-class frozen benchmark | **PASS** | Table 2 already stated `CWRU & 3`. No "4-class" statement exists anywhere. Section 4.3 reinforces "the three-class CWRU label space". |
| A16 | CWRU protocol: native 12 kHz only, TRAIN 0+1 hp / VAL 2 hp / TEST 3 hp | **PASS** | Stated in Methods and Table 2; explicitly "not an unseen-machine test". |
| A17 | Nine cells never described as independent datasets/test sets | **PASS** | Consistently "nine matched fold–seed cells". Figure 1 caption explicitly says the pool is "not a single independent test split". |
| A18 | No cross-machine / unseen-machine generalization claim | **PASS** | Denied in Methods, Table 2 caption, Limitations and Conclusion. |
| A19 | HIT decrease acknowledged | **PASS** | Section 4.2 explicitly uses it to block a "universal dominance" reading; Section 4.3 localizes it to class 1. |
| A20 | Validation-only checkpoint selection, TEST sealed | **PASS** | Stated in Methods §3.2, §3.6 and §3.4. |
| A21 | K1 primary, Q8 secondary | **PASS** | Q8 occupies 1 of 8 Results subsections and is labelled secondary in Abstract, Methods, Results, Limitations and Conclusion. |
| A22 | Results narrative order matches Section 19 of the review brief | **PASS** | 4.1 matched → 4.2 dataset → 4.3 class-level → 4.4 non-inferiority → 4.5 efficiency → 4.6 Q8 → 4.7 comparison → 4.8 limitations. |
| A23 | Section 4.3 does not duplicate Section 4.2 | **PASS** | 4.3 references the dataset-level scores rather than restating them; only the confusion percentages are new. |
| A24 | Equations: symbols defined, no unexplained constants | **PASS after fix** | See C2. |
| A25 | Bibliography integrity | **PASS** | 20 entries, 20 cited, 0 unused, 0 missing keys, 0 duplicates. |
| A26 | Dissertation residue in the manuscript | **PASS after fix** | See C3. (Apparent "thesis" hits are the substring in *hypothesis*.) |

---

## B. Overclaiming audit

Every occurrence of `first`, `novel(ty)`, `state-of-the-art`, `SOTA`, `outperform`,
`superior`, `generaliz*`, `foundation model`, `universal`, `all datasets`,
`edge deployment` and `best` was read in context.

**Result: no unsupported claim was found, and no claim needed softening.** The manuscript
was already written defensively. Every flagged token is either benign or an explicit
disclaimer:

| Token | Context | Verdict |
|---|---|---|
| `first` (×2) | "first averaged over windows", "first validation window" | Benign — not a priority claim. |
| `novelty` (×1) | "neither teacher–student compression nor multi-teacher distillation can be treated as an isolated novelty" | Disclaimer. |
| `outperform` (×1) | "do not imply that K1 must outperform Full-S1 for every dataset, fold, seed…" | Disclaimer. |
| `superior/superiority` (×5) | Always "non-inferiority experiment rather than a universal superiority claim", or the gated secondary test | Correctly subordinated. |
| `foundation model` (×1) | "PC-STE/K1 is **not** positioned as a larger or more general foundation model than these approaches" | Disclaimer. |
| `universal` (×4) | "not universal machinery generalization", "rather than universal dominance" | Disclaimers. |
| `edge deployment` (×1) | Attributed to BearingPGA-Net as the boundary the present work does *not* reach | Correct attribution. |
| `best` (×2) | `best.pt` filename; "best described as" | Benign. |

The research-gap paragraph uses "Less explored is…" rather than any "no previous work has…"
formulation. No wording changes were required in this category.

---

## C. Result corrections

**No manuscript number disagreed with a frozen artifact.** Every numerical value in the
prose and in all nine tables was verified against the sealed sources. The corrections below
are notational and editorial, not numerical.

| # | File | Correction | Justification |
|---|---|---|---|
| C1 | `sections/04_results_discussion.tex` | Results roadmap paragraph now includes the class-level/confusion step; Section 4.2 retitled from "Dataset-Level **and Class-Level** Performance" to "Dataset-Level Performance". | The roadmap and the 4.2 heading both predated the insertion of Section 4.3 and no longer described the section. Internal inconsistency introduced by the figure integration. |
| C2 | `sections/03_methodology.tex` | The non-inferiority margin symbol `$m=0.02$` was defined and then never used — the hypothesis and the shift were written with the literal `0.02`. Now `H_0: E[Δ] ≤ −m`, `H_1: E[Δ] > −m`, `Δ'_i = Δ_i + m`, with `m = 0.02` stated in the display. | Symbol defined but unused (review brief item 18). Numerically identical; no result changes. |
| C3 | `sections/03_methodology.tex` | "rather than the obsolete **dissertation-era** band count" → "rather than the superseded mixed-rate band count". | Dissertation residue in a publication manuscript (item 27). Meaning preserved. |
| C4 | `sections/03_methodology.tex` | "maximum validation **Macro-Domain F1**" → "maximum validation **Macro-4 Macro-F1** (recorded in the frozen artifacts as `macro_domain_f1`)". | Terminology standardization (item 6). **Verified safe:** `macro_domain_f1 == macro_4_macro_f1` in all 18 frozen per-cell reports. The frozen field name is retained for traceability. |
| C5 | `sections/04_results_discussion.tex` | The `sim` vs `cpu_dynamic` caveat now gives the MaFaulDa figure explicitly: "agreement falls to 95.0% on MaFaulDa, where the maximum absolute probability difference reaches 0.49", replacing the vague "with lower agreement on MaFaulDa". | Item 14 requires this caveat to be unambiguous. Values from `Q8_FINAL_REPORT.md` §7. |
| C6 | `tables/datasets_protocol.tex` | `X` columns changed to ragged-right. | The justified narrow cells produced a visible underfull box — the MaFaulDa row rendered as "Grouped&nbsp;&nbsp;&nbsp;&nbsp;by&nbsp;&nbsp;&nbsp;&nbsp;configura-/tion/severity". Formatting only; no content change. |
| C7 | `main.tex` | Removed `\usepackage{siunitx}`. | Verified programmatically that no siunitx macro (`\SI`, `\si`, `\num`, `\qty`, `\unit`, `\ang`, `\sisetup`, …) is used anywhere. Page count and rendering unchanged at 22 pages. Removes a build dependency that is not present in a default TeX Live 2023 install. |
| C8 | `main.tex` | Empty *Author Contributions*, *Data and Code Availability* and *Declaration of Competing Interest* blocks replaced with clearly marked italic editorial placeholders. | Per item 2, **no individual contributions were invented**. Placeholders were chosen over deletion so the blocks remain visible as outstanding actions for the co-authors. |

### Two numeric observations that are *not* errors (but a reviewer may ask)

1. **K1 FP32 serialized size appears as both 5.286 MiB and 5.285 MiB.** Table 6
   (efficiency) reports 5,542,468 B = 5.286 MiB from
   `CORRECTED_EFFICIENCY_SUMMARY.json`; Table 8 (Q8) reports 5,541,832 B = 5.285 MiB from
   `Q8_FINAL_REPORT.md` §4. Each matches its own frozen source; the 636-byte difference
   comes from two separate `torch.save` payloads. **Left as-is** — "correcting" either
   would break traceability. Flagged because it is the kind of detail a careful reader spots.
2. **K1 CPU batch-1 model-only latency appears as 28.560 ms (Table 6) and 31.046 ms
   (Table 8).** These come from two different benchmark runs (the corrected efficiency
   benchmark vs the Q8 benchmark). Both are correct in context; the Q8 table compares K1
   against Q8 *within* the Q8 run, which is the right internal control. **Left as-is**, but
   see G5.
3. **`98.63%` vs `98.62%`.** The frozen `Q8_FINAL_REPORT.md` §7 table gives an equal-domain
   mean of **0.9863**, while its §11 summary bullet rounds to 98.62%. The manuscript follows
   the §7 table, which is the authoritative computation. No change made.

---

## D. Citation issues requiring human verification

> **Status: all five items below were resolved in the release pass. See Section I.**

The bibliography is structurally clean (20/20 cited, no duplicates, no missing keys, DOIs
present on all 16 journal/conference entries). **No metadata was invented or altered.**
At the time of the polish pass, these items needed verification:

| # | Entry | Issue | Suggested action |
|---|---|---|---|
| D1 | `Nunes2023Challenges` | Authors are initials only ("Nunes, P. and Santos, J. and Rocha, E.") while every other entry uses full given names. | Expand to full names for consistency; verify against the published article. |
| D2 | The four `@misc` dataset entries (`CWRUDataset`, `JNUDataset`, `MaFaulDaDataset`, and the pattern generally) | They carry **no `year` field**, so `plainnat` renders them without a date. In the PDF the dataset citation reads *"[Case Western Reserve University Bearing Data Center, Jiangnan University, Hou et al., 2023, Federal University of Rio de Janeiro]"* — visibly inconsistent, with only the HIT reference carrying a year. | Decide a citation convention (release year, or an explicit access year with `urldate`/`note`). This is **visible in the compiled PDF** and worth fixing before submission. |
| D3 | `Hou2023HIT` | No `pages` field. | Verify whether JDMD assigns an article/page number. |
| D4 | `Mannone2026VibFM`, `Sun2026BFDKD` | Both dated 2026 (current year). | Confirm final volume/issue/pages are stable, not in-press values. |
| D5 | `Mannone2026VibFM` | BibTeX: *"can't use both volume and number fields"*. `plainnat` will not print both for an `@inproceedings`. | Keep whichever the PHM Society citation format requires and drop the other. |

**Lump citations:** the review brief asks for these to be reduced. One remains — the
four-dataset citation in Methods §3.2, `\citep{CWRUDataset,JNUDataset,Hou2023HIT,MaFaulDaDataset}`.
This is a legitimate grouped citation (one source per dataset for a single sentence about
dataset provenance) and was left intact. No other sentence attaches multiple citations to
multiple distinct claims.

---

## E. Terminology standardizations

| Change | Rationale |
|---|---|
| "Macro-Domain F1" → "Macro-4 Macro-F1 (frozen field `macro_domain_f1`)" | The only deviation from the canonical metric names found in the manuscript. Verified numerically identical in all 18 frozen reports. |
| Section 4.2 heading: "Dataset-Level and Class-Level Performance" → "Dataset-Level Performance" | Class-level analysis is now Section 4.3. |

**Already consistent, no change needed:** PC-STE, Full-S1, K1, Q8(K1), Macro-F1, Macro-AUC,
Macro-4, knowledge distillation, self-supervised. The manuscript uses "K1" canonically after
first definition; "the student" and "the lightweight student" appear only where the
teacher–student relationship is the subject of the sentence, which is appropriate. No
stray "S1", "full model", "base model" or "Full PC-STE" variants exist.

---

## F. LaTeX / build issues

Final build (`pdflatex` → `bibtex` → `pdflatex` ×3):

| Metric | Result |
|---|---|
| Errors | **0** |
| Undefined references | **0** |
| Undefined citations | **0** |
| `??` in PDF text | **0** |
| Overfull boxes | **0** |
| Underfull boxes | **0** (was 2) |
| LaTeX warnings | **0** |
| BibTeX warnings | 4 (metadata only — see D2 and D5) |
| Pages | 22 |

The four BibTeX warnings are metadata issues for the authors to resolve, not build defects:
`Warning--empty year in CWRUDataset / MaFaulDaDataset / JNUDataset` (D2) and
`Warning--can't use both volume and number fields in Mannone2026VibFM` (D5). They do not
affect compilation and produce no undefined citation.

Notes:

- **`siunitx` dependency removed** (C7). It was loaded but unused, and it is absent from a
  default TeX Live 2023 installation — it had to be installed into `~/texmf` on this machine
  before the manuscript would compile at all. The manuscript now builds with packages present
  in a standard distribution.
- **Build recipe.** The full-page float shifts labels, so the manuscript needs
  `pdflatex → bibtex → pdflatex → pdflatex → pdflatex`, or simply `latexmk -pdf main.tex`.
  `README.md` documents this.
- **Float placement.** All six figures and all nine tables appear at or near first mention;
  no float drifts more than one page. Figures 1, 2, 3, 5 and 6 are top floats sharing a page
  with text; Figure 4 takes its own float page because it is full-page by design.
- **Figure 4 sizing.** The confusion-matrix asset is 7.16 × 8.71 in, capped at
  `0.78\textheight` with `keepaspectratio` and rendering at 80.6% of natural size. Checked
  at 300 dpi: all MaFaulDa class names, percentages and `<0.1` markers remain legible in
  print.
- **Float class.** Figure 4 was changed from `figure*` to `figure`. In this single-column
  class `figure*` gains nothing, and keeping all six in one float class is what guarantees
  the 1–6 numbering order. Switch back to `figure*` if the manuscript migrates to a
  two-column journal class.

---

## G. Suggested supervisor questions

The items below are the ones most likely to be raised, based on what the manuscript actually
claims and where its evidence is thinnest. Each has an answer already in the text — the note
says where.

1. **Why does K1 outperform Full-S1 if it has half the temporal directions?**
   The single most likely first question. Section 4.8 offers regularization and the smoother
   three-teacher target as plausible mechanisms, and explicitly declines a causal claim.
   Expect a push for an ablation that isolates the mechanism (distillation vs directionality),
   which the current frozen design cannot answer.

2. **Is the −0.02 Macro-F1 margin justified, or was it chosen to be easy to pass?**
   The manuscript states the margin was fixed before TEST but does not give a domain
   rationale for the value. Worth preparing a justification (what magnitude of Macro-F1 loss
   would be operationally unacceptable) — reviewers routinely challenge non-inferiority margins.

3. **Nine cells is a small number of experimental units — is the sign-flip test adequate?**
   *p* = 0.001953125 is the *minimum attainable* value with 9 paired units, which the
   manuscript states honestly. A reviewer may ask why no confidence interval is given;
   Section 4.4 explains that none was pre-registered and none is introduced post hoc.

4. **Does the Q8 accuracy result apply to the model you would actually deploy?**
   No — and the manuscript says so in five places. The honest answer is that `sim` was
   evaluated and `cpu_dynamic` agrees with it on 98.63% of validation windows overall but
   only 95.0% on MaFaulDa. Expect a follow-up on whether the Q8 claim should be demoted to
   supplementary material until `cpu_dynamic` receives its own pre-registered evaluation.

5. **Why do two tables report different K1 CPU latencies (28.560 ms vs 31.046 ms)?**
   Different benchmark runs (corrected efficiency benchmark vs Q8 benchmark). Currently
   explained only implicitly. **Recommend adding one clarifying clause to the Table 8 caption**
   — see "Remaining author decisions" in the final report.

6. **Is a pure-PyTorch selective scan a fair basis for a speed-up claim?**
   Fused Mamba kernels were unavailable. Disclosed in Methods §3.9, Results §4.5 and
   Limitations §4.8. Expect the question of whether the 1.9× CPU advantage would survive a
   fused kernel, since a fused implementation may benefit the bidirectional model differently.

7. **Why a same-architecture, same-fold seed ensemble rather than heterogeneous teachers?**
   Sections 2.3 and 4.7 argue the diversity source is stochastic optimization and that this
   ties the distillation to the matched fold–seed evaluation. A reviewer may ask whether a
   single teacher would have worked as well — not tested.

8. **What generalization does this actually demonstrate?**
   Within-dataset held-out partitions under frozen protocols; CWRU is cross-load on one rig.
   Denied consistently, but expect a request to soften the framing further or to add an
   unseen-machine experiment.

9. **Do the figures duplicate the tables?**
   Partly by design: Figures 2, 3, 5 and 6 visualise data that Tables 4–9 also give
   numerically. The prose was not expanded when the figures were added, so no number is
   stated three times. If a journal page limit bites, the tables are the more compressible
   half — see Section H.

---

## H. Main-paper figure set — complete

The intended six-figure set is now present, integrated and numbered 1–6 in the intended
order. Verified in the compiled PDF by extracting each caption and its page.

| # | Figure | Label | Source script | First cited | Page |
|---|---|---|---|---|---|
| 1 | Study workflow overview | `fig:workflow` | `fig01_workflow.py` | §3 opening | 7 |
| 2 | Matched fold–seed performance | `fig:matched_cells` | `fig02_matched_cells.py` | §4.1 | 14 |
| 3 | Dataset-level Macro-F1 / Macro-AUC | `fig:dataset_level` | `fig03_dataset_level.py` | §4.2 | 16 |
| 4 | Aggregated confusion matrices | `fig:confusion_aggregated` | `generate_confusion_figure.py` | §4.3 | 17 |
| 5 | Computational efficiency | `fig:efficiency` | `fig05_efficiency.py` | §4.5 | 18 |
| 6 | Secondary Q8(K1) extension | `fig:q8` | `fig06_q8.py` | §4.6 | 20 |

Provenance and consistency notes:

- Every value plotted in Figures 2, 3, 5 and 6 is **read at build time** from the frozen
  artifacts (`aggregate_summary.json`, `matched_cells.csv`, `PER_DATASET_SUMMARY.csv`,
  `CORRECTED_EFFICIENCY_SUMMARY.json`, `q8_*.json`). No figure hard-codes a result, so a
  figure cannot drift from the sealed numbers. Figure 1 carries only protocol constants and
  reads the two parameter counts from the frozen efficiency artifact.
- The superseded `efficiency_v1` absolute timings and the chained Full-S1→Q8 ratio derived
  from them are never read by any figure script. Confirmed absent from the compiled PDF.
- One shared style module (`scripts/publication_figures/pubfig_style.py`) fixes the
  typeface, the Okabe-Ito colourblind-safe palette and the Full-S1 / K1 / Q8 encoding
  (blue circle / vermillion square / green triangle, with distinct hatching on bars), so
  identity is never carried by colour alone.
- Figure 2 draws the −0.02 non-inferiority margin in the Macro-F1 panel only, and labels the
  Macro-AUC block "descriptive; no margin defined".
- Figure 6 is deliberately subordinate: a shorter figure, K1-vs-Q8 ratios rather than
  headline speed-ups, and a caveat strip stating that the non-inferiority result covers the
  weight-only simulated representation, not the deployed `cpu_dynamic` artifact.
- Assets exist as PDF, SVG and PNG under `figures/publication_figures/`; only the
  manuscript-facing PDFs are copied into `manuscript/figures/`.

`SUPERVISOR_POLISH.diff` records the earlier editorial polish pass only; it predates this
figure integration.

---

## I. Final bibliography and front-matter release pass

Date: 2026-09-21. Authoritative sources: the CrossRef REST API
(`api.crossref.org/works/<doi>`), the OpenAlex API (`api.openalex.org`) for expanded author
names, and the publisher landing page reached through `https://doi.org/<doi>`. **No metadata
was invented.** Every value below was read from one of those records.

### I.1 Entries changed

| Entry | Issue before | Correction | Authoritative basis |
|---|---|---|---|
| `Mannone2026VibFM` | `@inproceedings` carrying both `volume` and `number`; BibTeX warned *"can't use both volume and number fields"* | Changed to `@article` with `journal = {PHM Society European Conference}`; volume 9, number 1, pages 1--15 retained | CrossRef records the DOI as **`type: journal-article`** on the PHM Society proceedings-journal platform. Warning eliminated without dropping any field. |
| `Nunes2023Challenges` | Author initials only (`Nunes, P. and Santos, J. and Rocha, E.`), inconsistent with every other entry | Expanded to `Nunes, Pedro and Santos, Jos\'{e} and Rocha, Eug\'{e}nio M.` | OpenAlex display names: *Pedro Nunes; José Santos; Eugénio M. Rocha* (CrossRef itself stores only initials). Matches the Elsevier record. No middle initial was guessed for José Santos. |
| `Hou2023HIT` | Title was the informal dataset name *"An Inter-Shaft Bearing Fault Diagnosis Dataset from an Aero-Engine System"*; `pages` missing | Title replaced with the official published title **"Inter-shaft Bearing Fault Diagnosis Based on Aero-engine System: A Benchmarking Dataset Study"**; `pages = {228--242}` added | Title confirmed identically by CrossRef, OpenAlex and the publisher's `citation_title` meta tag. Pages from the publisher landing page (`citation_firstpage` 228, `citation_lastpage` 242) — CrossRef and OpenAlex hold no page data for this DOI. Volume 2 / No. 4 confirmed on the same page ("Vol. 2 No. 4 (2023)"). |
| `CWRUDataset` | No `year`; BibTeX warned *"empty year"* | `year = {n.d.}` plus `note` recording that the provider states no publication date and the access date (21 September 2026) | Provider page carries no publication date. No date was guessed. |
| `JNUDataset` | No `year`; BibTeX warned *"empty year"* | Same undated-resource treatment; existing descriptive note preserved | As above. |
| `MaFaulDaDataset` | No `year`; BibTeX warned *"empty year"* | Same undated-resource treatment; existing descriptive note preserved | As above. |

### I.2 Undated dataset sources

CWRU, JNU and MaFaulDa are cited as web resources because no peer-reviewed dataset paper for
them exists in the bibliography. They now use the standard undated convention
`year = {n.d.}` with an explicit access date, which renders as e.g.
*"Case Western Reserve University Bearing Data Center. Bearing data center. Online dataset,
n.d. URL ... Accessed 21 September 2026."* This silences the BibTeX warning **honestly**: it
states that no date is available rather than substituting a guess.

HIT is the exception and already cites the peer-reviewed dataset paper (`Hou2023HIT`); no
source was substituted for any other dataset.

The Methods sentence that previously grouped all four dataset citations into one lump
`\citep{...}` now names each dataset at first mention —
*"CWRU~\citep{CWRUDataset}, JNU~\citep{JNUDataset}, HIT~\citep{Hou2023HIT}, and
MaFaulDa~\citep{MaFaulDaDataset}"* — which also removes the visually awkward combined
citation flagged in Section D.

### I.3 The two 2026-dated entries

Both are **final published articles**, not in-press placeholders:

- `Mannone2026VibFM` — CrossRef `issued` 2026-07-03, volume 9, issue 1, pages 1--15. Final.
- `Sun2026BFDKD` — CrossRef `published-online` 2025-09-18 but `published-print` **2026-08**,
  with `journal-issue` volume 48, issue 12, pages 3402--3413. The stored `year = {2026}` is
  correct for the print issue of record; volume, issue and pages already matched. No change
  needed beyond a provenance comment.

### I.4 Front- and end-matter

No CRediT role, corresponding author, funding source or competing-interest declaration was
invented.

| Block | State |
|---|---|
| Corresponding author | Added a visible line under the affiliation: *"Corresponding author to be designated before submission."* None assigned. |
| Author Contributions | *"Author contributions will be finalized by all authors before submission."* |
| Data and Code Availability | States that the repository exists but **is currently private**, that the public release and archive identifier are still to be finalized, and that the four datasets come from their original providers and **are not redistributed**. No Zenodo DOI invented. |
| Competing Interest | *"Competing-interest declarations will be confirmed by all authors before submission."* No declaration made on the authors' behalf. |
| Funding / acknowledgements | No such section exists; none was added rather than shipping an empty heading. Flagged for the authors. |

### I.5 Non-inferiority margin wording

Methods now states the factual protection explicitly, without inventing a domain rationale
for the value 0.02: *"The largest acceptable Macro-F1 loss was fixed at $m=0.02$ before the
final sealed TEST evaluation was run; it was not chosen after any TEST outcome had been
observed."* No claim is made that 0.02 is an established industrial or clinical margin.

### I.6 Final warning count

**BibTeX warnings: 0.** All five metadata issues were resolved from authoritative records;
none was suppressed with an invented value.

One implementation note: a `%` comment added above the `Mannone2026VibFM` entry initially
broke the build, because BibTeX scans for `@` even inside `%` lines and parsed the word
"@article" in the comment as an entry start. The comment was reworded; no entry data was
affected.

---

## J. What was *not* changed, deliberately

- No table was deleted or shortened. Decimal precision was left at the frozen precision.
- No paragraph was cut. The Introduction's results-preview paragraph (§1, para 8) duplicates
  numbers that also appear in the Abstract and Results; this is conventional and was left,
  but it is the first place to cut if a journal page limit bites.
- The efficiency-correction audit paragraph (§4.5) was retained. It is the manuscript's
  strongest integrity signal, and it cites only "roughly 2.2× to roughly 1.9×", not the
  superseded absolute timings.
- No new statistical test, figure, or claim was introduced.
