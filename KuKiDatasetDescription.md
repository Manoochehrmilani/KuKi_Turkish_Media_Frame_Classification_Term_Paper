# KuKi Annotated Corpus — Dataset Description (v1.0, 2026-09-24)

All numbers are computed from the six Label Studio exports (2026-09-22, TR-B re-exported 2026-09-24) via `aggregate.py` and `stats.py`.

## 1. Summary

- **What:** hand-annotated corpus of Russian and Turkish political opinion journalism for narrative, framing, persuasion and coded-language analysis.
- **Size:** 437 annotated articles (RU 207, TR 230), 645 individual annotations (RU 301, TR 344) by 6 annotators.
- **Layers:** L1 entity roles · L2 frames · L3 persuasion · L4 coded language (novel, open-text).
- **Files:**
  - `kuki_ru_aggregate.jsonl`, `kuki_tr_aggregate.jsonl` — one record per article: metadata, `content`, every annotator's normalised annotation, frame votes (`l2_frame_votes`), `l2_frames_majority`, `l2_frames_union`, and `spans_merged` (overlapping same-label spans unioned, with `support` = number of annotators).
  - `aggregate.py` / `stats.py` — reproduce the aggregate and all statistics below from the raw exports.

## 2. Source overview

- **Selection logic:** 8 outlets (4 per language), chosen in three steps:
  - *Candidate list:* an initial set of outlets was compiled in discussion with Russian and Turkish nationals.
  - *Technical feasibility:* outlets whose `robots.txt` or page design did not permit automated scraping were removed.
  - *Balance:* the final set was balanced between outlets more and less aligned with the government, approximated from civil-society assessments (e.g. RSF Press Freedom Index, Freedom House) and reach (Wikidata used as a proxy, as no reliable audience figures were available).
  - Genre was controlled at harvest time by scraping opinion/analysis sections only.
- **Why no content-based filtering:** data collection was carried out by a researcher who does not speak Russian or Turkish. We deliberately did not use machine translation or LLMs to filter by content, because pre-selecting texts with such systems would bias the corpus toward what they recognise and undermine its main use case: evaluating LLMs on exactly this kind of culturally embedded text. Instead, outlet sections (Ressorts) that publish almost exclusively opinion and analysis serve as proxies for genre, complemented by structural quality checks (length, paragraph structure, bylines, dates). Translation software was used at exactly one point: to cross-check the non-narrative clusters excluded during sampling (see below).
- **Harvest pool:** 43,696 scraped articles → 27,445 after structural pre-filter (`--strict --since 2020-01-01`).
- **Russian sources (annotated set)**

  | Source | Section | Articles | Mean words |
  |---|---|---|---|
  | RIA Novosti | Аналитика | 25 | 1,146 |
  | Nezavisimaya Gazeta (`ng`) | /vision/ | 60 | 791 |
  | Holod | Мнения и интервью | 60 | 1,322 |
  | The Insider | opinions | 62 | 1,427 |

- **Turkish sources (annotated set)**

  | Source | Section | Articles | Mean words |
  |---|---|---|---|
  | Cumhuriyet | yazarlar | 58 | 512 |
  | Sabah | yazarlar | 57 | 483 |
  | Yeniçağ | yazarlar | 57 | 505 |
  | Oda TV | yazarlar (allow-listed authors) | 58 | 540 |
- **Time span:** RU 2020-01-16 – 2026-08-06; TR 2020-03-19 – 2026-08-10.
- **Length:** RU median 1,188 words (320–2,014); TR median 490 words (257–863). Lengths are not comparable across languages (see §5).
- **Sampling of the 230 articles/language:** drawn in `analyze_and_select.ipynb` from the GNN cluster representation of the pre-filtered corpus (checkpoint `kuki_gnn_pretrained.pt`, repo `narrative-graphs`).
  - The planned LLM genre/stance judge panel was **not** run. Instead the clusters were inspected manually; clusters consisting of weather reports, sports results and stock tickers were identified without language knowledge (cross-checked with translation software) and removed. All remaining clusters were left untouched.
  - Rationale: far more was scraped than can be annotated, so annotation effort should not go into non-narrative text. The GNN concepts are general and distinct from the annotation labels, so they act as a proxy for narrativity, not as a filter on the dependent variable.
  - RIA Novosti contributes only 25 RU articles because its Аналитика section holds ~78 articles in total.

## 3. Annotation overview

- **Guidelines:** KuKi Codebook v1.0 (adapted SemEval schemes).
  - L1 Entity roles — SemEval-2025 Task 10 ST1, 22 sub-roles collapsed to PROTAGONIST / ANTAGONIST / INNOCENT; span-level.
  - L2 Frames — SemEval-2023 Task 3 ST2 / Media Frames Corpus, 14 frames; document-level multi-label.
  - L3 Persuasion — SemEval-2023 Task 3 ST3, 23 techniques collapsed to 6 families; span-level.
  - L4 Coded language — project-specific; span + three free-text fields (literal reading, insider meaning, why coded).
- **Tool:** Label Studio, one project per annotator slot (RU 6/7/8, TR 9/10/13); annotated field `content` = `title + "\n\n" + body`.
- **Annotators:** 3 per language, all native-speaking students. The codebook was discussed in the group, every annotator completed test/training annotations, and their feedback was incorporated into the codebook before final annotation. LS user IDs: RU A=8, B=4, C=10; TR A=11, B=7, C=6.
- **Design:** balanced overlap blocks per language — ABC 50, A/B/C 50 each, AB/BC/CA 10 each = 230 articles, 120 tasks per annotator.
- **Period:** annotations submitted 2026-08-24 – 2026-09-21; calibration round of 40 articles (20/20) before. Calibration led to minor changes only: ambiguities were clarified and examples for edge cases added (v1.0).
- **Completion**

  | | RU | TR |
  |---|---|---|
  | Articles in design | 230 | 230 |
  | Articles with ≥1 annotation | 207 | 230 |
  | …1 / 2 / 3 annotators | 139 / 42 / 26 | 151 / 44 / 35 |
  | Never annotated (all B-only block) | 23 | 0 |
  | Annotations per annotator (A/B/C) | 121 / 60 / 120 | 134 / 120 / 90 |

- **Label volume (all annotations)**
  - RU: L1 1,715 spans (ANT 926, PRO 432, INN 357) · L3 4,346 spans (Manipulative Wording 2,593, Attack on Reputation 679, Justification 518, Call 306, Simplification 234, Distraction 16) · L4 981 coded phrases · 4.4 frames per annotation.
  - TR: L1 1,322 spans (ANT 684, PRO 342, INN 296) · L3 790 spans (Attack on Reputation 214, Justification 197, Manipulative Wording 186, Simplification 90, Call 75, Distraction 28) · L4 218 coded phrases · 3.3 frames per annotation.
- **Most frequent frames (majority label, doc-level)**
  - RU: Political 122, Policy prescription 97, External regulation & reputation 92, Security & defense 91.
  - TR: Political 147, Fairness & equality 61, Cultural identity 60, Policy prescription 58.
- **Inter-annotator agreement (Krippendorff's α, nominal, docs with ≥2 annotators: RU 68, TR 79)**

  | Measure | RU | TR |
  |---|---|---|
  | L2 frames, pooled binary | 0.53 | 0.34 |
  | L2 best frames | Legality 0.72, Ext. reputation 0.68, Security 0.67 | Economic 0.64, Security 0.64, Cultural identity 0.51 |
  | L2 worst frames | Policy prescription 0.06, Quality of life 0.25 | Legality 0.06, Public opinion 0.07, Ext. reputation 0.10 |
  | L1 role present in doc (PRO/ANT/INN) | 0.34 / 0.40 / 0.13 | 0.18 / 0.32 / 0.25 |
  | L3 family present in doc | −0.11 – 0.12 | −0.24 – 0.19 |
  | L4 any coded phrase in doc | −0.18 | 0.06 |
  | Lenient span match rate L1 / L3 / L4* | 0.27 / 0.20 / 0.21 | 0.21 / 0.05 / 0.05 |

  \*Share of an annotator's spans overlapped by a same-label span of another annotator.
- **Aggregation rules**
  - L2: `l2_frames_majority` = frames chosen by > half of the article's annotators (single-annotated articles: that annotator's frames); `l2_frames_union` kept for recall-oriented use.
  - L1/L3/L4: overlapping spans with identical layer + label merged across annotators; `support` gives how many annotators back each merged span. No span is dropped; filter by `support ≥ 2` for a high-precision subset.
  - Span text is always re-derived as `content[start:end]` (LS exports serialise newlines inconsistently; all ~9,400 offsets verified).
  - Label conflicts (same span, different role/family) are kept side-by-side, not resolved (see §4).

## 4. Design decisions

- **Section-based genre control** instead of content filtering (no translation software, no LLM filters), keeping the corpus independent of the systems it is meant to evaluate (see §2).
- **Structural-only quality checks** (word/paragraph counts, bylines, date sanity).
- **Coarse label sets** (3 roles, 6 persuasion families) for cross-lingual transferability.
- **Title merged into `content`** for all layers — a departure from SemEval-2025, where titles are excluded from L1.
- **One LS project per annotator** instead of a shared project, to keep passes independent.
- **Self-duplicates:** when an annotator also worked in another slot's project on an article they had already annotated (19 cases: TR-A 17, RU-A 2), the copy from their own project is kept.
- **No group adjudication:** the planned group-labelling and conflict-resolution round was not completed, for two reasons:
  - *Resources:* first cases showed that labelling time rose sharply compared with individual annotation, so a full group relabelling was out of scope.
  - *Subjectivity:* the aborted session made clear that the task and codebook deliberately rely on subjective interpretation of both text and instructions. We do not see this as a failure; it underlines the interpretive nature of the chosen schemas. All annotations are therefore released individually, with vote/support counts.
- **Per-author caps avoided** in favour of date windows (since 2020-01-01).

## 5. Failures, limitations and known issues

- **Incomplete annotation:** RU-B finished 60/120 and TR-C 90/120; both were partly covered by the A annotators (3 and 31 tasks). TR is complete at the article level (230/230), RU is missing 23 articles of the B-only block. 66 % of articles (RU 67 %, TR 66 %) have a single annotator; IAA rests on 68 RU and 79 TR articles.
- **Low agreement on span layers:** L3 and L4 doc-level α are near or below zero — agreement is worse than chance-expected presence, i.e. annotators differ systematically in *whether* they use a layer, not only *where*.
- **Strong annotator effects:**
  - RU: A and C mark ~16–18 L3 spans/article (mostly Manipulative Wording), B ~4; C marks 5.3 L4 phrases/article vs 1.2–2.3 for others.
  - TR: C marks 5.2 L3 spans/article, A 0.7, B 1.9; L4 near-absent for A (0.3) and B (0.1).
  - Median time per article: TR 4.6–10.1 min vs RU 15–22 min (RU-B 94 min, likely idle-inclusive timer).
- **Codebook violations detected structurally:** all of these concern rules that require subjective judgement (e.g. which entity is *central* to a story), so there is no objective ground truth to measure them against.
  - L1 spans in the title (27: RU 9, TR 18) although the codebook excludes titles from L1.
  - L1 spans on generic or non-entity text (e.g. "children", abstract events) — not automatically quantifiable.
  - Many spans include leading/trailing whitespace (cosmetic; offsets unchanged).
- **Incomplete fields:**
  - L4 explanation: RU 875/981 fully described, TR only 81/218 (TR-B: 0/17).
  - 3 annotations without any frame (RU 2, TR 1).
- **Codebook/tool mismatch:** L4 has four fields in the codebook (§6.2) but three TextAreas in LS (the phrase is the span).
- **Source/length confounds:** RU articles are ~2.4× longer than TR; NG is shorter than other RU sources; RIA contributes only 25 articles. Normalise counts by length and stratify by source.

## 6. Record schema (`kuki_{ru,tr}_aggregate.jsonl`)

- `doc_id` (`source:article_id`), `lang`, `source`, `block`, `published_at`, `title`, `content`
- `n_annotators`, `annotators` (LS user IDs)
- `annotations` → per annotator: `project`, `created`, `lead_time_s`, `frames[]`, `spans[]` (`layer`, `label`, `start`, `end`, `text`; L4 adds `literal`, `insider`, `why`)
- `l2_frame_votes`, `l2_frames_majority`, `l2_frames_union`
- `spans_merged[]` → `layer`, `label`, `start`, `end`, `text`, `annotators`, `support`
