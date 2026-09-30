# KuKi Turkish Media Frame Classification

This repository contains the code, results, validation materials, and project documentation for the term paper:

**Automatic Media Frame Classification in Turkish Political Opinion Journalism:  
Lexical Representations, Cross-Outlet Generalization, and an Exploratory Analysis of Passive Voice**

**Seminar:** Advanced Topics in Natural Language Processing  
**Instructor:** Christoph Hau, University of Trier  
**Semester:** Summer Semester 2026

## Project overview

This project investigates automatic media-frame classification in the Turkish portion of the KuKi corpus, a hand-annotated collection of Russian and Turkish political opinion journalism.

The main classification experiment focuses on four document-level frames:

- Political
- Fairness & equality
- Cultural identity
- Policy prescription & evaluation

Two lexical representations are compared with the same one-versus-rest Linear SVM classifier:

1. **Word TF-IDF** using unigrams and bigrams
2. **Character TF-IDF** using character n-grams of length 3–5

The main evaluation uses five-fold multilabel cross-validation with inner threshold tuning. A leave-one-out-outlet evaluation is used as a source-held-out robustness check.

A secondary exploratory analysis investigates Turkish passive constructions and their relationship to entity-role annotation patterns.

## Main results

For the 230 Turkish articles, the main out-of-fold results are:

| Representation | Macro-F1 | Micro-F1 |
|---|---:|---:|
| Word TF-IDF | 0.579 | 0.600 |
| Character TF-IDF | 0.571 | 0.594 |

The overall difference between the two representations is small. Their performance differs more clearly at the level of individual frame categories and under source-held-out evaluation.

For the leave-one-out-outlet experiment, performance was averaged over five threshold-tuning seeds. Character features achieved a slightly higher mean macro-F1 on unseen outlets than word features.

## Repository structure

```text
.
├── README.md
│
├── kuki_tr_experiment.py
├── kuki_tr_supplementary_analyses.py
├── kuki_tr_passive_analysis.py
│
├── KuKiDatasetDescription.md
├── KuKi_Codebook_v0.3-2.pdf
│
├── kuki_tr_classification_results/
│   ├── word_oof_predictions.csv
│   ├── char_oof_predictions.csv
│   ├── word_fold_results.csv
│   ├── char_fold_results.csv
│   ├── representation_comparison.csv
│   └── loso_word_results.csv
│
├── supplementary_results/
│   ├── majority_baseline.csv
│   ├── bootstrap_word_minus_char.csv
│   ├── loso_summary_mean_over_seeds.csv
│   ├── cv_vs_loso_drop.csv
│   ├── alpha_per_frame.csv
│   ├── odatv_masking_check.csv
│   └── supplementary_summary.json
│
├── results_passive/
│   ├── summary.json
│   ├── sentence_features.csv
│   ├── multiannotated_sentence_features.csv
│   ├── article_disagreement_summary.csv
│   ├── source_summary.csv
│   ├── l1_disagreement_by_length_bin.csv
│   ├── l3_disagreement_by_length_bin.csv
│   ├── evidential_by_l3_family_per_annotator.csv
│   ├── manual_validation_50.csv
│   └── manual_validation_agent_balance.csv
│
└── paper/
    ├── KuKi_Turkish_Media_Frame_Classification_Term_Paper_FINAL.docx
    └── KuKi_final_fixed.pdf
```

## Code

### `kuki_tr_experiment.py`

This is the main classification script used for the reported word- and character-level TF-IDF experiments.

It implements:

- word TF-IDF with 1–2 grams
- character TF-IDF with 3–5 `char_wb` grams
- preservation of Turkish characters
- one-versus-rest Linear SVM
- balanced class weights
- five-fold multilabel cross-validation
- three-fold inner threshold tuning
- out-of-fold predictions
- fold-level evaluation

### `kuki_tr_supplementary_analyses.py`

This script contains the supplementary analyses reported in the paper, including:

- majority-label baseline
- bootstrap comparison of word and character macro-F1
- multi-seed leave-one-out-outlet evaluation
- per-frame annotation agreement
- outlet-name masking check

### `kuki_tr_passive_analysis.py`

This script implements the secondary Turkish linguistic analysis using the Turkish Stanza model.

It extracts:

- passive voice features
- non-firsthand evidential features
- candidate agentless passives
- sentence-level alignment with KuKi L1/L3 annotations
- disagreement and label-conflict measures

The script also produces the manual-validation samples used in the paper.

## Annotation and project documentation

The original annotations were produced using **Label Studio**.

Annotation platform:

https://label-studio-production-b977.up.railway.app/projects?page=1

The repository contains the project documentation:

- `KuKiDatasetDescription.md`
- `KuKi_Codebook_v0.3-2.pdf`

These documents describe the corpus, annotation layers, frame definitions, and annotation decisions.

## Data availability

The original annotated corpus files are **not included in this public repository**.

This includes:

- `kuki_tr_aggregate.jsonl`
- `kuki_ru_aggregate.jsonl`
- raw Label Studio exports

The corpus files are project data and are not redistributed here. The input files are also too large for practical GitHub upload.

The reported result files, validation materials, code, codebook, and dataset documentation are included instead.

Researchers with authorized access to the KuKi corpus can place the relevant input file in the location expected by the scripts and reproduce the analyses.

## Course tutorial

The course text-classification tutorial provided by the instructor was used as background material when developing the experiment, but the tutorial itself is **not redistributed in this repository**. The repository contains the code actually used for the reported experiments.

## Reproducibility

The main classification results can be traced to:

- `kuki_tr_experiment.py`
- `kuki_tr_classification_results/`

The supplementary results can be traced to:

- `kuki_tr_supplementary_analyses.py`
- `supplementary_results/`

The passive-voice analysis can be traced to:

- `kuki_tr_passive_analysis.py`
- `results_passive/`

The manual validation files record the native-speaker checks used to assess the parser-derived passive and agentless-passive variables.

## Paper

The final term paper is provided in both Word and PDF format under `paper/`.

## Citation

Milani, M. (2026). *Automatic Media Frame Classification in Turkish Political Opinion Journalism: Lexical Representations, Cross-Outlet Generalization, and an Exploratory Analysis of Passive Voice*. University of Trier.

## Notes and limitations

The Turkish dataset contains 230 articles. The main classification experiment uses four relatively frequent frames rather than all 14 KuKi frame labels.

The leave-one-out-outlet evaluation is a source-held-out robustness test within the same corpus; it is not an external-domain evaluation.

The passive-voice analysis is explicitly exploratory. Its parser-derived features were checked using small manual validation samples and should not be interpreted as definitive linguistic annotations.

## Contact

**Author:** Manoochehr Mahdavi Milani  
**University:** University of Trier  
**Programme:** M.Sc. Natural Language Processing  
**Semester:** Summer Semester 2026
