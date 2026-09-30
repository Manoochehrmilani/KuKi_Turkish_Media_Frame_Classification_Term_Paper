#!/usr/bin/env python3
"""KuKi Turkish passive/evidential analysis.

This script is an exploratory analysis for the Turkish KuKi corpus. It uses
Stanza to identify passive and non-firsthand evidential constructions and
checks how these features relate to the existing L1/L3 annotations.

Main analysis idea:
    - use all Turkish articles for the corpus-level analysis
    - use only multiply annotated articles for inter-annotator analyses
    - keep annotation disagreement separate from annotation presence

Before reporting parser-based percentages, inspect the generated
manual_validation_50.csv file and check a sample of the parser decisions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, wilcoxon


L3_FAMILIES = [
    "Attack on Reputation",
    "Justification",
    "Manipulative Wording",
    "Simplification",
    "Call",
    "Distraction",
]


def spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Return True when two character spans overlap."""
    return max(start_a, start_b) < min(end_a, end_b)


def normalize_span_text(text: str) -> str:
    """Normalize whitespace/case for a simple span-text comparison."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def get_disagreement_measures(annotations: dict) -> dict:
    """Calculate label, span, presence and pure label-conflict measures."""
    label_sets = []
    span_sets = []

    for spans in annotations.values():
        labels = tuple(sorted({span[0] for span in spans}))
        spans_text = tuple(
            sorted((span[0], normalize_span_text(span[1])) for span in spans)
        )
        label_sets.append(labels)
        span_sets.append(spans_text)

    non_empty_labels = [labels for labels in label_sets if labels]

    if len(non_empty_labels) >= 2:
        label_conflict = int(len(set(non_empty_labels)) > 1)
    else:
        label_conflict = np.nan

    return {
        # Includes cases where one annotator marked something and another did not.
        "role_disagreement": int(len(set(label_sets)) > 1),
        "span_disagreement": int(len(set(span_sets)) > 1),
        "presence_disagreement": int(
            len(set(bool(labels) for labels in label_sets)) > 1
        ),
        # Only compares labels when at least two annotators marked something.
        "label_conflict": label_conflict,
    }


def get_annotation_spans(article: dict, sent_start: int, sent_end: int, layer: str) -> dict:
    """Return annotation spans from each annotator that overlap a sentence."""
    result = {}

    for annotator_id, annotation in article.get("annotations", {}).items():
        spans = []

        for span in annotation.get("spans", []):
            if span.get("layer") != layer:
                continue

            start = int(span["start"])
            end = int(span["end"])

            if spans_overlap(start, end, sent_start, sent_end):
                spans.append(
                    (
                        span.get("label", ""),
                        span.get("text", ""),
                        start,
                        end,
                    )
                )

        result[annotator_id] = spans

    return result


def process_articles(nlp, input_path: Path) -> pd.DataFrame:
    """Run Stanza over the Turkish corpus and create sentence-level features."""
    rows = []

    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            article = json.loads(line)
            content = article["content"]
            document = nlp(content)

            search_position = 0
            number_of_annotators = int(
                article.get("n_annotators")
                or len(article.get("annotations", {}))
            )
            has_multiple_annotators = number_of_annotators >= 2

            for sentence_id, sentence in enumerate(document.sentences):
                if not sentence.tokens:
                    continue

                # Character offsets are available on Stanza Token objects.
                token_spans = []
                words = []

                for token in sentence.tokens:
                    start = getattr(token, "start_char", None)
                    end = getattr(token, "end_char", None)
                    if start is not None and end is not None:
                        token_spans.append((start, end))
                    words.extend(token.words)

                if token_spans:
                    sent_start = min(start for start, _ in token_spans)
                    sent_end = max(end for _, end in token_spans)
                    sentence_text = content[sent_start:sent_end]
                else:
                    # Fallback for Stanza versions without token offsets.
                    sentence_text_raw = getattr(sentence, "text", "")
                    sent_start = content.find(sentence_text_raw, search_position)
                    if sent_start < 0:
                        raise RuntimeError(
                            "Could not recover sentence character offsets."
                        )
                    sent_end = sent_start + len(sentence_text_raw)
                    sentence_text = sentence_text_raw
                    search_position = sent_end

                passive_words = []
                evidential_words = []
                agent_dependencies = []

                for word in words:
                    features = word.feats or ""

                    if "Voice=Pass" in features:
                        passive_words.append(word.text)
                    if "Evident=Nfh" in features:
                        evidential_words.append(word.text)
                    if getattr(word, "deprel", "") == "obl:agent":
                        agent_dependencies.append(word.text)

                has_tarafindan = bool(
                    re.search(r"\btarafından\b", sentence_text, flags=re.IGNORECASE)
                )
                has_explicit_agent = bool(agent_dependencies or has_tarafindan)
                has_passive = bool(passive_words)
                is_agentless_passive = has_passive and not has_explicit_agent

                row = {
                    "doc_id": article["doc_id"],
                    "source": article.get("source", ""),
                    "published_at": article.get("published_at", ""),
                    "n_annotators": number_of_annotators,
                    "sentence_id": sentence_id,
                    "start": sent_start,
                    "end": sent_end,
                    "text": sentence_text,
                    "passive": int(has_passive),
                    "passive_tokens": " | ".join(passive_words),
                    "n_passive_tokens": len(passive_words),
                    "evidential_nfh": int(bool(evidential_words)),
                    "evidential_tokens": " | ".join(evidential_words),
                    "n_evidential_tokens": len(evidential_words),
                    "explicit_agent": int(has_explicit_agent),
                    "agent_tokens": " | ".join(agent_dependencies),
                    "agentless_passive": int(is_agentless_passive),
                    "n_words": len(words),
                }

                l1_annotations = get_annotation_spans(
                    article, sent_start, sent_end, "l1_roles"
                )
                l3_annotations = get_annotation_spans(
                    article, sent_start, sent_end, "l3_persuasion"
                )

                for label in ["ANTAGONIST", "PROTAGONIST", "INNOCENT"]:
                    row[f"any_{label}"] = int(
                        any(
                            span[0] == label
                            for spans in l1_annotations.values()
                            for span in spans
                        )
                    )

                for label in L3_FAMILIES:
                    row[f"any_l3_{label}"] = int(
                        any(
                            span[0] == label
                            for spans in l3_annotations.values()
                            for span in spans
                        )
                    )

                row["l3_by_annotator"] = json.dumps(
                    {
                        annotator: sorted({span[0] for span in spans})
                        for annotator, spans in l3_annotations.items()
                    }
                )

                if has_multiple_annotators:
                    l1_measures = get_disagreement_measures(l1_annotations)
                    l3_measures = get_disagreement_measures(l3_annotations)

                    row.update({f"l1_{k}": v for k, v in l1_measures.items()})
                    row.update({f"l3_{k}": v for k, v in l3_measures.items()})
                    row["l1_annotated_any"] = int(any(l1_annotations.values()))
                    row["l3_annotated_any"] = int(any(l3_annotations.values()))

                rows.append(row)

    return pd.DataFrame(rows)


def build_article_summary(multi_annotated: pd.DataFrame) -> pd.DataFrame:
    """Summarize passive/non-passive disagreement rates per article."""
    rows = []

    for doc_id, group in multi_annotated.groupby("doc_id"):
        def rate(mask, column):
            values = group.loc[mask, column]
            return float(values.mean()) if len(values) else np.nan

        rows.append(
            {
                "doc_id": doc_id,
                "source": group.source.iloc[0],
                "n_annotators": group.n_annotators.iloc[0],
                "n_sentences": len(group),
                "n_passive": int(group.passive.sum()),
                "n_agentless_passive": int(group.agentless_passive.sum()),
                "n_evidential": int(group.evidential_nfh.sum()),
                "l1_disagree_rate_all": float(group.l1_role_disagreement.mean()),
                "l1_span_disagree_rate_all": float(group.l1_span_disagreement.mean()),
                "l3_disagree_rate_all": float(group.l3_role_disagreement.mean()),
                "l3_span_disagree_rate_all": float(group.l3_span_disagreement.mean()),
                "l1_disagree_rate_passive": rate(group.passive.astype(bool), "l1_role_disagreement"),
                "l1_disagree_rate_nonpassive": rate(~group.passive.astype(bool), "l1_role_disagreement"),
                "l1_disagree_rate_agentless": rate(group.agentless_passive.astype(bool), "l1_role_disagreement"),
                "l1_disagree_rate_explicit_agent": rate(
                    group.passive.astype(bool) & group.explicit_agent.astype(bool),
                    "l1_role_disagreement",
                ),
                "l3_disagree_rate_passive": rate(group.passive.astype(bool), "l3_role_disagreement"),
                "l3_disagree_rate_nonpassive": rate(~group.passive.astype(bool), "l3_role_disagreement"),
                "l3_disagree_rate_agentless": rate(group.agentless_passive.astype(bool), "l3_role_disagreement"),
                "l3_disagree_rate_explicit_agent": rate(
                    group.passive.astype(bool) & group.explicit_agent.astype(bool),
                    "l3_role_disagreement",
                ),
            }
        )

    return pd.DataFrame(rows)


def paired_wilcoxon(data: pd.DataFrame, first: str, second: str) -> dict:
    """Compare two article-level rates with a paired Wilcoxon test."""
    values = data[[first, second]].dropna()

    if len(values) < 5:
        return {"n": len(values), "stat": None, "p": None, "median_diff": None}

    differences = values[first] - values[second]
    non_zero = differences[differences != 0]

    if len(non_zero) < 5:
        return {
            "n": len(values),
            "stat": None,
            "p": None,
            "median_diff": float(np.median(differences)),
        }

    result = wilcoxon(values[first], values[second], zero_method="wilcox")
    return {
        "n": len(values),
        "stat": float(result.statistic),
        "p": float(result.pvalue),
        "median_diff": float(np.median(differences)),
    }


def pooled_passive_test(data: pd.DataFrame, outcome: str):
    """Simple passive/non-passive pooled comparison (secondary result)."""
    passive = data[data.passive == 1]
    non_passive = data[data.passive == 0]

    table = np.array(
        [
            [int(passive[outcome].sum()), len(passive) - int(passive[outcome].sum())],
            [int(non_passive[outcome].sum()), len(non_passive) - int(non_passive[outcome].sum())],
        ]
    )

    if (table < 5).any():
        odds_ratio, p_value = fisher_exact(table)
        test_name = "Fisher exact"
    else:
        _, p_value, _, _ = chi2_contingency(table)
        odds_ratio = (
            (table[0, 0] + 0.5) * (table[1, 1] + 0.5)
            / ((table[0, 1] + 0.5) * (table[1, 0] + 0.5))
        )
        test_name = "Chi-square"

    return table.tolist(), float(odds_ratio), float(p_value), test_name


def make_validation_sheet(data: pd.DataFrame) -> pd.DataFrame:
    """Create a balanced passive/non-passive sample for manual parser checking."""
    candidates = data[data.text.str.len() >= 40].copy()

    passive_sample = candidates[candidates.passive == 1].sample(
        min(25, int(candidates.passive.sum())), random_state=42
    )
    non_passive_sample = candidates[candidates.passive == 0].sample(
        min(25, int((candidates.passive == 0).sum())), random_state=43
    )

    sample = pd.concat(
        [
            passive_sample.assign(sample_group="parser_passive"),
            non_passive_sample.assign(sample_group="parser_nonpassive"),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=44).reset_index(drop=True)

    columns = [
        "sample_group",
        "doc_id",
        "source",
        "sentence_id",
        "text",
        "passive",
        "passive_tokens",
        "agentless_passive",
        "explicit_agent",
        "evidential_nfh",
        "evidential_tokens",
    ]

    sample = sample[columns].copy()
    sample["manual_passive_gold"] = ""
    sample["manual_agentless_gold"] = ""
    sample["manual_notes"] = ""
    return sample


def logistic_regression_with_clustered_se(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    cluster_column: str = "doc_id",
) -> dict:
    """Fit a logistic model with article-clustered standard errors."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "Please install statsmodels first."}

    model_data = data.dropna(subset=[outcome] + predictors).copy()
    model_data["log_words"] = np.log1p(model_data["n_words"])

    if model_data[outcome].nunique() < 2:
        return {"error": "The outcome has no variation.", "n": len(model_data)}

    formula = f"{outcome} ~ " + " + ".join(predictors + ["log_words"])

    try:
        model = smf.logit(formula, data=model_data).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={
                "groups": pd.factorize(model_data[cluster_column])[0]
            },
        )
    except Exception as exc:
        return {"error": str(exc), "n": len(model_data)}

    confidence_intervals = model.conf_int()
    odds_ratios = {}

    for name in model.params.index:
        if name == "Intercept":
            continue

        odds_ratios[name] = {
            "OR": float(np.exp(model.params[name])),
            "CI95": [
                float(np.exp(confidence_intervals.loc[name, 0])),
                float(np.exp(confidence_intervals.loc[name, 1])),
            ],
            "p": float(model.pvalues[name]),
        }

    return {
        "n_sentences": int(len(model_data)),
        "n_docs": int(model_data[cluster_column].nunique()),
        "formula": formula,
        "odds_ratios": odds_ratios,
    }


def length_binned_rates(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Show disagreement rates within sentence-length quartiles."""
    result = data.copy()
    result["length_bin"] = pd.qcut(result["n_words"], 4, duplicates="drop")

    table = (
        result.groupby(["length_bin", "passive"], observed=True)[outcome]
        .agg(["mean", "count"])
        .unstack("passive")
    )
    table.columns = [f"{name}_passive={value}" for name, value in table.columns]
    return table.round(3)


def evidential_l3_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Descriptive evidential rates for each L3 family, including annotator breakdowns."""
    rows = []
    overall_baseline = data["evidential_nfh"].mean()
    per_annotator = data["l3_by_annotator"].map(json.loads)
    annotators = sorted({annotator for row in per_annotator for annotator in row})

    for family in L3_FAMILIES:
        family_mask = data[f"any_l3_{family}"] == 1
        rows.append(
            {
                "family": family,
                "annotator": "ALL",
                "n_sentences": int(family_mask.sum()),
                "evidential_rate": float(data.loc[family_mask, "evidential_nfh"].mean())
                if family_mask.any()
                else np.nan,
                "baseline_all_sentences": float(overall_baseline),
            }
        )

        for annotator in annotators:
            annotator_mask = per_annotator.map(
                lambda labels: family in labels.get(annotator, [])
            )
            seen_mask = per_annotator.map(lambda labels: annotator in labels)

            rows.append(
                {
                    "family": family,
                    "annotator": annotator,
                    "n_sentences": int(annotator_mask.sum()),
                    "evidential_rate": float(
                        data.loc[annotator_mask, "evidential_nfh"].mean()
                    )
                    if annotator_mask.any()
                    else np.nan,
                    "baseline_all_sentences": float(
                        data.loc[seen_mask, "evidential_nfh"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows).round(3)


def build_pipeline(language: str):
    """Load the Stanza pipeline needed for this analysis."""
    import stanza

    try:
        return stanza.Pipeline(
            language,
            processors="tokenize,mwt,pos,lemma,depparse",
            use_gpu=False,
            verbose=False,
        )
    except Exception:
        # Fallback for installations without a Turkish MWT package.
        return stanza.Pipeline(
            language,
            processors="tokenize,pos,lemma,depparse",
            use_gpu=False,
            verbose=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore passive/evidential constructions in Turkish KuKi data."
    )
    parser.add_argument(
        "--input",
        default="kuki_tr_aggregate.jsonl",
        help="Path to the Turkish KuKi JSONL file.",
    )
    parser.add_argument(
        "--out",
        default="results_passive",
        help="Directory for analysis outputs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import stanza  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Stanza is not installed. Run: pip install -U stanza"
        ) from exc

    nlp = build_pipeline("tr")
    data = process_articles(nlp, Path(args.input))

    data.to_csv(
        output_dir / "sentence_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    multi = data[data["n_annotators"] >= 2].copy()
    multi.to_csv(
        output_dir / "multiannotated_sentence_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    article_summary = build_article_summary(multi)
    article_summary.to_csv(
        output_dir / "article_disagreement_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_summary = (
        data.groupby("source")
        .agg(
            articles=("doc_id", "nunique"),
            sentences=("sentence_id", "count"),
            passive_sentences=("passive", "sum"),
            agentless_passives=("agentless_passive", "sum"),
            evidential_sentences=("evidential_nfh", "sum"),
        )
        .reset_index()
    )
    source_summary["passive_rate"] = (
        source_summary["passive_sentences"] / source_summary["sentences"]
    )
    source_summary.to_csv(
        output_dir / "source_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_validation_sheet(data).to_csv(
        output_dir / "manual_validation_50.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "articles": int(data.doc_id.nunique()),
        "sentences": int(len(data)),
        "passive_sentences": int(data.passive.sum()),
        "agentless_passive_sentences": int(data.agentless_passive.sum()),
        "evidential_nfh_sentences": int(data.evidential_nfh.sum()),
        "multiannotated_articles": int(multi.doc_id.nunique()),
        "note": (
            "Stanza outputs are parser predictions. Validate the sample before "
            "reporting final parser-based rates."
        ),
    }

    # The following is the part closest to the main RQ3 question.
    annotated_l1 = multi[multi["l1_annotated_any"] == 1]
    annotated_l3 = multi[multi["l3_annotated_any"] == 1]

    summary["CONDITIONAL_primary"] = {
        "description": (
            "role_disagreement includes presence differences; label_conflict "
            "is restricted to sentences where at least two annotators marked "
            "something and compares their labels."
        ),
        "l1_logit_given_annotated": logistic_regression_with_clustered_se(
            annotated_l1,
            "l1_role_disagreement",
            ["passive", "evidential_nfh"],
        ),
        "l1_label_conflict_given_both_annotated": logistic_regression_with_clustered_se(
            multi,
            "l1_label_conflict",
            ["agentless_passive", "evidential_nfh"],
        ),
        "l1_label_conflict_n_and_rate": [
            int(multi.l1_label_conflict.notna().sum()),
            float(multi.l1_label_conflict.mean()),
        ],
        "l3_label_conflict_given_both_annotated": logistic_regression_with_clustered_se(
            multi,
            "l3_label_conflict",
            ["passive", "evidential_nfh"],
        ),
        "l3_label_conflict_n_and_rate": [
            int(multi.l3_label_conflict.notna().sum()),
            float(multi.l3_label_conflict.mean()),
        ],
        "l1_agentless_given_annotated": logistic_regression_with_clustered_se(
            annotated_l1,
            "l1_role_disagreement",
            ["agentless_passive", "evidential_nfh"],
        ),
    }

    summary["PRESENCE_secondary"] = {
        "l1_any_annotation": logistic_regression_with_clustered_se(
            multi,
            "l1_annotated_any",
            ["passive", "evidential_nfh"],
        ),
        "l3_any_annotation": logistic_regression_with_clustered_se(
            multi,
            "l3_annotated_any",
            ["passive", "evidential_nfh"],
        ),
    }

    # Descriptive role-specific analysis on the full Turkish corpus.
    summary["role_presence_models"] = {}
    for label in ["ANTAGONIST", "PROTAGONIST", "INNOCENT"]:
        summary["role_presence_models"][label] = logistic_regression_with_clustered_se(
            data,
            f"any_{label}",
            ["agentless_passive", "evidential_nfh"],
        )

    summary["evidential_attack_on_reputation"] = logistic_regression_with_clustered_se(
        data.rename(columns={"any_l3_Attack on Reputation": "any_AoR"}),
        "any_AoR",
        ["evidential_nfh", "passive"],
    )

    evidential_l3_summary(data).to_csv(
        output_dir / "evidential_by_l3_family_per_annotator.csv",
        index=False,
        encoding="utf-8-sig",
    )

    length_binned_rates(
        annotated_l1, "l1_role_disagreement"
    ).to_csv(
        output_dir / "l1_disagreement_by_length_bin.csv",
        encoding="utf-8-sig",
    )

    length_binned_rates(
        annotated_l3, "l3_role_disagreement"
    ).to_csv(
        output_dir / "l3_disagreement_by_length_bin.csv",
        encoding="utf-8-sig",
    )

    summary["sentence_length_by_passive"] = (
        data.groupby("passive")["n_words"].mean().round(2).to_dict()
    )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
