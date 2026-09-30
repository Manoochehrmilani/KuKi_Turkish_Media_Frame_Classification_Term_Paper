import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

SEED = 42
LABELS = [
    'Political',
    'Fairness & equality',
    'Cultural identity',
    'Policy prescription & evaluation',
]
DATA = Path('/mnt/data/kuki_tr_aggregate.jsonl')
OUT = Path('/mnt/data/kuki_tr_results')
OUT.mkdir(exist_ok=True)

with DATA.open(encoding='utf-8') as f:
    rows = [json.loads(line) for line in f if line.strip()]
texts = [r['content'] for r in rows]
y = np.asarray([
    [int(label in r['l2_frames_majority']) for label in LABELS]
    for r in rows
], dtype=int)


def multilabel_folds(Y, k, seed):
    """Greedy multilabel stratification with balanced label counts and fold sizes."""
    n, L = Y.shape
    rng = np.random.default_rng(seed)
    inv = 1.0 / (Y.sum(axis=0) + 1e-9)
    sample_priority = (Y * inv).sum(axis=1)
    # Tie-break randomly but deterministically.
    jitter = rng.random(n) * 1e-6
    order = np.argsort(-(sample_priority + jitter))
    fold_indices = [[] for _ in range(k)]
    fold_label_counts = np.zeros((k, L), dtype=float)
    fold_sizes = np.zeros(k, dtype=int)
    target_labels = Y.sum(axis=0) / k
    target_size = n / k

    for idx in order:
        pos = np.flatnonzero(Y[idx])
        costs = []
        for f in range(k):
            deficit = np.maximum(target_labels - fold_label_counts[f], 0)
            benefit = deficit[pos].sum() if len(pos) else 0.0
            size_deficit = max(target_size - fold_sizes[f], 0)
            costs.append(-(benefit + 0.25 * size_deficit) + 0.05 * fold_sizes[f])
        best = min(costs)
        candidates = np.flatnonzero(np.isclose(costs, best))
        f = int(rng.choice(candidates))
        fold_indices[f].append(int(idx))
        fold_sizes[f] += 1
        fold_label_counts[f] += Y[idx]

    return [np.asarray(sorted(f), dtype=int) for f in fold_indices]


def make_model(kind):
    if kind == 'word':
        vec = TfidfVectorizer(
            lowercase=True,
            strip_accents=None,  # Preserve Turkish characters: ı, ş, ğ, ç, ö, ü
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
    elif kind == 'char':
        vec = TfidfVectorizer(
            lowercase=True,
            strip_accents=None,
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
    else:
        raise ValueError(kind)

    # Linear SVM is used as the classifier for both representations so the
    # experiment isolates the representation choice.
    return Pipeline([
        ('tfidf', vec),
        ('clf', OneVsRestClassifier(
            LinearSVC(C=1.0, class_weight='balanced', random_state=SEED)
        )),
    ])


def tune_thresholds(train_indices, kind, seed):
    """Tune one decision threshold per label using 3-fold inner OOF predictions."""
    folds = multilabel_folds(y[train_indices], k=3, seed=seed)
    inner_scores = np.zeros((len(train_indices), len(LABELS)), dtype=float)
    for fold in folds:
        dev_pos = fold
        train_pos = np.concatenate([f for f in folds if not np.array_equal(f, dev_pos)])
        a = train_indices[train_pos]
        b = train_indices[dev_pos]
        model = make_model(kind)
        model.fit([texts[i] for i in a], y[a])
        inner_scores[dev_pos] = model.decision_function([texts[i] for i in b])

    thresholds = []
    for j in range(len(LABELS)):
        best_t, best_f = 0.0, -1.0
        for t in np.linspace(-0.8, 0.8, 81):
            f = f1_score(y[train_indices, j], inner_scores[:, j] >= t, zero_division=0)
            if f > best_f:
                best_t, best_f = float(t), float(f)
        thresholds.append(best_t)
    return np.asarray(thresholds)


def evaluate(kind):
    outer_folds = multilabel_folds(y, k=5, seed=SEED)
    oof_pred = np.zeros_like(y)
    fold_rows = []
    t0 = time.time()

    for fold_id, test_idx in enumerate(outer_folds, start=1):
        train_idx = np.concatenate([f for j, f in enumerate(outer_folds) if j != fold_id - 1])
        thresholds = tune_thresholds(train_idx, kind, seed=100 + fold_id)
        model = make_model(kind)
        model.fit([texts[i] for i in train_idx], y[train_idx])
        scores = model.decision_function([texts[i] for i in test_idx])
        pred = (scores >= thresholds[None, :]).astype(int)
        oof_pred[test_idx] = pred

        per = f1_score(y[test_idx], pred, average=None, zero_division=0)
        fold_rows.append({
            'model': kind,
            'fold': fold_id,
            'macro_f1': f1_score(y[test_idx], pred, average='macro', zero_division=0),
            'micro_f1': f1_score(y[test_idx], pred, average='micro', zero_division=0),
            **{f'f1_{LABELS[j]}': per[j] for j in range(len(LABELS))},
            **{f'threshold_{LABELS[j]}': thresholds[j] for j in range(len(LABELS))},
        })

    overall = {
        'model': kind,
        'macro_f1': f1_score(y, oof_pred, average='macro', zero_division=0),
        'micro_f1': f1_score(y, oof_pred, average='micro', zero_division=0),
        'weighted_f1': f1_score(y, oof_pred, average='weighted', zero_division=0),
    }
    per = f1_score(y, oof_pred, average=None, zero_division=0)
    for j, label in enumerate(LABELS):
        overall[f'f1_{label}'] = per[j]

    elapsed = time.time() - t0
    overall['elapsed_seconds'] = elapsed

    oof = pd.DataFrame({
        'doc_id': [r['doc_id'] for r in rows],
        'source': [r['source'] for r in rows],
        'title': [r['title'] for r in rows],
    })
    for j, label in enumerate(LABELS):
        oof[f'{label}_true'] = y[:, j]
        oof[f'{label}_pred'] = oof_pred[:, j]
    oof.to_csv(OUT / f'{kind}_oof_predictions.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / f'{kind}_fold_results.csv', index=False)
    return overall, fold_rows


if __name__ == '__main__':
    summary = []
    for kind in ['word', 'char']:
        result, folds = evaluate(kind)
        summary.append(result)
        print('\n', kind)
        print(classification_report(
            y,
            pd.read_csv(OUT / f'{kind}_oof_predictions.csv')[[f'{l}_pred' for l in LABELS]].to_numpy(),
            target_names=LABELS,
            digits=3,
            zero_division=0,
        ))
        print(result)

    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    print('\nSaved results to', OUT)
