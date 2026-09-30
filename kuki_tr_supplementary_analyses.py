"""
Supplementary analyses for the KuKi-TR term paper (reproduces every number
that is not produced by kuki_tr_experiment.py).

    python kuki_tr_supplementary_analyses.py

Requires kuki_tr_experiment.py (same folder) and kuki_tr_aggregate.jsonl.
Outputs go to ./supplementary_results/.

1. Majority-label baseline (predict Political for every article).
2. Paired bootstrap for word-minus-character macro-F1 (2,000 resamples).
3. Leave-one-outlet-out (LOSO) for word AND character TF-IDF, with the same
   inner threshold tuning as the main experiment, repeated over 5 seeds
   (threshold tuning is seed-dependent, so single-seed LOSO scores are unstable).
4. Per-frame Krippendorff's alpha (nominal, binary) on multi-annotated articles.
5. Outlet-name masking check (removes the string 'odatv', which occurs in all
   Oda TV articles) for the word model under random 5-fold CV.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import kuki_tr_experiment as E   # reuses data loading, folds, models, thresholds

OUT = Path("supplementary_results")
OUT.mkdir(exist_ok=True)
L = E.LABELS
Y = E.y
SRC = np.array([r["source"] for r in E.rows])
SEEDS = [1, 2, 3, 4, 5]


def macro(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


# 1. Majority-label baseline --------------------------------------------------
base = np.zeros_like(Y)
base[:, L.index("Political")] = 1
baseline = {"macro_f1": macro(Y, base),
            "micro_f1": f1_score(Y, base, average="micro", zero_division=0)}
print("Majority baseline:", {k: round(v, 3) for k, v in baseline.items()})

# 2. Paired bootstrap ---------------------------------------------------------
res = {}
for kind in ["word", "char"]:
    overall, _ = E.evaluate(kind)            # also writes OOF predictions
    res[kind] = overall
oof = {k: pd.read_csv(E.OUT / f"{k}_oof_predictions.csv") for k in ["word", "char"]}
W = oof["word"][[f"{l}_pred" for l in L]].to_numpy()
C = oof["char"][[f"{l}_pred" for l in L]].to_numpy()
rng = np.random.default_rng(E.SEED)
diffs = []
for _ in range(2000):
    i = rng.integers(0, len(Y), len(Y))
    diffs.append(macro(Y[i], W[i]) - macro(Y[i], C[i]))
ci = np.percentile(diffs, [2.5, 97.5])
print(f"Bootstrap word-char macro-F1: diff={macro(Y, W) - macro(Y, C):.4f}, "
      f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")

# 3. Leave-one-outlet-out over several seeds ----------------------------------
rows = []
for kind in ["word", "char"]:
    for seed in SEEDS:
        for s in sorted(set(SRC)):
            tr, te = np.where(SRC != s)[0], np.where(SRC == s)[0]
            th = E.tune_thresholds(tr, kind, seed=seed)
            m = E.make_model(kind)
            m.fit([E.texts[i] for i in tr], Y[tr])
            p = (m.decision_function([E.texts[i] for i in te]) >= th).astype(int)
            per = f1_score(Y[te], p, average=None, zero_division=0)
            rows.append({"model": kind, "seed": seed, "held_out": s,
                         "macro_f1": macro(Y[te], p),
                         "micro_f1": f1_score(Y[te], p, average="micro", zero_division=0),
                         **{l: per[j] for j, l in enumerate(L)}})
loso = pd.DataFrame(rows)
loso.to_csv(OUT / "loso_all_seeds.csv", index=False)
cols = ["macro_f1", "micro_f1"] + L
loso_mean = loso.groupby(["model", "held_out"])[cols].mean()
loso_sd = loso.groupby(["model", "held_out"])["macro_f1"].std().rename("macro_f1_sd_over_seeds")
loso_summary = loso_mean.join(loso_sd).round(3)
loso_summary.to_csv(OUT / "loso_summary_mean_over_seeds.csv")
print("\nLOSO (mean over seeds)\n", loso_summary)

drop = []
for kind in ["word", "char"]:
    looo = loso[loso.model == kind].groupby("held_out")["macro_f1"].mean().mean()
    drop.append({"model": kind, "random_cv_macro_f1": res[kind]["macro_f1"],
                 "loso_mean_macro_f1": looo,
                 "absolute_drop": res[kind]["macro_f1"] - looo,
                 "relative_drop_%": 100 * (res[kind]["macro_f1"] - looo) / res[kind]["macro_f1"]})
drop = pd.DataFrame(drop).round(3)
drop.to_csv(OUT / "cv_vs_loso_drop.csv", index=False)
print("\nRandom CV vs LOSO\n", drop)

# 4. Per-frame Krippendorff's alpha -------------------------------------------
def alpha_nominal(units):
    units = [u for u in units if len(u) >= 2]
    vals = sorted({v for u in units for v in u})
    ix = {v: i for i, v in enumerate(vals)}
    o = np.zeros((len(vals), len(vals)))
    for u in units:
        for a in range(len(u)):
            for b in range(len(u)):
                if a != b:
                    o[ix[u[a]], ix[u[b]]] += 1 / (len(u) - 1)
    nc = o.sum(1); n = nc.sum()
    de = (n * n - (nc ** 2).sum()) / (n - 1)
    return 1.0 if de == 0 else 1 - (n - np.trace(o)) / de

multi = [r for r in E.rows if r["n_annotators"] >= 2]
alpha = {f: alpha_nominal([[int(f in a["frames"]) for a in r["annotations"].values()]
                           for r in multi])
         for f in L + ["Economic", "Security & defense"]}
pd.Series(alpha, name="alpha").round(2).to_csv(OUT / "alpha_per_frame.csv")
print("\nAlpha per frame:", {k: round(v, 2) for k, v in alpha.items()})

# 5. Outlet-name masking check ------------------------------------------------
n_odatv = sum(bool(re.search("odatv", r["content"], re.I)) for r in E.rows if r["source"] == "odatv")
n_other = sum(bool(re.search("odatv", r["content"], re.I)) for r in E.rows if r["source"] != "odatv")
original = list(E.texts)
E.texts[:] = [re.sub(r"odatv(\.com)?", " ", t, flags=re.I) for t in original]
main_out = E.OUT
E.OUT = OUT / "masked_run"; E.OUT.mkdir(exist_ok=True)   # do not overwrite main OOF files
masked, _ = E.evaluate("word")
E.OUT = main_out
E.texts[:] = original
mask = {"odatv_string_in_odatv_articles": n_odatv, "odatv_string_in_other_articles": n_other,
        "word_macro_f1_original": res["word"]["macro_f1"],
        "word_macro_f1_masked": masked["macro_f1"]}
print("\nMasking check:", mask)

json.dump({"baseline": baseline, "bootstrap_ci": ci.tolist(), "alpha": alpha,
           "masking": mask, "drop": drop.to_dict("records")},
          open(OUT / "supplementary_summary.json", "w"), indent=2, default=float)
print(f"\nSaved to {OUT}/")
