"""
CivicLens - analyses computed after the main run (reported in Sections 3-4 and Appendix C).
Reads only the saved outputs of civiclens_experiment.py; it does not retrain anything.

Usage:  python posthoc_analysis.py --results results --out results/posthoc
Writes:
  extra_paired_bootstrap_posthoc.csv          paired cluster-bootstrap comparisons not in paired_bootstrap.csv
  small_category_recall.csv                   recall of the 10 smallest test categories, unweighted vs balanced
  posthoc_f1_change_by_category_size.csv      mean per-class F1 change (balanced - unweighted), smaller vs larger half
"""
import argparse, importlib.util, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/posthoc")
ap.add_argument("--script", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "civiclens_experiment.py"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# reuse the exact bootstrap functions of the main script
spec = importlib.util.spec_from_file_location("ce", a.script)
ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)

tp = pd.read_csv(f"{a.results}/test_predictions.csv", keep_default_na=False)
labels = sorted(pd.read_csv(f"{a.results}/class_distribution.csv", keep_default_na=False)["label"])
code = {l: i for i, l in enumerate(labels)}
y = np.array([code[v] for v in tp["true"]])
models = [c for c in tp.columns if c not in ("text", "true", "group_id")]
P = {m: np.array([code[v] for v in tp[m].astype(str)]) for m in models}
# the public test_predictions.csv has no report texts (privacy); group_id identifies the near-duplicate group
groups = tp["group_id"].values if "group_id" in tp.columns else tp["text"].map(ce.group_key).values
idx = ce.bootstrap_idx(groups, np.random.default_rng(ce.SEED))

# 1. extra paired comparisons
pairs = [("TF-IDF + LinearSVM", "TF-IDF + NB"), ("TF-IDF + LinearSVM", "MiniLM + LogReg"),
         ("TF-IDF + LinearSVM (balanced)", "MiniLM + LogReg"), ("TF-IDF + LinearSVM (balanced)", "TF-IDF + NB (balanced)"),
         ("TF-IDF + LogReg", "TF-IDF + NB"), ("MiniLM + LogReg", "TF-IDF + NB")]
rows = []
for m1, m2 in pairs:
    o, lo, hi, p = ce.paired(y, P[m1], P[m2], idx, labels)
    rows.append(dict(comparison=f"{m1} - {m2}", delta_macro_f1=round(o, 4), ci_low=round(lo, 4), ci_high=round(hi, 4), p=round(p, 3)))
pd.DataFrame(rows).to_csv(f"{a.out}/extra_paired_bootstrap_posthoc.csv", index=False)

# 2. recall of the 10 smallest test categories
support = tp["true"].value_counts().sort_values()
small = support.index[:10]
t = pd.DataFrame({"support": support[small].astype(int)})
for m in models:
    if m == "Majority baseline":
        continue
    t[m] = [round(float((tp.loc[tp["true"] == c, m] == c).mean()), 3) for c in small]
t.loc["MEAN"] = [np.nan] + [round(float(t[c].mean()), 3) for c in t.columns[1:]]
t.to_csv(f"{a.out}/small_category_recall.csv")

# 3. per-class F1 change by category size
def per_class(m):
    p = pd.read_csv(f"{a.results}/per_class/{ce.safe(m)}.csv", index_col=0)
    return p.drop(index=[i for i in p.index if i in ("accuracy", "macro avg", "weighted avg")])
rows = []
for base in ["TF-IDF + NB", "TF-IDF + LogReg", "TF-IDF + LinearSVM", "MiniLM + LogReg"]:
    A, B = per_class(base), per_class(base + " (balanced)")
    d = B["f1-score"] - A["f1-score"]; smaller = A["support"] <= A["support"].median()
    rows.append(dict(model=base, mean_dF1_smaller_half=round(d[smaller].mean(), 4), mean_dF1_larger_half=round(d[~smaller].mean(), 4),
                     n_smaller=int(smaller.sum()), n_larger=int((~smaller).sum())))
pd.DataFrame(rows).to_csv(f"{a.out}/posthoc_f1_change_by_category_size.csv", index=False)
print("written to", a.out)
