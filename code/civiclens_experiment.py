"""
CivicLens - text classification experiment on open 311 data (version 3)
Author: Aron Palamoottil Biju (Vistula University, student ID 73706)

Design (matches Section 2 of the paper):
  1. Profile the raw CSV (row counts, missing values, class counts, duplicates).
  2. Clean the text. Remove empty texts and texts that only repeat their label.
  3. Resolve duplicates BEFORE splitting: identical texts are kept once, with
     their most frequent label (texts whose labels tie are removed).
  4. Merge categories with fewer than MIN_CLASS reports into "Other"
     (no report is dropped because of its label).
  5. Near-duplicate groups: texts that share their first 6 words after all
     digits are removed (e.g. one template with different house numbers)
     form one group. A group is never split between train and test.
  6. ONE fixed stratified, group-aware split (~80/20, seed 42). The test set is
     never used for any choice; it is evaluated once per model at the end.
  7. Hyperparameters: stratified group 3-fold CV on the training set, macro-F1.
     (v3: smaller grid and capped vocabulary so the run fits in a free Colab
     session; every finished model is saved at once and the run can resume.)
  8. Every learning model in two versions: unweighted and class-balanced.
  9. Uncertainty: 95% cluster-bootstrap confidence intervals (test groups are
     resampled) and paired comparisons between models (2,000 resamples).

Usage (Google Colab, CPU is enough):
  !pip install sentence-transformers
  !python civiclens_experiment.py --csv Open311.csv --text description --label service_name
  (add --no_embed to skip the MiniLM model)
"""

import argparse, hashlib, json, os, re, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, frozen
N_BOOT = 2000
CV_FOLDS = 3


# ------------------------------------------------------------ 1. data
def clean_text(s):
    """Lower-case; remove URLs, e-mails and digit runs of 4+ (phones, IDs)."""
    s = str(s).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    s = re.sub(r"\d{4,}", " ", s)
    s = re.sub(r"[^a-z0-9\s'.,!?-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def group_key(text, n_words=6):
    """Near-duplicate key: remove ALL digits, keep the first n_words words."""
    return " ".join(re.sub(r"\d+", " ", text).split()[:n_words])


def profile(raw, text_col, label_col):
    """Statistics of the downloaded file itself (not portal metadata)."""
    desc = raw[text_col]
    empty = desc.isna() | (desc.astype(str).str.strip() == "")
    vc = desc[~empty].astype(str).str.strip().value_counts()
    return {
        "rows": int(len(raw)),
        "missing_or_empty_description": int(empty.sum()),
        "missing_label": int(raw[label_col].isna().sum()),
        "n_categories_raw": int(raw[label_col].nunique()),
        "top10_categories": raw[label_col].value_counts().head(10).to_dict(),
        "most_repeated_description": vc.index[0] if len(vc) else None,
        "most_repeated_description_count": int(vc.iloc[0]) if len(vc) else 0,
        "rows_whose_text_is_repeated": int(vc[vc > 1].sum()),
    }


def prepare(raw, text_col, label_col, min_class, log):
    df = raw[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(str).str.strip()
    df["text"] = df["text"].fillna("").map(clean_text)
    df = df[df["text"].str.len() > 0]
    log["after_removing_empty_text"] = len(df)

    same = df["text"] == df["label"].str.lower().map(clean_text)
    log["removed_text_equals_label"] = int(same.sum())
    df = df[~same]

    # Duplicates -> one row per unique text with its most frequent label.
    counts = df.groupby(["text", "label"]).size().rename("n").reset_index()
    counts = counts.sort_values(["text", "n"], ascending=[True, False])
    top = counts.groupby("text").head(2)
    tie = top.groupby("text")["n"].transform(lambda x: len(x) == 2 and x.iloc[0] == x.iloc[1])
    tied_texts = set(top.loc[tie.astype(bool), "text"])
    log["removed_texts_with_tied_labels"] = len(tied_texts)
    n_conflict = int((df.groupby("text")["label"].nunique() > 1).sum())
    log["unique_texts_with_more_than_one_label"] = n_conflict
    before = len(df)
    df = counts.groupby("text").head(1)
    df = df[~df["text"].isin(tied_texts)][["text", "label"]]
    log["rows_removed_by_deduplication"] = before - len(df)

    # Rare categories -> "Other". Uses only label counts; no text, no model.
    vc = df["label"].value_counts()
    rare = vc[vc < min_class].index
    log["categories_merged_into_Other"] = sorted(rare.tolist())
    log["rows_relabelled_Other"] = int(df["label"].isin(rare).sum())
    df.loc[df["label"].isin(rare), "label"] = "Other"
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    log["final_rows"] = len(df)
    log["final_categories"] = int(df["label"].nunique())
    return df


# ------------------------------------------------------------ 2. models
def tfidf():
    # inside a Pipeline -> fitted on the training folds only
    # unigrams + bigrams, terms in >= 3 training documents, at most 100,000 terms
    return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100_000)


def model_list():
    """(name, estimator, grid, input, balanced). 'balanced' means class-balanced
    sample weights for NB (it has no class_weight option) and
    class_weight='balanced' for LogReg and SVM. Both give each class the same
    total weight: n_samples / (n_classes * n_class_samples)."""
    m = [("Majority baseline", DummyClassifier(strategy="most_frequent"), {}, "raw", False)]
    for bal in (False, True):
        tag = " (balanced)" if bal else ""
        cw = "balanced" if bal else None
        m += [
            ("TF-IDF + NB" + tag, Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
             {"clf__alpha": [0.01, 0.1, 1.0]}, "text", bal),
            ("TF-IDF + LogReg" + tag, Pipeline([("tfidf", tfidf()),
             ("clf", LogisticRegression(max_iter=1000, class_weight=cw, random_state=SEED))]),
             {"clf__C": [1.0, 10.0]}, "text", False),
            ("TF-IDF + LinearSVM" + tag, Pipeline([("tfidf", tfidf()),
             ("clf", LinearSVC(class_weight=cw, random_state=SEED))]),
             {"clf__C": [0.1, 0.5]}, "text", False),
            ("MiniLM + LogReg" + tag, LogisticRegression(max_iter=1000, class_weight=cw, random_state=SEED),
             {"C": [1.0, 10.0]}, "emb", False),
        ]
    return [x for x in m if "TF-IDF + LogReg" not in x[0]] + [x for x in m if "TF-IDF + LogReg" in x[0]]   # slow models last


def encode(texts, out_dir):
    """Frozen pretrained encoder: weights are NOT updated and labels are never
    seen, so train and test texts are encoded independently with the same fixed
    function. Mean pooling, L2-normalised, 384 dimensions; texts longer than
    256 word pieces are truncated by the model."""
    key = hashlib.md5("\n".join(texts).encode()).hexdigest()[:12]
    path = f"{out_dir}/emb_{key}.npy"
    if os.path.exists(path):
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    import torch
    enc = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    X = enc.encode(list(texts), batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    np.save(path, X)
    return X


# ------------------------------------------------------------ 3. metrics
def scores(y, p):
    return {"accuracy": accuracy_score(y, p),
            "macro_precision": precision_score(y, p, average="macro", zero_division=0),
            "macro_recall": recall_score(y, p, average="macro", zero_division=0),
            "macro_f1": f1_score(y, p, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, p, average="weighted", zero_division=0)}


def bootstrap_idx(groups, rng):
    """Cluster bootstrap: resample whole near-duplicate groups of the test set."""
    members = pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).apply(np.array).tolist()
    out = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(members), len(members))
        out.append(np.concatenate([members[i] for i in pick]))
    return out


def fast_macro_f1(yt, yp, k):
    """Macro-F1 over k integer-coded classes; F1 = 2TP / (n_pred + n_true),
    and 0 for a class absent from both (same as sklearn, zero_division=0)."""
    tp = np.bincount(yt[yt == yp], minlength=k)
    denom = np.bincount(yp, minlength=k) + np.bincount(yt, minlength=k)
    f = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return f.mean()


def ci(y, p, idx, labels):
    k = len(labels)
    v = [fast_macro_f1(y[i], p[i], k) for i in idx]
    return np.percentile(v, [2.5, 97.5])


def paired(y, pa, pb, idx, labels):
    """Difference A - B in macro-F1 on the same resamples; two-sided p-value =
    2 * share of resamples on the 'wrong' side of zero (bounded by 1)."""
    k = len(labels)
    d = np.array([fast_macro_f1(y[i], pa[i], k) - fast_macro_f1(y[i], pb[i], k) for i in idx])
    obs = fast_macro_f1(y, pa, k) - fast_macro_f1(y, pb, k)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return obs, lo, hi, p


def safe(n):
    return re.sub(r"[^A-Za-z0-9]+", "_", n).strip("_")


# ------------------------------------------------------------ 4. main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text", default="description")
    ap.add_argument("--label", default="service_name")
    ap.add_argument("--out", default="results")
    ap.add_argument("--min_class", type=int, default=50)
    ap.add_argument("--no_embed", action="store_true")
    a = ap.parse_args()
    for sub in ["", "/per_class", "/confusion", "/figures"]:
        os.makedirs(a.out + sub, exist_ok=True)

    import sklearn, platform
    run_info = {"csv_sha256": hashlib.sha256(open(a.csv, "rb").read()).hexdigest(),
                "csv_modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(a.csv))),
                "python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "scikit_learn": sklearn.__version__, "seed": SEED, "n_boot": N_BOOT, "cv_folds": CV_FOLDS, "script_version": "v3"}
    if not a.no_embed:
        import sentence_transformers
        run_info["sentence_transformers"] = sentence_transformers.__version__
    json.dump(run_info, open(f"{a.out}/run_info.json", "w"), indent=2)
    raw = pd.read_csv(a.csv, low_memory=False)
    json.dump(profile(raw, a.text, a.label), open(f"{a.out}/dataset_profile.json", "w"), indent=2, default=str)
    log = {}
    df = prepare(raw, a.text, a.label, a.min_class, log)
    df["label"].value_counts().rename("count").to_csv(f"{a.out}/class_distribution.csv")

    # ---- one fixed stratified, group-aware split (first of 5 group folds = test)
    df["group"] = df["text"].map(group_key)
    gsizes = df["group"].value_counts()
    log.update(near_duplicate_groups=int(len(gsizes)), largest_group=int(gsizes.iloc[0]),
               rows_in_multi_row_groups=int(gsizes[gsizes > 1].sum()))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, te = next(splitter.split(df["text"], df["label"], groups=df["group"]))
    Xtr, Xte = df["text"].values[tr], df["text"].values[te]
    ytr, yte = df["label"].values[tr], df["label"].values[te]
    gtr, gte = df["group"].values[tr], df["group"].values[te]
    assert not set(Xtr) & set(Xte), "leakage: identical text in train and test"
    assert not set(gtr) & set(gte), "leakage: near-duplicate group in train and test"
    missing = sorted(set(df["label"]) - set(yte))
    log.update(train_rows=len(tr), test_rows=len(te), test_share=round(len(te) / len(df), 4),
               categories_missing_from_test=missing)
    json.dump(log, open(f"{a.out}/preprocessing_log.json", "w"), indent=2)
    print(json.dumps(log, indent=2), flush=True)

    Etr = Ete = None
    if not a.no_embed:
        Etr, Ete = encode(list(Xtr), a.out), encode(list(Xte), a.out)

    labels = sorted(df["label"].unique())
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    os.makedirs(f"{a.out}/preds", exist_ok=True)
    rows, preds = [], {}
    for name, est, grid, kind, nb_bal in model_list():
        if kind == "emb" and Etr is None:
            continue
        pfile, rfile = f"{a.out}/preds/{safe(name)}.npy", f"{a.out}/preds/{safe(name)}.json"
        if os.path.exists(pfile) and os.path.exists(rfile):      # resume after a disconnect
            preds[name] = np.load(pfile, allow_pickle=True)
            rows.append(json.load(open(rfile)))
            print(f"{name:30s} loaded from earlier run", flush=True)
            continue
        A, B = (Etr, Ete) if kind == "emb" else (Xtr, Xte)
        if kind == "raw":
            A, B = np.zeros((len(Xtr), 1)), np.zeros((len(Xte), 1))
        fit_kw = {"clf__sample_weight": compute_sample_weight("balanced", ytr)} if nb_bal else {}
        t0 = time.time()
        gs = GridSearchCV(est, grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=3)
        gs.fit(A, ytr, groups=gtr, **fit_kw)
        p = gs.predict(B)
        preds[name] = p
        rows.append({"model": name, "cv_macro_f1": round(gs.best_score_, 4),
                     "best_params": json.dumps({k: str(v) for k, v in gs.best_params_.items()}),
                     **{k: round(v, 4) for k, v in scores(yte, p).items()},
                     "fit_seconds": round(time.time() - t0, 1)})
        np.save(pfile, p); json.dump(rows[-1], open(rfile, "w"))
        print(f"{name:30s} test macro-F1 {rows[-1]['macro_f1']:.4f} (CV {rows[-1]['cv_macro_f1']:.4f}) "
              f"{rows[-1]['fit_seconds']:.0f}s", flush=True)

    # ---- uncertainty on the fixed test set
    rng = np.random.default_rng(SEED)
    idx = bootstrap_idx(gte, rng)
    code = {l: i for i, l in enumerate(labels)}
    yte_i = np.array([code[v] for v in yte])
    preds_i = {m: np.array([code[v] for v in p]) for m, p in preds.items()}
    res = pd.DataFrame(rows)
    cis = [ci(yte_i, preds_i[m], idx, labels) for m in res["model"]]
    res["macro_f1_ci_low"] = [round(c[0], 4) for c in cis]
    res["macro_f1_ci_high"] = [round(c[1], 4) for c in cis]
    res.to_csv(f"{a.out}/model_comparison.csv", index=False)

    comp = []
    for base in ["TF-IDF + NB", "TF-IDF + LogReg", "TF-IDF + LinearSVM", "MiniLM + LogReg"]:
        if base in preds and base + " (balanced)" in preds:
            o, lo, hi, p = paired(yte_i, preds_i[base + " (balanced)"], preds_i[base], idx, labels)
            comp.append({"comparison": f"{base}: balanced - unweighted", "delta_macro_f1": o, "ci_low": lo, "ci_high": hi, "p": p})
    ranked = res[res.model != "Majority baseline"].sort_values("macro_f1", ascending=False)["model"].tolist()
    for other in ranked[1:]:
        o, lo, hi, p = paired(yte_i, preds_i[ranked[0]], preds_i[other], idx, labels)
        comp.append({"comparison": f"{ranked[0]} - {other}", "delta_macro_f1": o, "ci_low": lo, "ci_high": hi, "p": p})
    pd.DataFrame(comp).round(4).to_csv(f"{a.out}/paired_bootstrap.csv", index=False)

    # ---- per-class, confusion matrices, predictions, figures
    for name, p in preds.items():
        pd.DataFrame(classification_report(yte, p, labels=labels, output_dict=True, zero_division=0)).T \
          .round(4).to_csv(f"{a.out}/per_class/{safe(name)}.csv")
        pd.DataFrame(confusion_matrix(yte, p, labels=labels), index=labels, columns=labels) \
          .to_csv(f"{a.out}/confusion/{safe(name)}.csv")
    pd.DataFrame({"text": Xte, "true": yte, **preds}).to_csv(f"{a.out}/test_predictions.csv", index=False)

    best = ranked[0]
    cm = confusion_matrix(yte, preds[best], labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(0.45 * len(labels) + 3, 0.45 * len(labels) + 2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Row-normalised confusion matrix: {best}", fontsize=9)
    fig.colorbar(im, fraction=0.04); fig.tight_layout(); fig.savefig(f"{a.out}/figures/confusion_best.png", dpi=200)

    r = res.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(r) + 1))
    ax.barh(r["model"], r["macro_f1"], xerr=[r["macro_f1"] - r["macro_f1_ci_low"], r["macro_f1_ci_high"] - r["macro_f1"]],
            color="#2F5D8A", capsize=3)
    ax.set_xlabel("Test macro-F1 (95% bootstrap CI)"); fig.tight_layout()
    fig.savefig(f"{a.out}/figures/macro_f1_by_model.png", dpi=200)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
