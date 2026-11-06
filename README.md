# CivicLens – Stage 1: Reproducibility Package

**Paper:** *CivicLens: Multimodal Classification and Urgency Prioritization of Citizen-Reported Urban Issues. Stage 1: Classifying Citizen Reports from Text on Open Municipal 311 Data*

**Author:** Aron Palamoottil Biju (Student ID 73706), Vistula University, Warsaw

**Course:** Architecture of Computers – Student Research Challenge 2026/27

This folder contains everything needed to check and reproduce the results in the paper: the code, the dependency list, how to get the data, all raw outputs, and the limitations.

---

## 1. Contents

| Folder or file | What it is |
|---|---|
| `paper/73706_PalamoottilBiju_CivicLens_SRC.pdf` | The submitted paper |
| `code/civiclens_experiment.py` | The main experiment (script version 3, as executed; only the header comment was corrected from "version 2" to "version 3") |
| `code/posthoc_analysis.py` | Re-creates the analyses done after the main run, from the saved outputs |
| `code/make_paper_figures.py` | Re-creates Figures 2 and 3 of the paper from the saved outputs |
| `code/CivicLens_reproduce.ipynb` | One-click Google Colab notebook that runs everything |
| `requirements.txt` | Library versions used in the reported run |
| `data/DATA.md` | Where the dataset comes from, its licence, and its checksum |
| `results/` | All raw outputs of the reported run (see Section 4) |

## 2. How to reproduce (Google Colab, about 30–60 minutes)

1. Open `code/CivicLens_reproduce.ipynb` in Google Colab: **File → Upload notebook**.
2. Set **Runtime → Change runtime type → T4 GPU**. A GPU is only needed to make the MiniLM encoding fast (about 1 minute instead of about 30); everything else runs on the CPU.
3. **To use the exact data file**, put it at the top of your Google Drive (*My Drive*) as `CivicLens_Open311.csv` (see `data/DATA.md`). Otherwise the notebook downloads the current portal file, which is newer and gives slightly different numbers.
4. Click **Runtime → Run all**, and allow Google Drive access when asked. Progress is saved to Drive, so if Colab disconnects, just run all the cells again and the run continues.
5. At the end, `civiclens_results.zip` downloads automatically.

**Or run it locally:**
```
pip install -r requirements.txt
python code/civiclens_experiment.py --csv CivicLens_Open311.csv --text description --label service_name --out results
python code/posthoc_analysis.py --results results --out results/posthoc
python code/make_paper_figures.py --results results --out results/paper_figures
```

## 3. Data

See `data/DATA.md`. In short:
- **Source:** City of Bloomington (Indiana) Open311, dataset `aw6y-t4ix`.
- **Licence:** PDDL 1.0 (public domain).
- **Copy:** for privacy, the file is not republished here (see Section 7). The author keeps the exact file and can share it on request; the checksum lets anyone verify it.
- **File:** the exact file used has SHA-256 `802e12249d4e088f3efc813378dd9e1fe126733945ccc2c1b1f814eb1354cc97`. The script records the checksum of whatever file it reads in `results/run_info.json`.

## 4. Where each number in the paper comes from

| Paper | File in `results/` |
|---|---|
| Table 1 (dataset and preprocessing) | `dataset_profile.json`, `preprocessing_log.json` |
| Table 2, Figure 2 (model comparison, 95% CIs) | `model_comparison.csv` |
| Table 3 (effect of balancing) | `paired_bootstrap.csv`, `model_comparison.csv` |
| Section 3.1 comparisons | `paired_bootstrap.csv`, `posthoc/extra_paired_bootstrap_posthoc.csv` |
| Section 3.2, ten smallest categories | `posthoc/small_category_recall.csv` |
| Section 3.3, Figure 3 (per-class results, confusions) | `per_class/*.csv`, `confusion/*.csv` |
| Section 4.1, F1 change by category size | `posthoc/posthoc_f1_change_by_category_size.csv` |
| Table A2 (selected hyperparameters) | `model_comparison.csv` (column `best_params`) |
| Run settings and versions | `run_info.json`, `pip_freeze.txt`, `console_log.txt` |
| Every test prediction | `test_predictions.csv` (report texts replaced by `group_id`, see Section 7), `preds/*.npy` |

## 5. Settings of the reported run

| Setting | Value |
|---|---|
| Random seed | 42 (split, folds, models, bootstrap) |
| Test set | The first fold of `StratifiedGroupKFold` with 5 folds (15,924 reports, 20.0%) |
| Near-duplicate group key | First six words of a text after removing all digits |
| Hyperparameter tuning | Grouped, stratified 3-fold CV on the training set, scored by macro-F1 |
| Confidence intervals | 2,000 cluster-bootstrap resamples of test groups |
| Small categories | Categories with fewer than 50 reports merged into the existing "Other" category |
| Sentence encoder | `sentence-transformers/all-MiniLM-L6-v2`, frozen |
| Environment | Python 3.13.16, scikit-learn 1.6.1, pandas 2.2.3, NumPy 2.4.6, sentence-transformers 5.7.0 |

**Run notes:**
- The run was interrupted once and resumed. Finished models were loaded from saved predictions, and the two slowest models (TF-IDF + LogReg) were trained last.
- Because the split and all seeds are fixed, this does not change any result.
- An earlier run with a larger search grid was stopped by a Colab disconnect and is not reported.

## 6. Limitations

These are also stated in Section 4.3 of the paper.
- **One city:** the data come from one US city, in English, with one category scheme, so the results may not transfer to other cities (e.g. Warsaw 19115).
- **Random split:** the train/test split is random, not by time. The confidence intervals cover only sampling variation of this test set.
- **Small search grid:** a smaller search grid and 3-fold CV were used, to fit a free Colab session.
- **Naive Bayes weights:** during tuning of balanced Naive Bayes, the class weights were computed from the whole training set rather than within each CV fold. This did not involve the test set, but it may have influenced the selected alpha. *This was not corrected.*
- **Near-duplicate key:** the six-word grouping key may miss some near-duplicates.
- **Frozen MiniLM:** MiniLM was not fine-tuned.
- **No urgency labels:** the data have no urgency labels, so urgency was not tested.
- **Post-hoc analyses:** several analyses were done after the main analysis; they are marked as such.

## 7. Privacy

The report texts are public on the city's portal, but some contain names, home addresses, e-mail addresses or phone numbers written by citizens. To avoid republishing personal information, this repository contains **no report texts**:
- `results/test_predictions.csv` has a `group_id` column (an anonymous number for each test report's near-duplicate group) instead of the text.
- The dataset file itself is not included (see `data/DATA.md`).

`code/posthoc_analysis.py` uses `group_id` and gives exactly the same results as with the texts. If you run `civiclens_experiment.py` yourself, it writes the texts again into your own output folder; do not republish them.

## 8. AI use

Claude (Anthropic) and ChatGPT (OpenAI) were used under the author's direction. See the declaration in the paper.
