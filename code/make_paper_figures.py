"""
CivicLens - recreates Figure 2 (macro-F1 with 95% CIs) and Figure 3 (confusion matrix, 12 largest
categories, best model) of the paper from the saved result files.

Usage:  python make_paper_figures.py --results results --out results/paper_figures
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/paper_figures")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# Figure 2
mc = pd.read_csv(f"{a.results}/model_comparison.csv")
order = ["Majority baseline", "TF-IDF + NB", "TF-IDF + NB (balanced)", "TF-IDF + LogReg", "TF-IDF + LogReg (balanced)",
         "TF-IDF + LinearSVM", "TF-IDF + LinearSVM (balanced)", "MiniLM + LogReg", "MiniLM + LogReg (balanced)"]
d = mc.set_index("model").loc[order[::-1]]
col = ["#B07A1E" if "balanced" in m else "#2F5D8A" for m in d.index]; col[-1] = "#999999"
fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
ax.barh(d.index, d.macro_f1, color=col, xerr=[d.macro_f1 - d.macro_f1_ci_low, d.macro_f1_ci_high - d.macro_f1],
        capsize=2.5, error_kw=dict(lw=0.8))
for i, (v, h) in enumerate(zip(d.macro_f1, d.macro_f1_ci_high)):
    ax.text(h + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
ax.set_xlim(0, 0.62); ax.set_xlabel("Test macro-F1 (bars: 95% cluster-bootstrap CI)", fontsize=8); ax.tick_params(labelsize=7.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Patch(color="#2F5D8A", label="unweighted"), Patch(color="#B07A1E", label="class-balanced")],
          fontsize=7, loc="upper right", frameon=False)
fig.tight_layout(); fig.savefig(f"{a.out}/fig2_macro_f1.png"); plt.close()

# Figure 3
cm = pd.read_csv(f"{a.results}/confusion/TF_IDF_LogReg_balanced.csv", index_col=0)
top = cm.sum(axis=1).sort_values(ascending=False).index[:12]
sub = cm.loc[top, top].div(cm.loc[top].sum(axis=1), axis=0)          # rows normalised over all 56 categories
short = {"Potholes, Other Street Repair": "Potholes/street repair", "Debris Removal (Sand, General Street Debris)": "Debris removal",
         "Temporary Signage w/o permit": "Temp. signage", "Sewer Problems (Sanitary Sewers)": "Sewer problems",
         "Water Utility Problems": "Water utility"}
lab = [short.get(x, x) for x in top]
fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=200)
im = ax.imshow(sub.values, cmap="Blues", vmin=0, vmax=1)
for i in range(len(top)):
    for j in range(len(top)):
        v = sub.values[i, j]
        if v >= 0.02:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.5 else "black")
ax.set_xticks(range(len(top)), lab, rotation=60, ha="right", fontsize=6.5); ax.set_yticks(range(len(top)), lab, fontsize=6.5)
ax.set_xlabel("Predicted category", fontsize=7.5); ax.set_ylabel("True category", fontsize=7.5)
fig.colorbar(im, fraction=0.04).ax.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig(f"{a.out}/fig3_confusion_top12.png"); plt.close()
print("written to", a.out)
