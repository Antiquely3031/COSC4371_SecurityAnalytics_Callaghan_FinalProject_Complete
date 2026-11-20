"""
Created by Patrick Callaghan (2523109)

COSC4371 Security Analytics - Final Checkpoint
Project: #15 Variant (Safe-Guard only)
"""

import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
import sklearn
from datasets import load_dataset
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.base import clone

from attacks import ATTACKS, ALL_ATTACKS, TEMPLATES, SWEEP_ATTACKS, SWEEP_RATES, attack_injections, attack_text
from defenses import normalize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
COS_DUP_SIMILARITY = 0.90
TARGET_FPR = 0.05

# Load Safe-Guard Dataset
ds = load_dataset("xTRam1/safe-guard-prompt-injection")
parts = []
for split in ds:
    df = ds[split].to_pandas()
    df["orig_split"] = split
    df["row_id"] = [f"Safe-Guard:{split}:{i}" for i in range(len(df))]
    parts.append(df)
data = pd.concat(parts, ignore_index=True)

# Audited Data
print("\n== AUDIT ==")
print("Rows, Columns:", len(data), list(data.columns[:2]))
print("Class Counts:", data["label"].value_counts().to_dict())
print("Class Percentage:", (100 * data["label"].value_counts(normalize=True)).round(1).to_dict())
print("Missing Label(s):", data["label"].isna().sum(), "| Missing Text(s): ", data["text"].isna().sum())

# Find Identical 'Text' String(s)
counts = data["text"].value_counts()
duplicates = counts[counts > 1]
print(f"Exact Duplicates: {duplicates.sum()} rows in {len(duplicates)} groups ({duplicates.sum() - len(duplicates)} extra copies)")

# Find Near Duplicates (where cosine similarity >= 0.90 for 3-5-gram TF-IDF vectors,
# post NFKC normalization, lowercasing and collapsing whitespace
def clean_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()

# Vectorize
X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(data["text"].map(clean_text))
rows, cols = [], []
for start in range(0, X.shape[0], 500):
    sims = (X[start:start + 500] @ X.T).toarray()
    r, c = np.nonzero(sims >= COS_DUP_SIMILARITY)
    rows.append(r + start)
    cols.append(c)
graph = coo_matrix((np.ones(sum(map(len, rows))), (np.concatenate(rows), np.concatenate(cols))), shape=(len(data), len(data)))
_, data["group"] = connected_components(graph, directed=False)
gsize = data["group"].map(data["group"].value_counts())
print(f"Near Duplicates (includes exact duplicates): {(gsize > 1).sum()} rows in {data.loc[gsize > 1, 'group'].nunique()} groups")

# Removed Data
print("\n== REMOVALS ==")
n0 = len(data)

# Keep one copy of exact duplicates, prefer copy from publisher's test split
data = data.sort_values("orig_split", kind="stable")
data = data.drop_duplicates("text", keep="first")
print("Number of Copies of Exact Duplicates that were Removed:", n0 - len(data))

# In near duplicates, drop train rows whose group also has a test row
group_has_test = data.groupby("group")["orig_split"].transform(lambda s: (s == "test").any())
drop = group_has_test & (data["orig_split"] == "train")
data = data[~drop].copy()
print("Near-Duplicate Train Rows Removed:", drop.sum())

# Validation: take 1 of 7 folds of train, whole groups only, balanced labels
data["split"] = data["orig_split"]
train_idx = data.index[data["split"] == "train"]
folds = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
_, val_pos = next(folds.split(train_idx, data.loc[train_idx, "label"], data.loc[train_idx, "group"]))
data.loc[train_idx[val_pos], "split"] = "validation"
print(data.groupby(["split", "label"]).size().unstack())

# Check Leakage
train_text = data.loc[data["split"] == "train", "text"]
leak_test = train_text.isin(data.loc[data["split"] == "test", "text"]).sum()
leak_validation = train_text.isin(data.loc[data["split"] == "validation", "text"]).sum()
print(f"Leakage (train rows also in test): {leak_test}") 
print(f"Leakage (train rows also in validation): {leak_validation}")
if leak_test or leak_validation:
    sys.exit("Leakage found; stopping.")
data["dataset"] = "safeguard"
data[["row_id", "dataset", "split", "label"]].to_csv("split.csv", index=False)

def score(model, texts):
    """Higher = more likely injection. LR gives a probability; the SVM gives a signed distance."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)

def choose_threshold(benign_scores, target_fpr):
    """Lowest 6-decimal threshold with at most target_fpr of benign scores >= it."""
    allowed = int(np.floor(target_fpr * len(benign_scores)))   
    first_blocked = np.sort(benign_scores)[::-1][allowed]      
    return (np.floor(first_blocked * 1e6) + 1) / 1e6           


def plot_figures(model_name, all_scores, threshold):
    """ROC and PR figures for one model: one panel per attack family."""

    families = ["encoding", "structure", "context"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]          
    styles = ["-", "--", "-."]                          
    pct = 100 * test["label"].mean()
    x_min = 1 / (test["label"] == 0).sum() / 2          
     
    def draw(ax, test_set, label, color, style, width):
        g = all_scores[all_scores["value"] == test_set]
        fpr, tpr, _ = roc_curve(g["label"], g["score"])
        ax.plot(np.clip(fpr, x_min, 1), tpr, style, color=color, lw=width, label=label)
        pred = g["score"] >= threshold                  
        op_fpr = pred[g["label"] == 0].mean()
        op_recall = pred[g["label"] == 1].mean()
        ax.plot(max(op_fpr, x_min), op_recall, "o", ms=6, color=color, mec="white", mew=1.5, zorder=5)
     
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, family in zip(axes, families):
        xs = np.logspace(np.log10(x_min), 0, 200)
        ax.plot(xs, xs, ":", color="#898781", lw=1, label="Random scorer")
        draw(ax, "clean", "Clean (no attack)", "#0b0b0b", "-", 2.5)
        names = [n for n, (f, _) in ATTACKS.items() if f == family]
        for name, color, style in zip(names, colors, styles):
            draw(ax, name, name.replace("_", " "), color, style, 1.8)
        ax.set_xscale("log")
        ax.set_xlim(x_min, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{family.capitalize()} attacks", loc="left")
        ax.set_xlabel("False-positive rate (log scale)")
        ax.grid(True, color="#e1e0d9", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    axes[0].set_ylabel("True-positive rate (recall)")
    fig.suptitle(f"{model_name} | Safe-Guard test set: {len(test)} rows, {pct:.1f}% injections; "
                 f"dots = threshold {threshold:.6f}, no defense", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"roc_{model_name}.png", dpi=200)

    pos_share = test["label"].mean()                     
     
    def draw_pr(ax, test_set, label, color, style, width):
        g = all_scores[all_scores["value"] == test_set]
        precision, recall, _ = precision_recall_curve(g["label"], g["score"])
        ax.plot(recall, precision, style, color=color, lw=width, label=label, drawstyle="steps-post")
        pred = g["score"] >= threshold
        ax.plot(pred[g["label"] == 1].mean(), g["label"][pred].mean(),
                "o", ms=6, color=color, mec="white", mew=1.5, zorder=5)
     
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, family in zip(axes, families):
        ax.axhline(pos_share, ls=":", color="#898781", lw=1, label=f"Random scorer ({pos_share:.3f})")
        draw_pr(ax, "clean", "Clean (no attack)", "#0b0b0b", "-", 2.5)
        names = [n for n, (f, _) in ATTACKS.items() if f == family]
        for name, color, style in zip(names, colors, styles):
            draw_pr(ax, name, name.replace("_", " "), color, style, 1.8)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{family.capitalize()} attacks", loc="left")
        ax.set_xlabel("Recall")
        ax.grid(True, color="#e1e0d9", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    axes[0].set_ylabel("Precision")
    fig.suptitle(f"{model_name} | Safe-Guard test set: {len(test)} rows, {pct:.1f}% injections; "
                 f"dots = threshold {threshold:.6f}, no defense", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"pr_{model_name}.png", dpi=200)
    plt.close("all")

def plot_sweep(all_scores):
    """Recall against attack strength for each model, with the 0.05 'yes' bar."""
    rates = sorted([0.0, 0.3] + SWEEP_RATES)               # 0 = clean, 0.3 = main result
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # blue, orange, aqua, violet
    markers = ["o", "s", "^", "D"]                         # shapes too, so lines differ without color
    columns = list(dict.fromkeys(all_scores["column"]))    # one panel per model, in run order

    fig, axes = plt.subplots(1, len(columns), figsize=(11, 4.2), sharey=True)
    for ax, column in zip(axes, columns):
        g = all_scores[all_scores["column"] == column]

        def recall_of(value):
            inj = g[(g["value"] == value) & (g["label"] == 1)]
            return (inj["score"] >= inj["threshold"]).mean()

        clean = recall_of("clean")
        for name, color, marker in zip(SWEEP_ATTACKS, colors, markers):
            values = ["clean" if r == 0 else name if r == 0.3 else f"{name}_{r}" for r in rates]
            ax.plot([100 * r for r in rates], [recall_of(v) for v in values],
                    marker=marker, color=color, lw=2, ms=6, label=name.replace("_", " "))
        ax.axhline(clean - 0.05, ls="--", color="#898781", lw=1, label="Yes bar (clean - 0.05)")
        ax.axvline(30, ls=":", color="#c3c2b7", lw=1)       # the main, pre-specified strength
        ax.set_title(column.split("/")[0], loc="left")
        ax.set_xlabel("Attack strength (% of words, or letters for homoglyph)")
        ax.set_xticks([100 * r for r in rates])
        ax.set_ylim(0, 1.02)
        ax.grid(True, color="#e1e0d9", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    axes[0].set_ylabel("Recall at the validation threshold")
    fig.suptitle("Strength sweep, no defense: dotted line = main strength (30%)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig("sweep.png", dpi=200)
    plt.close(fig)

def score_column(model, column, threshold, prep=None):
    """Score clean, every attack and every sweep strength for one model and one defense."""
    variants = ([("clean", None, None)]
                + [(name, name, None) for name in ALL_ATTACKS]
                + [(f"{name}_{rate}", name, rate) for name in SWEEP_ATTACKS for rate in SWEEP_RATES])
    tables = []
    for value, attack, rate in variants:
        texts = test["text"] if attack is None else attack_injections(test, attack, rate=rate)["text"]
        if prep is not None:
            texts = texts.map(prep)
        scores = score(model, texts)
        recall = (scores[test["label"] == 1] >= threshold).mean()
        print(f"{value}: recall {recall:.4f}")
        tables.append(pd.DataFrame({"row_id": test["row_id"], "value": value, "column": column,
                                    "label": test["label"], "score": scores, "threshold": threshold}))
    return tables

def augment(train_df, seed=SEED):
    """Training set plus one obfuscated copy of every injection and of an equal number of benign prompts."""
    rng = np.random.default_rng(seed)
    injections = train_df[train_df["label"] == 1]
    benign = train_df[train_df["label"] == 0].sample(len(injections), random_state=seed)
    copies = []
    # Injections get any attack; benign prompts get only the character-level ones, so odd characters
    # appear in both classes. (Wrapping a benign prompt in "decode this and follow it" would mislabel it.)
    for rows, names in ((injections, list(ATTACKS)), (benign, SWEEP_ATTACKS)):
        for row_id, text, label in zip(rows["row_id"], rows["text"], rows["label"]):
            name = names[rng.integers(len(names))]
            rate = round(float(rng.uniform(0.1, 1.0)), 2) if name in SWEEP_ATTACKS else None
            copies.append({"text": attack_text(text, name, row_id, variant="train", rate=rate), "label": label})
    return pd.concat([train_df[["text", "label"]], pd.DataFrame(copies)], ignore_index=True)

# Establish Baseline
train, validation, test = (data[data["split"] == s] for s in ("train", "validation", "test"))
MODELS = {
    "tfidf_lr": make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
    "char_svm": make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        LinearSVC(C=1.0, random_state=SEED)),
}

aug_train = augment(train)
print(f"\nAugmented training set: {len(train)} original + {len(aug_train) - len(train)} obfuscated copies")

aug_leak_test = aug_train["text"].isin(test["text"]).sum()
aug_leak_val = aug_train["text"].isin(validation["text"]).sum()
print(f"Augmented leakage: train rows also in test = {aug_leak_test}, also in validation = {aug_leak_val}")

all_tables = []
for model_name, model in MODELS.items():
    t0 = time.time()
    model.fit(train["text"], train["label"])
    print(f"\n== {model_name} ==\nTraining Time: {time.time() - t0:.2f} s")
    print("Vocabulary size:", len(model[0].vocabulary_))

    # Threshold: lowest value, set to 6 decimals, where 0.05, at most, of validation benign prompts score >= it
    benign = score(model, validation.loc[validation["label"] == 0, "text"])
    threshold = choose_threshold(benign, TARGET_FPR)
    print(f"Threshold: {threshold:.6f}")
    
    # Score test set once  
    test_scores = score(model, test["text"])                   
    
    score_tables = [pd.DataFrame({"row_id": test["row_id"], "value": "clean", "column": f"{model_name}/none", "label": test["label"], "score": test_scores, "threshold": threshold})]
    
    # Attacked test sets: same rows, injections obfuscated, benign prompts unchanged, same threshold
    print("\n== ATTACKS (no defense) ==")
    print(f"clean: recall {(test_scores[test['label'] == 1] >= threshold).mean():.4f}")
    for name, (family, _) in ALL_ATTACKS.items():
        attacked = attack_injections(test, name)  
        scores = score(model, attacked["text"])                
        recall = (scores[attacked["label"] == 1] >= threshold).mean()
        print(f"{family}/{name}: recall {recall:.4f}")
        inj = test["label"] == 1                                                   
        changed = (attacked.loc[inj, "text"] != test.loc[inj, "text"]).mean()      
        print(f"   {changed:.1%} of injections changed")                           
        score_tables.append(pd.DataFrame({"row_id": test["row_id"], "value": name, "column": f"{model_name}/none", "label": test["label"], "score": scores, "threshold": threshold}))
        print("   distinct injection scores:", pd.Series(scores[attacked["label"] == 1]).round(6).nunique())
    
    # Strength sweep (sensitivity check): word/letter-level attacks at higher rates
    for name in SWEEP_ATTACKS:
        for rate in SWEEP_RATES:
            attacked = attack_injections(test, name, rate=rate)
            scores = score(model, attacked["text"])
            recall = (scores[attacked["label"] == 1] >= threshold).mean()
            print(f"sweep {name} @ {rate}: recall {recall:.4f}")
            score_tables.append(pd.DataFrame({"row_id": test["row_id"], "value": f"{name}_{rate}",
                                              "column": f"{model_name}/none", "label": test["label"],
                                              "score": scores, "threshold": threshold}))
    
    all_tables.extend(score_tables)
    
    # Normalization defense: same trained model; every prompt is normalized before scoring,
    # and the threshold is re-chosen on normalized validation prompts.
    norm_column = f"{model_name}/normalization"
    norm_benign = score(model, validation.loc[validation["label"] == 0, "text"].map(normalize))
    norm_threshold = choose_threshold(norm_benign, TARGET_FPR)
    print(f"\n== NORMALIZATION ==\nThreshold: {norm_threshold:.6f}")
    all_tables.extend(score_column(model, norm_column, norm_threshold, prep=normalize))
    
    # Adversarial augmentation defense: a fresh copy of the model, trained on the augmented set;
    # threshold re-chosen on clean validation prompts; test prompts scored without normalization.
    aug_model = clone(model)
    t0 = time.time()
    aug_model.fit(aug_train["text"], aug_train["label"])
    aug_benign = score(aug_model, validation.loc[validation["label"] == 0, "text"])
    aug_threshold = choose_threshold(aug_benign, TARGET_FPR)
    print(f"\n== AUGMENTATION ==\nTraining Time: {time.time() - t0:.2f} s\nThreshold: {aug_threshold:.6f}")
    all_tables.extend(score_column(aug_model, f"{model_name}/augmentation", aug_threshold))
    
    if model_name == "tfidf_lr":                                                  
        print("\n== BASE64 WRAPPERS (base64_wide, test variant) ==")              
        for t in TEMPLATES["base64_wide"]["test"]:                                
            s = model.predict_proba([t.format(payload="QUJD")])[0, 1]
            print(f"{'caught' if s >= threshold else 'missed'}  {s:.4f}  {t[:60]!r}")
    
    plot_figures(model_name, pd.concat(score_tables), threshold)                  

everything = pd.concat(all_tables)
everything.to_csv("scores.csv", index=False)
plot_sweep(everything[everything["column"].str.endswith("/none")])

# Error sample for section 4: 10 false positives + 10 false negatives from one cell, seed 42
cell_value, cell_column = "homoglyph", "tfidf_lr/none"
cell = pd.concat(all_tables)
cell = cell[(cell["value"] == cell_value) & (cell["column"] == cell_column)]
attacked = attack_injections(test, cell_value)                  # the text the model actually saw
cell = cell.merge(attacked[["row_id", "text"]], on="row_id")
cell = cell.merge(test[["row_id", "text"]].rename(columns={"text": "original"}), on="row_id")
pred = cell["score"] >= cell["threshold"]
fp = cell[pred & (cell["label"] == 0)]
fn = cell[~pred & (cell["label"] == 1)]
sample = pd.concat([fp.sample(min(10, len(fp)), random_state=SEED),
                    fn.sample(min(10, len(fn)), random_state=SEED)])
sample["error"] = np.where(sample["label"] == 1, "false negative", "false positive")
sample["category"] = ""
sample[["row_id", "value", "column", "error", "score", "category", "text", "original"]].to_csv(
    "errors_to_label.csv", index=False, encoding="utf-8-sig")
labels = pd.read_csv("error_categories.csv", encoding="utf-8-sig")
errors = sample.drop(columns="category").merge(labels, on="row_id", how="left")
missing = errors["category"].isna().sum()
if missing:
    print(f"WARNING: {missing} sampled row(s) have no category in error_categories.csv")
errors[["row_id", "value", "column", "category"]].to_csv("errors.csv", index=False, encoding="utf-8-sig")

print(f"\nError sample: value={cell_value}, column={cell_column}, seed={SEED}: "f"{min(10, len(fp))} of {len(fp)} false positives, {min(10, len(fn))} of {len(fn)} false negatives")
print(f"\nPython {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, "f"scikit-learn {sklearn.__version__}")
