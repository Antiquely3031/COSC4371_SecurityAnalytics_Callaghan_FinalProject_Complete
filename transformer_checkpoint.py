"""
COSC 4371 Project #15 variant: transformer detectors (run on a GPU, such as a Colab T4).

Reads the committed split.csv, so the rows are exactly those used by final_checkpoint.py,
then scores DistilBERT (none, normalization, augmentation) and ProtectAI (none,
normalization) on clean, every attack and every sweep strength. Writes
scores_transformers.csv, which final_checkpoint.py merges into scores.csv.

Run: python transformer_checkpoint.py
"""

import sys
import time

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from sklearn.base import clone

from attacks import ATTACKS, ALL_ATTACKS, SWEEP_ATTACKS, SWEEP_RATES, attack_injections, attack_text
from defenses import normalize
from transformer_models import DEVICE, DistilBertDetector, ProtectAIDetector

SEED = 42
TARGET_FPR = 0.05

# The dataset supplies the text; split.csv supplies which rows to keep, their split and their label.
ds = load_dataset("xTRam1/safe-guard-prompt-injection")
texts = pd.concat([pd.DataFrame({"row_id": [f"Safe-Guard:{s}:{i}" for i in range(len(ds[s]))],
                                 "text": list(ds[s]["text"])}) for s in ds], ignore_index=True)
split = pd.read_csv("split.csv")
data = split.merge(texts, on="row_id", how="left", validate="one_to_one")
assert data["text"].notna().all(), "some row_ids in split.csv were not found in the dataset"
train, validation, test = (data[data["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test"))
print(f"Rows from split.csv: {len(train)} train, {len(validation)} validation, {len(test)} test; device: {DEVICE}")

# Copied unchanged from final_checkpoint.py, so both scripts score, threshold and augment identically.
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


MODELS = {
    "distilbert": DistilBertDetector(seed=SEED),
    "protectai": ProtectAIDetector(),
}

aug_train = augment(train)
print(f"Augmented training set: {len(train)} original + {len(aug_train) - len(train)} obfuscated copies")
print(f"Augmented leakage: train rows also in test = {aug_train['text'].isin(test['text']).sum()}, "
      f"also in validation = {aug_train['text'].isin(validation['text']).sum()}")

benign = validation.loc[validation["label"] == 0, "text"]
all_tables = []
for model_name, model in MODELS.items():
    t0 = time.time()
    model.fit(train["text"], train["label"])
    threshold = choose_threshold(score(model, benign), TARGET_FPR)
    print(f"\n== {model_name} ==\nTraining (or loading) time: {time.time() - t0:.1f} s\nThreshold: {threshold:.6f}")

    print("\n== NO DEFENSE ==")
    all_tables.extend(score_column(model, f"{model_name}/none", threshold))

    # Normalization is a test-time defense, so it uses the threshold of the model it is applied to.
    print("\n== NORMALIZATION ==")
    all_tables.extend(score_column(model, f"{model_name}/normalization", threshold, prep=normalize))

    # Augmentation retrains a fresh copy; ProtectAI is pretrained and is skipped.
    if getattr(model, "trainable", True):
        aug_model = clone(model)
        t0 = time.time()
        aug_model.fit(aug_train["text"], aug_train["label"])
        aug_threshold = choose_threshold(score(aug_model, benign), TARGET_FPR)
        print(f"\n== AUGMENTATION ==\nTraining time: {time.time() - t0:.1f} s\nThreshold: {aug_threshold:.6f}")
        all_tables.extend(score_column(aug_model, f"{model_name}/augmentation", aug_threshold))

pd.concat(all_tables).to_csv("scores_transformers.csv", index=False)
print(f"\nPython {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, "
      f"pandas {pd.__version__}, numpy {np.__version__}, device {DEVICE}")
