# Prompt-Injection Detection Under Obfuscation Attacks

COSC 4371 Security Analytics, Fall 2026 — Project #15 variant (Safe-Guard only)
Patrick Callaghan

This project trains two prompt-injection detectors on the Safe-Guard dataset and measures how
much their recall drops under seven obfuscation attacks in three families (encoding, structure,
context), and how much of that loss two defenses (normalization and adversarial augmentation)
recover.

## Quick start

Tested with Python 3.13 in a virtual environment. No GPU is needed.

```bash
python -m pip install -r requirements.txt
python final_checkpoint.py
python check_final_checkpoint.py --split split.csv --scores scores.csv --errors errors.csv --metric recall --reference clean --direction lower > checker_output.txt
```

`final_checkpoint.py` runs in about a minute. The checker takes several minutes because it
bootstraps 150 cells; with `> checker_output.txt` its output goes to that file instead of the screen.
`check_final_checkpoint.py` is the course checker from Canvas.

The data is downloaded from Hugging Face (`xTRam1/safe-guard-prompt-injection`) with the
`datasets` library when the script runs. It is not committed to this repository.

## Files

| File | Contents |
|---|---|
| `final_checkpoint.py` | The whole pipeline: load, audit, deduplicate, split, train, attack, defend, score, plot, sample errors |
| `attacks.py` | The seven attacks, their templates, and the strength-sweep settings |
| `defenses.py` | The normalization defense (`normalize`) |
| `error_categories.csv` | Hand-assigned error categories for the 20 sampled rows (the only manual input) |
| `split.csv` | Every kept row with its split (train, validation, test) and label |
| `scores.csv` | Test-set scores for every attack, model and defense, with each threshold |
| `errors.csv` | The error sample with categories, built from `error_categories.csv` |
| `checker_output.txt` | Output of the course checker for the files above |
| `roc_*.png`, `pr_*.png` | ROC and precision-recall curves per model, one panel per attack family (no defense) |
| `sweep.png` | Recall against attack strength for each model (no defense) |
| `requirements.txt` | Exact package versions |

`errors_to_label.csv` is regenerated on every run (sampled rows with their text, for labeling) and
is not committed.

## Method

**Data and split.** Safe-Guard's published train/test partition (10,296 rows). Exact duplicates on
`text` keep one copy, preferring the test copy (159 removed). Near duplicates are rows whose
character 3–5-gram TF-IDF vectors have cosine similarity of at least 0.90 after Unicode NFKC,
lowercasing and whitespace collapsing; training rows in a group that also contains a test row are
removed (233). Validation is one of seven folds of the training set, grouped by near-duplicate group
and stratified by label (seed 42). Final split: 6,733 train, 1,122 validation, 2,049 test. A leakage
check finds 0 training rows that exactly duplicate a test or validation row.

**Models.**

| Name | Features | Classifier |
|---|---|---|
| `tfidf_lr` | Word TF-IDF, unigrams and bigrams, `min_df=2`, sublinear tf | Logistic regression, C = 1 |
| `char_svm` | Character TF-IDF (`char_wb`, 3–5-grams), `min_df=2`, sublinear tf | Linear SVM, C = 1 |

**Threshold.** For each model and defense: the lowest threshold (on a 0.000001 grid) at which at
most 5% of benign validation prompts score at or above it. Chosen on validation only and reused
unchanged on every test set. Positive class: injection (label 1).

**Attacks** (applied to test injections only; benign prompts are never changed):

| Family | Attack | What it does |
|---|---|---|
| encoding | `zero_width` | inserts invisible zero-width characters inside words |
| encoding | `homoglyph` | swaps Latin letters for look-alike Cyrillic letters |
| encoding | `base64` | encodes the whole prompt in Base64 inside a "decode this" request |
| structure | `token_split` | breaks words with a space, hyphen or period |
| structure | `typo` | adds one typo (swap, drop, double or replace a letter) to words |
| context | `payload_split` | cuts the prompt into variables and asks for them to be joined |
| context | `role_framing` | wraps the prompt in a harmless-sounding role-play frame |

The main, pre-specified strength is 30% of eligible words (letters, for homoglyph). Two sensitivity
checks are reported separately: `base64_wide` (ten wrapper templates instead of two) and a strength
sweep at 10, 20, 60 and 100% for the four character-level attacks.

**Defenses.**

| Defense | Where it acts | What it does |
|---|---|---|
| normalization | input | removes invisible format characters, applies NFKC, maps Cyrillic and Greek look-alikes to Latin, decodes embedded Base64 |
| augmentation | training data | retrains each model with one obfuscated copy of every training injection and of an equal number of benign prompts, using separate training-time templates and random strengths of 10–100% |

**Primary metric and verdict.** Recall at the validation threshold. A change of at least 0.05 below
the clean recall in the same column counts as "yes"; anything smaller, including increases, counts
as "no".

**Error sample.** From value `homoglyph`, column `tfidf_lr/none`: 10 of 63 false positives and 10 of
15 false negatives, drawn with pandas `DataFrame.sample` (`random_state=42`).

## Reproducibility

- Random seed: 42, set in `final_checkpoint.py` and `attacks.py`; every attack is seeded per row, so
  reruns produce identical attacked text, scores and samples.
- Package versions: `requirements.txt`.
- Thresholds from the submitted run:

| Model | none | normalization | augmentation |
|---|---|---|---|
| `tfidf_lr` | 0.135581 | 0.135581 | 0.171031 |
| `char_svm` | −0.724044 | −0.724044 | −0.708861 |

SVM scores are signed distances from the decision boundary, so its thresholds are on a different
scale from logistic regression's probabilities.

## AI assistance

Code and design choices were developed with assistance from Claude (Anthropic). The full prompts and
responses are submitted with the report.
