"""
Transformer detectors for COSC 4371 Project #15 variant, wrapped to behave like
scikit-learn models (fit / predict_proba), so the threshold, scoring and clone()
code used for the TF-IDF models works on them unchanged.

    Name        What it is                                  Trained here?
    distilbert  distilbert-base-uncased, fine-tuned         yes (also on the augmented set)
    protectai   protectai/deberta-v3-base-prompt-           no (published detector, scored only)
                injection-v2
"""

import math

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _injection_probability(model, tokenizer, texts, positive_index, max_length, batch_size):
    """Return an (n, 2) array of [benign, injection] probabilities, the same shape as predict_proba."""
    model.eval()
    texts = list(texts)
    probs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], truncation=True, max_length=max_length,
                              padding=True, return_tensors="pt").to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                logits = model(**batch).logits
            probs.append(torch.softmax(logits.float(), dim=-1)[:, positive_index].cpu().numpy())
    p = np.concatenate(probs)
    return np.column_stack([1 - p, p])


class DistilBertDetector(BaseEstimator, ClassifierMixin):
    """Fine-tunes DistilBERT to classify prompts as injection (1) or benign (0)."""

    def __init__(self, checkpoint="distilbert-base-uncased", epochs=2, learning_rate=5e-5,
                 batch_size=32, max_length=256, seed=42):
        self.checkpoint = checkpoint
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed

    def fit(self, texts, labels):
        torch.manual_seed(self.seed)
        texts, labels = list(texts), np.asarray(labels)
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=2).to(DEVICE)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate)
        steps = self.epochs * math.ceil(len(texts) / self.batch_size)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
        order_rng = np.random.default_rng(self.seed)

        self.model_.train()
        for epoch in range(self.epochs):
            order = order_rng.permutation(len(texts))                 # shuffle each epoch, reproducibly
            for start in range(0, len(texts), self.batch_size):
                idx = order[start:start + self.batch_size]
                batch = self.tokenizer_([texts[i] for i in idx], truncation=True, max_length=self.max_length,
                                        padding=True, return_tensors="pt").to(DEVICE)
                target = torch.tensor(labels[idx]).to(DEVICE)
                loss = self.model_(**batch, labels=target).loss       # how wrong the model is on this batch
                optimizer.zero_grad()
                loss.backward()                                       # work out how to adjust each weight
                optimizer.step()                                      # adjust them
                scheduler.step()                                      # lower the learning rate a little
            print(f"   epoch {epoch + 1}/{self.epochs} done, last batch loss {loss.item():.4f}")

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, texts):
        return _injection_probability(self.model_, self.tokenizer_, texts, 1, self.max_length, 2 * self.batch_size)


class ProtectAIDetector(BaseEstimator, ClassifierMixin):
    """The published ProtectAI prompt-injection detector. It is already trained, so fit() only
    loads it: the training data is ignored, and the model cannot be augmented."""

    trainable = False

    def __init__(self, checkpoint="protectai/deberta-v3-base-prompt-injection-v2", batch_size=64, max_length=512):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.max_length = max_length

    def fit(self, texts=None, labels=None):
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(DEVICE)
        label_index = {name.upper(): i for i, name in self.model_.config.id2label.items()}
        self.positive_index_ = label_index["INJECTION"]       # which of the two outputs means "injection"
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, texts):
        return _injection_probability(self.model_, self.tokenizer_, texts, self.positive_index_,
                                      self.max_length, self.batch_size)    