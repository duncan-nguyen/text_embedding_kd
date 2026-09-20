"""Regression guards for the evaluation pipeline.

Covers the embedding helpers, the threshold sweep and the leakage check, plus a
small end-to-end pair-classification pass with a fake model so the
``Dataset`` free path (``_embed_texts``) stays wired up after cleanups.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.evaluation.evaluation_automodel import (
    ClasssifyDataset,
    PairDataset,
    STSDataset,
    _cosine_similarity,
    _pooled_embedding,
    _read_eval_csv,
    _validate_classification_pair,
    eval_pair_task,
    eval_sts_task,
    get_metric_pair_classification,
)


class FakeModel(nn.Module):
    """Short inputs embed to one direction, long ones to an orthogonal direction."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask):
        batch, length = input_ids.shape
        lengths = attention_mask.sum(dim=1)
        is_short = (lengths <= 3).float()
        pooled = torch.stack([is_short, 1.0 - is_short], dim=1).repeat(1, 2)
        hidden = pooled.unsqueeze(1).repeat(1, length, 1)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeTokenizer:
    def __call__(
        self,
        texts,
        truncation=True,
        padding=True,
        max_length=None,
        return_tensors="pt",
    ):
        if isinstance(texts, str):
            texts = [texts]
        sequences = []
        for text in texts:
            ids = [101] + [1000 + (ord(char) % 50) for char in text]
            if max_length is not None:
                ids = ids[:max_length]
            sequences.append(ids)
        width = max(len(ids) for ids in sequences)
        input_ids = [ids + [0] * (width - len(ids)) for ids in sequences]
        attention_mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


# --------------------------------------------------------------------------
# Embedding helpers
# --------------------------------------------------------------------------

def test_pooled_embedding_reads_stella_style_dict():
    pooled = torch.randn(2, 3)

    assert _pooled_embedding({"pooled": pooled}) is pooled


def test_pooled_embedding_reads_transformers_object_cls():
    hidden = torch.randn(2, 5, 3)
    output = SimpleNamespace(last_hidden_state=hidden)

    assert torch.equal(_pooled_embedding(output), hidden[:, 0, :])


def test_pooled_embedding_reads_transformers_dict_cls():
    hidden = torch.randn(2, 5, 3)

    assert torch.equal(_pooled_embedding({"last_hidden_state": hidden}), hidden[:, 0, :])


def test_cosine_similarity_matches_reference_values():
    left = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    right = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, -1.0]])

    similarity = _cosine_similarity(left, right)

    assert similarity[0] == pytest.approx(1.0)
    assert similarity[1] == pytest.approx(0.0)
    assert similarity[2] == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# Threshold sweep
# --------------------------------------------------------------------------

def test_pair_metric_finds_a_perfect_threshold():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    metrics = get_metric_pair_classification(scores, labels)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert 0.2 < metrics["best_threshold"] <= 0.8


def test_pair_metric_honours_an_explicit_threshold():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    metrics = get_metric_pair_classification(scores, labels, threshold=0.5)

    assert metrics["best_threshold"] == 0.5
    assert metrics["accuracy"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# CSV loading and leakage validation
# --------------------------------------------------------------------------

def test_read_eval_csv_caches_parses(tmp_path):
    path = tmp_path / "cached.csv"
    pd.DataFrame({"text": ["a", "b"]}).to_csv(path, index=False)

    first = _read_eval_csv(str(path))
    second = _read_eval_csv(str(path))

    assert first is second


def test_validation_leakage_is_rejected(tmp_path):
    value_dir = tmp_path / "val_set"
    value_dir.mkdir()
    train = value_dir / "emotion_train.csv"
    validation = value_dir / "emotion_validation.csv"
    pd.DataFrame({"text": ["Same Text", "unique"]}).to_csv(train, index=False)
    pd.DataFrame({"text": ["same   text", "other"]}).to_csv(validation, index=False)

    with pytest.raises(ValueError, match="leakage"):
        _validate_classification_pair(str(train), str(validation))


def test_disjoint_validation_split_passes(tmp_path):
    value_dir = tmp_path / "val_set"
    value_dir.mkdir()
    train = value_dir / "emotion_train2.csv"
    validation = value_dir / "emotion_validation2.csv"
    pd.DataFrame({"text": ["alpha", "beta"]}).to_csv(train, index=False)
    pd.DataFrame({"text": ["gamma", "delta"]}).to_csv(validation, index=False)

    assert _validate_classification_pair(str(train), str(validation)) is None


# --------------------------------------------------------------------------
# End-to-end with a fake model
# --------------------------------------------------------------------------

def test_eval_pair_task_returns_metrics_and_thresholds(tmp_path):
    path = tmp_path / "mrpc_validation.csv"
    pd.DataFrame(
        {
            "sentence1": ["a", "bb", "cccccc", "dddddd"],
            "sentence2": ["a", "bb", "a", "b"],
            "label": [1, 1, 0, 0],
        }
    ).to_csv(path, index=False)

    results, thresholds = eval_pair_task(FakeModel(), [str(path)], FakeTokenizer())

    metrics = results[str(path)]
    assert {"accuracy", "f1", "precision", "recall", "average_precision", "best_threshold"} <= set(metrics)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert thresholds[0] == metrics["best_threshold"]


def test_eval_sts_task_returns_spearman(tmp_path):
    path = tmp_path / "stsb_validation.csv"
    pd.DataFrame(
        {
            "sentence1": ["a", "bb", "cccccc", "dddddd"],
            "sentence2": ["a", "a", "cccccc", "a"],
            "score": [0.0, 1.0, 2.0, 3.0],
        }
    ).to_csv(path, index=False)

    results = eval_sts_task(FakeModel(), [str(path)], FakeTokenizer())

    assert isinstance(results[str(path)], float)


def test_dataset_classes_expose_csv_columns(tmp_path):
    path = tmp_path / "rows.csv"
    pd.DataFrame(
        {"sentence1": ["a"], "sentence2": ["b"], "score": [1.0], "text": ["x"], "label": [0]}
    ).to_csv(path, index=False)

    assert len(STSDataset(str(path))) == 1
    assert len(PairDataset(str(path))) == 1
    assert len(ClasssifyDataset(str(path))) == 1
