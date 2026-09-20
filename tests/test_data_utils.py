"""Regression guards for the dataset and collate helpers."""

import pytest
import torch

from src.data_utils.dataset import DualTokenizerCollate, TextPairRaw
from src.data_utils.dataset_cache import (
    DualTokenizerCollateWithFusionTarget,
    DualTokenizerCollateWithTeacher,
    TextPairWithFusionTarget,
    TextPairWithTeacher,
)


class FakeTokenizer:
    """Minimal tokenizer with the surface the collates rely on."""

    def __call__(
        self,
        texts,
        max_length=None,
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_special_tokens_mask=False,
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
        special_tokens_mask = [
            [1] + [0] * (len(ids) - 1) + [0] * (width - len(ids)) for ids in sequences
        ]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "special_tokens_mask": torch.tensor(special_tokens_mask),
        }


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def test_text_pair_raw_single_cls_builds_labelled_samples():
    frame = _frame({"text": ["a", "b"], "label": [0, 1]})

    dataset = TextPairRaw(frame, "single_cls")

    assert len(dataset) == 2
    assert dataset[0] == ("a", None, 0)


def test_text_pair_raw_pair_cls_keeps_both_sides():
    frame = _frame({"premise": ["a", "b"], "hypothesis": ["c", "d"], "label": [0, 1]})

    dataset = TextPairRaw(frame, "pair_cls")

    assert dataset[1] == ("b", "d", 1)


def test_text_pair_raw_pair_reg_uses_score_column():
    frame = _frame({"sentence1": ["a"], "sentence2": ["b"], "score": [3.5]})

    dataset = TextPairRaw(frame, "pair_reg")

    assert dataset[0] == ("a", "b", 3.5)


def test_text_pair_raw_rejects_missing_columns():
    with pytest.raises(ValueError, match="single_cls"):
        TextPairRaw(_frame({"text": ["a"]}), "single_cls")


def test_text_pair_raw_rejects_unknown_task():
    with pytest.raises(ValueError, match="Unsupported task"):
        TextPairRaw(_frame({"text": ["a"]}), "regression")


def test_dual_tokenizer_collate_emits_student_and_teacher_pair_encodings(tokenizer):
    frame = _frame({"premise": ["a", "bb"], "hypothesis": ["c", "dd"], "label": [0, 1]})
    collate = DualTokenizerCollate(tokenizer, tokenizer, "pair_cls", max_len=16)

    batch = collate([TextPairRaw(frame, "pair_cls")[0], TextPairRaw(frame, "pair_cls")[1]])

    assert {"input_ids1_stu", "input_ids1_tea", "input_ids2_stu", "input_ids2_tea"} <= set(batch)
    assert batch["labels"].dtype == torch.long


def test_dual_tokenizer_collate_single_cls_has_no_second_text(tokenizer):
    frame = _frame({"text": ["a", "bb"], "label": [0, 1]})
    dataset = TextPairRaw(frame, "single_cls")
    collate = DualTokenizerCollate(tokenizer, tokenizer, "single_cls", max_len=16)

    batch = collate([dataset[0], dataset[1]])

    assert "input_ids2_stu" not in batch
    assert batch["labels"].dtype == torch.long


def test_dual_tokenizer_collate_pair_reg_labels_are_float(tokenizer):
    frame = _frame({"sentence1": ["a"], "sentence2": ["b"], "score": [1.0]})
    dataset = TextPairRaw(frame, "pair_reg")
    collate = DualTokenizerCollate(tokenizer, tokenizer, "pair_reg", max_len=16)

    batch = collate([dataset[0]])

    assert batch["labels"].dtype == torch.float32


def test_dual_tokenizer_collate_omits_labels_when_unlabelled(tokenizer):
    collate = DualTokenizerCollate(tokenizer, tokenizer, "pair_cls", max_len=16)

    batch = collate([("a", "b", None), ("c", "d", None)])

    assert "labels" not in batch


def test_dual_tokenizer_collate_rejects_mixed_text_arity(tokenizer):
    collate = DualTokenizerCollate(tokenizer, tokenizer, "pair_cls", max_len=16)

    with pytest.raises(ValueError, match="mix single-text and pair-text"):
        collate([("a", "b", 0), ("c", None, 1)])


def test_dual_tokenizer_collate_rejects_mixed_labelling(tokenizer):
    collate = DualTokenizerCollate(tokenizer, tokenizer, "pair_cls", max_len=16)

    with pytest.raises(ValueError, match="mix labeled and unlabeled"):
        collate([("a", "b", 0), ("c", "d", None)])


def test_cached_teacher_collate_stacks_teacher_vectors(tokenizer):
    frame = _frame({"premise": ["a", "bb"], "hypothesis": ["c", "dd"]})
    teacher = torch.randn(2, 6)
    dataset = TextPairWithTeacher(frame, "pair_cls", teacher)
    collate = DualTokenizerCollateWithTeacher(tokenizer, "pair_cls", max_len=16)

    batch = collate([dataset[0], dataset[1]])

    assert batch["teacher_cls"].shape == (2, 6)
    assert torch.allclose(batch["teacher_cls"], teacher)
    assert "input_ids1_stu" in batch and "input_ids2_stu" in batch


def test_text_pair_with_teacher_pairs_each_sample_with_its_vector():
    frame = _frame({"premise": ["a", "bb"], "hypothesis": ["c", "dd"]})
    teacher = torch.randn(2, 6)
    dataset = TextPairWithTeacher(frame, "pair_cls", teacher)

    item, vector = dataset[1]

    assert item == ("bb", "dd")
    assert torch.allclose(vector, teacher[1])


def test_text_pair_with_fusion_target_pairs_each_sample_with_its_target():
    frame = _frame({"premise": ["a", "bb"], "hypothesis": ["c", "dd"], "label": [0, 1]})
    targets = torch.randn(2, 2, 5)
    dataset = TextPairWithFusionTarget(frame, "pair_cls", targets)

    item, target = dataset[1]

    assert item == ("bb", "dd", 1)
    assert torch.allclose(target, targets[1])


def test_fusion_collate_emits_student_encodings_and_both_targets(tokenizer):
    frame = _frame({"premise": ["a", "bb"], "hypothesis": ["c", "dd"], "label": [0, 1]})
    targets = torch.randn(2, 2, 5)
    dataset = TextPairWithFusionTarget(frame, "pair_cls", targets)
    collate = DualTokenizerCollateWithFusionTarget(tokenizer, "pair_cls", max_len=16, num_sides=2)

    batch = collate([dataset[0], dataset[1]])

    assert "input_ids1_stu" in batch and "input_ids2_stu" in batch
    assert not any(key.endswith("_tea") for key in batch)
    assert batch["target1"].shape == (2, 5)
    assert batch["target2"].shape == (2, 5)
    assert torch.allclose(batch["target2"], targets[:, 1, :])
    assert batch["labels"].dtype == torch.long


def test_fusion_collate_omits_target2_for_a_single_side(tokenizer):
    frame = _frame({"premise": ["a", "bb"], "hypothesis": ["c", "dd"]})
    targets = torch.randn(2, 1, 5)
    dataset = TextPairWithFusionTarget(frame, "pair_cls", targets)
    collate = DualTokenizerCollateWithFusionTarget(tokenizer, "pair_cls", max_len=16, num_sides=1)

    batch = collate([dataset[0], dataset[1]])

    assert batch["target1"].shape == (2, 5)
    assert "target2" not in batch


def _frame(columns):
    import pandas as pd

    return pd.DataFrame(columns)
