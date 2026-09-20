"""Regression guards for teacher embedding caching."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.cache_teacher import cache_teacher_embeddings, load_cached_embeddings


class FakeTeacher(nn.Module):
    """Deterministic stand-in for a HF model's last_hidden_state."""

    def __init__(self, dim=4):
        super().__init__()
        self.dim = dim
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, return_dict=True):
        batch, length = input_ids.shape
        hidden = torch.arange(
            batch * length * self.dim, dtype=torch.float32
        ).reshape(batch, length, self.dim)
        return SimpleNamespace(last_hidden_state=hidden)


class IndexedDataset:
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids1_tea": self.input_ids[index],
            "attention_mask1_tea": self.attention_masks[index],
        }


def pad_collate(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids1_tea"] for item in batch], batch_first=True
    )
    attention = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask1_tea"] for item in batch], batch_first=True
    )
    return {"input_ids1_tea": input_ids, "attention_mask1_tea": attention}


def make_loader():
    dataset = IndexedDataset(
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        [torch.tensor([1, 1, 1]), torch.tensor([1, 1])],
    )
    return DataLoader(dataset, batch_size=2, collate_fn=pad_collate)


def test_cache_teacher_embeddings_returns_one_row_per_sample():
    embeddings = cache_teacher_embeddings(
        FakeTeacher(dim=4), make_loader(), torch.device("cpu"), pooling_method="last_token"
    )

    assert embeddings.shape == (2, 4)


def test_cache_teacher_embeddings_round_trips_through_disk(tmp_path):
    cache_path = tmp_path / "nested" / "teacher.pt"

    written = cache_teacher_embeddings(
        FakeTeacher(dim=4),
        make_loader(),
        torch.device("cpu"),
        pooling_method="last_token",
        cache_path=str(cache_path),
    )
    loaded = load_cached_embeddings(str(cache_path))

    assert cache_path.is_file()
    assert torch.allclose(written, loaded)


def test_cache_teacher_embeddings_reuses_an_existing_cache(tmp_path):
    cache_path = tmp_path / "teacher.pt"
    torch.save(torch.ones(3, 7), cache_path)

    embeddings = cache_teacher_embeddings(
        FakeTeacher(dim=4),
        make_loader(),
        torch.device("cpu"),
        pooling_method="last_token",
        cache_path=str(cache_path),
    )

    assert embeddings.shape == (3, 7)


@pytest.mark.parametrize("pooling", ["last_token", "mean", "cls"])
def test_supported_pooling_methods_produce_a_row_per_sample(pooling):
    embeddings = cache_teacher_embeddings(
        FakeTeacher(dim=4), make_loader(), torch.device("cpu"), pooling_method=pooling
    )

    assert embeddings.shape == (2, 4)


def test_unknown_pooling_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown pooling method"):
        cache_teacher_embeddings(
            FakeTeacher(dim=4), make_loader(), torch.device("cpu"), pooling_method="max"
        )


def test_missing_cache_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cached_embeddings(str(tmp_path / "missing.pt"))
