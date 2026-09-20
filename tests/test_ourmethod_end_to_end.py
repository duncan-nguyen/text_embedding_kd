"""End-to-end run of the OurMethod path with mocked models and tokenizer.

This exercises setup_models -> setup_data (target build + cache) ->
setup_training -> train_step -> train_epoch -> save_checkpoint -> train, so a
wiring regression fails here instead of in a GPU run.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

import distiller as distiller_module
from config import OurMethodConfig
from distiller import KnowledgeDistiller


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeEncoder(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)
        self.config = SimpleNamespace(hidden_size=dim)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        return _Output(self.proj(self.embed(input_ids)))


class _FakeTokenizer:
    mask_token_id = 103

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
            ids = [101] + [1000 + (ord(char) % 50) for char in str(text)]
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
def patched(monkeypatch):
    def fake_from_pretrained(name, **kwargs):
        return _FakeEncoder(vocab=1100, dim=16)

    class _AutoModel:
        from_pretrained = staticmethod(fake_from_pretrained)

    class _AutoTokenizer:
        from_pretrained = staticmethod(lambda name, **kwargs: _FakeTokenizer())

    monkeypatch.setattr(distiller_module, "AutoModel", _AutoModel)
    monkeypatch.setattr(distiller_module, "AutoTokenizer", _AutoTokenizer)
    monkeypatch.setattr(
        KnowledgeDistiller, "evaluate", lambda self, split="validation": {}
    )
    return monkeypatch


def _write_corpus(tmp_path, rows: int = 16):
    frame = pd.DataFrame({"text": [f"sample text number {i}" for i in range(rows)]})
    path = tmp_path / "corpus.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_ourmethod_end_to_end_trains_and_caches(tmp_path, patched):
    config = OurMethodConfig(
        train_data_path=_write_corpus(tmp_path),
        student_model_name="fake",
        base_student_model_name="fake",
        teacher_model_name="fake",
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        max_length=16,
        subspace_rank=4,
        num_blocks=2,
        stability_view="auto",
        target_view="both",
        target_batch_size=4,
        cache_path=str(tmp_path / "cache" / "ourmethod" / "targets.pt"),
        save_dir=str(tmp_path / "save"),
        weights_dir=None,
        num_workers=0,
        use_wandb=False,
        diagnostics=True,
        seed=42,
    )

    distiller = KnowledgeDistiller(config)
    try:
        distiller.train()
    finally:
        distiller.close()

    assert (tmp_path / "cache" / "ourmethod" / "targets.pt").exists()
    assert (tmp_path / "save" / "metrics.jsonl").exists()
    assert (tmp_path / "save" / "diagnostics" / "gate_histogram.png").exists()

    # Rerun uses the cache rather than rebuilding from scratch.
    distiller2 = KnowledgeDistiller(config)
    try:
        cached = distiller2.fused_targets
    finally:
        distiller2.close()
    assert torch.isfinite(cached).all()
