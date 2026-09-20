"""Behavioural tests for the fused-target builder and its cache.

The real models and tokenizers are replaced with a tiny deterministic encoder
and hand-built batches, so the tests pin the pipeline (embedding, subspace,
alignment, gate, reconstruction, caching) without downloading anything.
"""

import os

import pytest
import torch
from torch import nn

from config import OurMethodConfig
from src.criterions.our_method import block_slices
from src.our_method_target import (
    FusedTargets,
    OurMethodTargetBuilder,
    sibling_cache_path,
)


class _FakeOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeEncoder(nn.Module):
    """Deterministic encoder: embedding -> linear -> optional dropout."""

    def __init__(self, vocab: int, dim: int, dropout: float = 0.0):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        hidden = self.dropout(self.proj(self.embed(input_ids)))
        return _FakeOutput(hidden)


def _make_batches(
    num_samples: int = 32,
    vocab: int = 50,
    seq_len: int = 6,
    batch_size: int = 8,
    sides=(1, 2),
) -> list[dict]:
    generator = torch.Generator().manual_seed(123)
    batches = []
    for start in range(0, num_samples, batch_size):
        size = min(batch_size, num_samples - start)
        batch = {}
        for side in sides:
            input_ids = torch.randint(
                1, vocab, (size, seq_len), generator=generator
            )
            attention_mask = torch.ones(size, seq_len, dtype=torch.long)
            special = torch.zeros(size, seq_len, dtype=torch.long)
            special[:, 0] = 1
            batch[f"input_ids{side}_stu"] = input_ids
            batch[f"attention_mask{side}_stu"] = attention_mask
            batch[f"special_tokens_mask{side}_stu"] = special
            batch[f"input_ids{side}_tea"] = input_ids.clone()
            batch[f"attention_mask{side}_tea"] = attention_mask.clone()
            batch[f"special_tokens_mask{side}_tea"] = special.clone()
        batches.append(batch)
    return batches


def _config(tmp_path, **overrides) -> OurMethodConfig:
    config = OurMethodConfig(
        target_view="text1",
        stability_view="augment",
        pooling_method="cls",
        subspace_rank=4,
        num_blocks=2,
        stability_margin=0.01,
        stability_tau=0.05,
        cache_path=str(tmp_path / "targets.pt"),
        seed=42,
        num_workers=0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _builder(config, model_teacher, model_base=None, mask_token_ids=None):
    model_base = model_base if model_base is not None else model_teacher
    return OurMethodTargetBuilder(
        model_teacher=model_teacher,
        model_base=model_base,
        device_s=torch.device("cpu"),
        device_t=torch.device("cpu"),
        config=config,
        mask_token_ids=mask_token_ids,
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_sibling_cache_path_keeps_the_stem_and_changes_the_suffix(tmp_path):
    path = sibling_cache_path(str(tmp_path / "targets.pt"), "teacher")

    assert path == str(tmp_path / "targets_teacher.pt")


def test_auto_view_mode_prefers_dropout_and_falls_back_to_augment(tmp_path):
    config = _config(tmp_path, stability_view="auto")
    with_dropout = _FakeEncoder(vocab=20, dim=8, dropout=0.2)
    without_dropout = _FakeEncoder(vocab=20, dim=8, dropout=0.0)

    assert _builder(config, with_dropout).base_view_mode == "dropout"
    assert _builder(config, without_dropout).base_view_mode == "augment"


def test_num_blocks_above_rank_is_rejected(tmp_path):
    config = _config(tmp_path, subspace_rank=3, num_blocks=4)

    with pytest.raises(ValueError, match="num_blocks must satisfy"):
        _builder(config, _FakeEncoder(vocab=20, dim=8))


def test_explicit_dropout_view_mode_is_honoured(tmp_path):
    config = _config(tmp_path, stability_view="dropout")
    model = _FakeEncoder(vocab=20, dim=8, dropout=0.0)

    assert _builder(config, model).base_view_mode == "dropout"


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def test_dimensions_finiteness_and_diagnostics_for_two_sides(tmp_path):
    config = _config(tmp_path, target_view="both")
    model = _FakeEncoder(vocab=50, dim=16)
    batches = _make_batches(num_samples=32)

    result = _builder(config, model).load_or_build(batches, len(batches) * 8)

    assert isinstance(result, FusedTargets)
    assert result.targets.shape == (32, 2, 16)
    assert torch.isfinite(result.targets).all()
    assert result.rank == 4
    assert result.slices == block_slices(4, 2)
    assert result.num_sides == 2
    assert set(result.diagnostics) == {
        "subspace",
        "alignment",
        "stability",
        "gate",
        "target",
    }
    assert result.diagnostics["gate"]["values"].shape == (2,)


def test_identical_teacher_and_base_closes_the_gate_and_recovers_the_base(tmp_path):
    config = _config(tmp_path)
    model = _FakeEncoder(vocab=50, dim=16)
    batches = _make_batches(num_samples=24)

    result = _builder(config, model).load_or_build(batches, len(batches) * 8)

    assert torch.allclose(
        result.diagnostics["gate"]["values"],
        torch.zeros(2, dtype=result.diagnostics["gate"]["values"].dtype),
    )
    assert result.diagnostics["alignment"]["error_before"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert result.diagnostics["target"]["target_displacement"] == pytest.approx(
        0.0, abs=1e-6
    )

    base = torch.load(sibling_cache_path(config.cache_path, "stu"))["tensor"]
    assert torch.allclose(result.targets[:, 0, :], base[:, 0, :], atol=1e-6)


def test_max_target_samples_only_subsamples_the_estimate(tmp_path):
    config = _config(tmp_path, max_target_samples=8)
    model = _FakeEncoder(vocab=50, dim=16)
    batches = _make_batches(num_samples=24)

    result = _builder(config, model).load_or_build(batches, len(batches) * 8)

    assert result.targets.shape == (24, 1, 16)
    assert torch.isfinite(result.targets).all()
    assert result.diagnostics["gate"]["values"].shape == (2,)


def test_target_displacement_is_non_negative_and_bounded(tmp_path):
    config = _config(tmp_path)
    teacher = _FakeEncoder(vocab=50, dim=16)
    base = _FakeEncoder(vocab=60, dim=16)
    batches = _make_batches(num_samples=24)

    result = _builder(config, teacher, base).load_or_build(batches, len(batches) * 8)

    displacement = result.diagnostics["target"]["target_displacement"]
    assert 0.0 <= displacement <= 2.0


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def test_second_build_loads_the_cache_and_is_identical(tmp_path):
    config = _config(tmp_path)
    model = _FakeEncoder(vocab=50, dim=16)
    batches = _make_batches(num_samples=24)

    first = _builder(config, model).load_or_build(batches, len(batches) * 8)
    second = _builder(config, model).load_or_build(batches, len(batches) * 8)

    assert torch.equal(first.targets, second.targets)
    for suffix in ("tea", "stu", "tea_views", "stu_views"):
        assert os.path.exists(sibling_cache_path(config.cache_path, suffix))


def test_cache_row_mismatch_raises(tmp_path):
    config = _config(tmp_path)
    model = _FakeEncoder(vocab=50, dim=16)
    batches = _make_batches(num_samples=24)
    _builder(config, model).load_or_build(batches, len(batches) * 8)

    with pytest.raises(ValueError, match="corpus has"):
        _builder(config, model).load_or_build(batches, 999)


def test_force_recompute_rebuilds_the_target_file(tmp_path):
    config = _config(tmp_path)
    model = _FakeEncoder(vocab=50, dim=16)
    batches = _make_batches(num_samples=16)
    builder = _builder(config, model)
    builder.load_or_build(batches, len(batches) * 8)
    first_mtime = os.path.getmtime(config.cache_path)

    config.force_recompute = True
    _builder(config, model).load_or_build(batches, len(batches) * 8)

    assert os.path.getmtime(config.cache_path) >= first_mtime
