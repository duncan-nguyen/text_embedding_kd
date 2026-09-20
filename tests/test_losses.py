"""Numerical regression guards for the shared loss helpers."""

import pytest
import torch

from src.loss import (
    cosine_embedding_loss,
    get_score_diff,
    info_nce,
    pair_inbatch_similarity_loss,
    pair_inbatch_triplet_loss,
)


def test_info_nce_returns_logits_and_low_loss_for_aligned_pairs():
    embeddings = torch.eye(4)

    loss, logits = info_nce(embeddings, embeddings, temperature=0.1)

    assert logits.shape == (4, 4)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_info_nce_penalises_misaligned_pairs_more():
    embeddings = torch.eye(4)
    shifted = torch.roll(embeddings, 1, dims=0)

    aligned = info_nce(embeddings, embeddings, temperature=0.1)[0]
    misaligned = info_nce(embeddings, shifted, temperature=0.1)[0]

    assert misaligned.item() > aligned.item()


def test_cosine_embedding_loss_is_zero_for_identical_inputs():
    embeddings = torch.randn(5, 8)

    assert cosine_embedding_loss(embeddings, embeddings).item() == pytest.approx(0.0, abs=1e-6)


def test_cosine_embedding_loss_is_two_for_opposite_inputs():
    left = torch.randn(5, 8)
    right = -left

    assert cosine_embedding_loss(left, right).item() == pytest.approx(2.0, abs=1e-5)


def test_pair_inbatch_similarity_loss_is_zero_for_identical_inputs():
    embeddings = torch.randn(5, 8)

    assert pair_inbatch_similarity_loss(embeddings, embeddings).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_pair_inbatch_triplet_loss_is_zero_for_identical_inputs_without_margin():
    embeddings = torch.randn(5, 8)

    loss = pair_inbatch_triplet_loss(embeddings, embeddings, triplet_margin=0.0)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_get_score_diff_enumerates_upper_triangle_pair_differences():
    embeddings = torch.eye(4)

    # Six unordered pairs -> their 6x6 difference matrix, upper triangle kept.
    assert get_score_diff(embeddings).shape == (15,)
