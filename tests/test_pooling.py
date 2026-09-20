"""Regression guards for the pooling helpers."""

import torch

from src.pooling import last_token_pool, mean_pooling


def test_last_token_pool_picks_the_last_unmasked_token_for_right_padding():
    hidden = torch.tensor([[[1.0], [2.0], [3.0], [0.0]], [[4.0], [5.0], [0.0], [0.0]]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    pooled = last_token_pool(hidden, mask)

    assert torch.equal(pooled, torch.tensor([[3.0], [5.0]]))


def test_last_token_pool_picks_the_final_position_for_left_padding():
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])

    pooled = last_token_pool(hidden, mask)

    assert torch.equal(pooled, hidden[:, -1])


def test_mean_pooling_ignores_padded_positions():
    hidden = torch.tensor([[[1.0], [3.0], [100.0]], [[2.0], [100.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    pooled = mean_pooling(hidden, mask)

    assert torch.allclose(pooled, torch.tensor([[2.0], [2.0]]))
