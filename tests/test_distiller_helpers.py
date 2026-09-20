"""Regression guards for the distiller's module-level helpers and static methods."""

import pytest
import torch
from torch import nn

from distiller import (
    IOD_BENCHMARKS,
    KnowledgeDistiller,
    assert_module_parameters_finite,
    grads_are_finite,
    is_finite,
    nonfinite_details,
    should_save_epoch,
)


def test_is_finite_accepts_only_finite_tensors():
    assert is_finite(torch.tensor([1.0, 2.0])) is True
    assert is_finite(torch.tensor([1.0, float("nan")])) is False
    assert is_finite(torch.tensor([float("inf")])) is False
    assert is_finite("not a tensor") is False


def test_nonfinite_details_counts_nan_and_inf():
    details = nonfinite_details("x", torch.tensor([1.0, float("nan"), float("inf")]))

    assert "nan_count=1" in details
    assert "inf_count=1" in details


def test_assert_module_parameters_finite_passes_for_finite_model():
    assert_module_parameters_finite(nn.Linear(3, 2), "linear")


def test_assert_module_parameters_finite_rejects_nan_parameter():
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight[0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="became NaN/Inf"):
        assert_module_parameters_finite(layer, "linear")


def test_grads_are_finite_treats_missing_grads_as_finite():
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    assert grads_are_finite(optimizer) is True

    parameter.grad = torch.tensor([float("nan")])
    assert grads_are_finite(optimizer) is False


@pytest.mark.parametrize(
    "epoch_index,save_every,expected",
    [(0, 3, False), (1, 3, False), (2, 3, True), (3, 1, True)],
)
def test_should_save_epoch(epoch_index, save_every, expected):
    assert should_save_epoch(epoch_index, save_every) is expected


def test_iod_benchmarks_are_the_three_in_domain_datasets():
    assert IOD_BENCHMARKS == frozenset({"emotion", "wic", "stsb"})


def test_benchmark_name_strips_the_split_suffix():
    assert KnowledgeDistiller._benchmark_name("data/val_set/banking77_validation.csv", "validation") == "banking77"
    assert KnowledgeDistiller._benchmark_name("data/test_set/stsb_test.csv", "test") == "stsb"


def test_metric_details_formats_known_metrics_as_percentages():
    details = KnowledgeDistiller._metric_details({"accuracy": 0.5, "f1": 0.25})

    assert "Acc=50.00" in details
    assert "F1=25.00" in details


def test_benchmark_group_averages_split_iod_and_ood():
    scores = {"emotion": 0.8, "wic": 0.6, "stsb": 0.7, "mrpc": 0.5}

    averages = KnowledgeDistiller._benchmark_group_averages(scores)

    assert averages["avg_iod"]["score"] == pytest.approx((0.8 + 0.6 + 0.7) / 3)
    assert averages["avg_ood"]["score"] == pytest.approx(0.5)
    assert averages["avg_all"]["score"] == pytest.approx((0.8 + 0.6 + 0.7 + 0.5) / 4)
    assert averages["avg_iod"]["members"] == ["emotion", "stsb", "wic"]


def test_flatten_metrics_flattens_nested_numeric_dicts():
    flat = KnowledgeDistiller._flatten_metrics(
        "train", {"loss": 1, "nested": {"accuracy": 2.0}, "label": "skip"}
    )

    assert flat == {"train/loss": 1.0, "train/nested/accuracy": 2.0}
