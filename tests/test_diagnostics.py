"""Tests for the OurMethod diagnostics flattening and plotting."""

import os

import pytest
import torch

from src.diagnostics import flatten_ourmethod_diagnostics, save_ourmethod_diagnostics


def _diagnostics() -> dict:
    return {
        "subspace": {
            "teacher_eigenvalues": torch.linspace(1.0, 0.1, 4),
            "student_eigenvalues": torch.linspace(1.0, 0.1, 4),
            "teacher_explained_variance": 0.9,
            "student_explained_variance": 0.8,
            "teacher_effective_rank": 3.5,
            "student_effective_rank": 3.2,
            "teacher_condition_number": 10.0,
            "student_condition_number": 8.0,
            "teacher_block_energy": torch.tensor([1.5, 0.6]),
            "student_block_energy": torch.tensor([1.4, 0.5]),
        },
        "alignment": {
            "error_before": 1.0,
            "error_after": 0.2,
            "improvement": 0.8,
            "latent_cka": 0.7,
            "residual_norm": 0.5,
        },
        "stability": {
            "teacher": torch.tensor([0.9, 0.8]),
            "student": torch.tensor([0.7, 0.85]),
            "delta": torch.tensor([0.2, -0.05]),
            "teacher_mean": 0.85,
            "student_mean": 0.775,
            "delta_mean": 0.075,
            "delta_std": 0.125,
            "teacher_win_ratio": 0.5,
            "student_win_ratio": 0.5,
            "margin_pass_ratio": 0.5,
        },
        "gate": {
            "values": torch.tensor([0.8, 0.0]),
            "mean": 0.4,
            "std": 0.4,
            "min": 0.0,
            "max": 0.8,
            "zero_ratio": 0.5,
            "low_ratio": 0.5,
            "mid_ratio": 0.0,
            "high_ratio": 0.5,
            "inject_ratio": 0.6,
            "score_delta_corr": 0.9,
            "score_energy_corr": 0.1,
        },
        "target": {
            "cos_target_base": 0.95,
            "cos_target_teacher": 0.5,
            "target_displacement": 0.05,
            "correction_norm": 0.3,
        },
    }


def test_flatten_ourmethod_diagnostics_is_all_scalars_or_lists():
    flat = flatten_ourmethod_diagnostics(_diagnostics())

    assert flat["alignment/error_after"] == 0.2
    assert flat["gate/mean"] == 0.4
    assert flat["target/target_displacement"] == 0.05
    assert flat["stability/delta_by_block"] == pytest.approx([0.2, -0.05], abs=1e-6)
    for key, value in flat.items():
        if isinstance(value, list):
            assert all(isinstance(item, float) for item in value), key
        else:
            assert isinstance(value, float), key


def test_save_ourmethod_diagnostics_writes_four_plots(tmp_path):
    paths = save_ourmethod_diagnostics(_diagnostics(), str(tmp_path))

    assert len(paths) == 4
    expected = {
        "gate_histogram.png",
        "delta_stability_by_block.png",
        "gate_vs_delta_stability.png",
        "spectral_energy_teacher_vs_student.png",
    }
    assert {os.path.basename(path) for path in paths} == expected
    for path in paths:
        assert os.path.getsize(path) > 0
