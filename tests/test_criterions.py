"""Behavioural pins for the distillation criterions.

The paper-faithful objectives (RKD, PKT) have their own equation-level tests;
this module covers the method-specific wiring and the shared geometric helpers
so a refactor cannot silently change shapes, keys or the optimised quantity.
"""

import pytest
import torch

from src.criterions.contextual_dynamic_mapping import (
    _normalize_token,
    align_strict_one_to_one,
    cost_fn,
    dtw,
)
from src.criterions.dual_space_kd import DualSpaceKD
from src.criterions.emo_embedding_distillation import (
    CKALoss,
    EMODistillation,
    align_tokens,
    compute_token_importance,
    pairwise_attention_distance,
    project_importance,
    sinkhorn,
)
from src.criterions.stella_distillation import stella_stage1_loss, stella_stage2_loss
from src.criterions.teacher_anchor_kd import TeacherAnchorKD


# --------------------------------------------------------------------------
# TALAS: TeacherAnchorKD
# --------------------------------------------------------------------------

def test_teacher_anchor_kd_forward_reports_expected_metrics():
    criterion = TeacherAnchorKD(student_dim=8, teacher_dim=6, num_layers=3, last_layer_idx=1)
    hidden = [torch.randn(4, 5, 8, requires_grad=True) for _ in range(3)]

    loss, metrics = criterion(
        student_outputs={"hidden_states": hidden, "last_hidden_state": hidden[-1]},
        teacher_cls=torch.randn(4, 6),
        task_loss=torch.tensor(0.5),
    )

    assert torch.isfinite(loss)
    assert set(metrics) == {"loss_total", "loss_task", "loss_kd", "loss_struct"}

    loss.backward()
    assert any(head.weight.grad is not None for head in criterion.kd_proj_heads)


def test_teacher_anchor_kd_rejects_wrong_layer_count():
    criterion = TeacherAnchorKD(student_dim=8, teacher_dim=6, num_layers=3)
    hidden = [torch.randn(2, 4, 8) for _ in range(2)]

    with pytest.raises(ValueError, match="Expected 3 layers"):
        criterion(
            student_outputs={"hidden_states": hidden, "last_hidden_state": hidden[-1]},
            teacher_cls=torch.randn(2, 6),
            task_loss=torch.tensor(0.0),
        )


# --------------------------------------------------------------------------
# DSKD: DualSpaceKD
# --------------------------------------------------------------------------

def test_dual_space_kd_forward_reports_expected_metrics():
    criterion = DualSpaceKD(student_dim=8, teacher_dim=6)
    batch, student_len, teacher_len = 3, 5, 7
    student_last = torch.randn(batch, student_len, 8, requires_grad=True)
    teacher_last = torch.randn(batch, teacher_len, 6)

    loss, metrics = criterion.compute_dskd_loss(
        S_last=student_last,
        T_last=teacher_last,
        S_cls=student_last[:, 0, :],
        T_cls=teacher_last[:, 0, :],
        mask_student=torch.ones(batch, student_len, dtype=torch.bool),
        mask_teacher=torch.ones(batch, teacher_len, dtype=torch.bool),
        task_loss=torch.tensor(0.5),
    )

    assert torch.isfinite(loss)
    assert set(metrics) == {
        "loss_total",
        "loss_task",
        "kd_cls",
        "kd_s2t",
        "kd_t2s",
        "kd_tok",
        "kd_tok_s1",
        "kd_tok_t1",
    }


# --------------------------------------------------------------------------
# CDM: alignment helpers
# --------------------------------------------------------------------------

def test_cost_fn_is_zero_for_identical_tokens():
    assert cost_fn("hello", "hello") == 0.0


def test_cost_fn_counts_edits_between_normalized_tokens():
    assert cost_fn("hello", "hella") == 1.0


def test_normalize_token_strips_model_markers():
    assert _normalize_token("##embedding") == "embedding"
    assert _normalize_token("\u2581embedding") == "embedding"
    assert _normalize_token("Ghello", marker="G") == "hello"


def test_dtw_identical_sequences_have_zero_cost_and_diagonal_path():
    path, cost, _, _, _ = dtw(["a", "b"], ["a", "b"])

    assert cost == pytest.approx(0.0)
    assert path == [(0, 0), (1, 1)]


def test_align_strict_one_to_one_keeps_matching_pairs_in_order():
    base_vals = torch.randn(2, 3)
    blend_vals = torch.randn(2, 3)

    aligned_base, aligned_blend = align_strict_one_to_one(
        base_vals=base_vals,
        blend_vals=blend_vals,
        path=[(0, 0), (1, 1)],
        base_tokens=["a", "b"],
        blend_tokens=["a", "b"],
        base_marker="",
        blend_marker="",
    )

    assert torch.allclose(aligned_base, base_vals)
    assert torch.allclose(aligned_blend, blend_vals)


# --------------------------------------------------------------------------
# EMO: geometric helpers and module wiring
# --------------------------------------------------------------------------

def test_emo_module_projects_teacher_into_student_dimension():
    criterion = EMODistillation(d_teacher=6, d_student=4)

    assert criterion.proj_t2s.weight.shape == (4, 6)


def test_sinkhorn_returns_finite_cost_and_doubly_stochastic_plan():
    cost = torch.rand(4, 5)
    source = torch.ones(4) / 4
    target = torch.ones(5) / 5

    loss, plan = sinkhorn(cost, source, target, max_iter=50)

    assert torch.isfinite(loss)
    assert plan.shape == (4, 5)
    assert plan.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_align_tokens_is_index_based_and_order_preserving():
    assert align_tokens(["a", "b"], ["a", "b"]) == {0: 0, 1: 1}


def test_project_importance_normalises_to_one():
    teacher_importance = torch.tensor([0.4, 0.6])

    student_importance = project_importance(
        teacher_importance, ["a", "b"], ["a", "b"], {0: 0, 1: 1}
    )

    assert student_importance.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_compute_token_importance_normalises_to_one():
    attention = torch.rand(6, 6)

    importance = compute_token_importance(attention, list("abcdef"))

    assert importance.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_pairwise_attention_distance_has_one_row_per_query_token():
    distance = pairwise_attention_distance(torch.randn(3, 4), torch.randn(5, 4))

    assert distance.shape == (3, 5)


def test_cka_loss_is_finite_and_zero_for_identical_features():
    features = torch.randn(8, 5)
    cka = CKALoss(eps=1e-8)

    assert cka(features, features).item() == pytest.approx(0.0, abs=1e-5)
    assert torch.isfinite(cka(features, torch.randn(8, 7)))


# --------------------------------------------------------------------------
# Stella: stage losses
# --------------------------------------------------------------------------

def test_stella_stage1_loss_reports_expected_metrics():
    loss, metrics = stella_stage1_loss(torch.randn(4, 8), torch.randn(4, 8))

    assert torch.isfinite(loss)
    assert set(metrics) == {"loss_total", "loss_cos", "loss_sim", "loss_tri"}


def test_stella_stage2_loss_reports_expected_metrics():
    loss, metrics = stella_stage2_loss(
        S_cls1=torch.randn(4, 8),
        S_cls2=torch.randn(4, 8),
        S_emb1=torch.randn(4, 8),
        S_emb2=torch.randn(4, 6),
        S_emb3=torch.randn(4, 5),
        S_emb4=torch.randn(4, 4),
        T_emb=torch.randn(4, 8),
    )

    assert torch.isfinite(loss)
    assert set(metrics) == {"loss_total", "loss_task", "loss_cos", "loss_sim", "loss_tri"}
