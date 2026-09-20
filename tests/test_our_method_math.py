"""Equation-level tests for the student-anchored subspace fusion statistics.

Everything here runs in float64 on synthetic tensors so the assertions can pin
the mathematics (orthonormality, the Procrustes closed form, CKA invariants,
the gate and the reconstruction identity) rather than just tensor shapes.
"""

import math

import pytest
import torch

from src.criterions.our_method import (
    OurMethodDistillation,
    block_slices,
    block_stability,
    build_gate_matrix,
    center_features,
    covariance,
    linear_cka,
    orthogonal_procrustes,
    procrustes_error,
    reconstruct_target,
    stability_gates,
    top_spectral_subspace,
)

# --------------------------------------------------------------------------
# center_features / covariance
# --------------------------------------------------------------------------

def test_center_features_subtracts_the_column_mean():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)

    centered, mean = center_features(features)

    assert torch.allclose(mean, torch.tensor([[3.0, 4.0]], dtype=torch.float64))
    assert torch.allclose(centered.mean(dim=0), torch.zeros(2, dtype=torch.float64))


def test_center_features_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        center_features(torch.zeros(3, dtype=torch.float64))


def test_covariance_matches_hand_computed_value():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)

    result = covariance(features)

    expected = torch.tensor([[8.0 / 3.0, 8.0 / 3.0], [8.0 / 3.0, 8.0 / 3.0]], dtype=torch.float64)
    assert torch.allclose(result, expected)
    assert torch.allclose(result, result.transpose(0, 1))


# --------------------------------------------------------------------------
# top_spectral_subspace
# --------------------------------------------------------------------------

def test_top_spectral_subspace_is_orthonormal_and_solves_the_eigen_equation():
    torch.manual_seed(0)
    features = torch.randn(50, 10, dtype=torch.float64)

    result = top_spectral_subspace(features, rank=4)
    c = covariance(features)

    identity = torch.eye(4, dtype=torch.float64)
    assert torch.allclose(result.U.transpose(0, 1) @ result.U, identity, atol=1e-10)
    # C U = U diag(lambda)
    assert torch.allclose(c @ result.U, result.U * result.eigenvalues, atol=1e-10)


def test_top_spectral_subspace_matches_eigvalsh_and_is_descending():
    torch.manual_seed(1)
    features = torch.randn(80, 12, dtype=torch.float64)
    c = covariance(features)
    reference = torch.linalg.eigvalsh(0.5 * (c + c.transpose(0, 1)))

    result = top_spectral_subspace(features, rank=5)

    assert torch.allclose(result.full_eigenvalues, reference)
    assert torch.allclose(result.eigenvalues, reference[-5:].flip(0), atol=1e-10)
    assert torch.all(result.eigenvalues[:-1] >= result.eigenvalues[1:])


def test_explained_variance_is_the_top_rank_share():
    torch.manual_seed(2)
    features = torch.randn(60, 9, dtype=torch.float64)
    result = top_spectral_subspace(features, rank=3)

    expected = result.eigenvalues.sum() / result.full_eigenvalues.sum()
    assert result.explained_variance.item() == pytest.approx(expected.item(), abs=1e-12)
    assert 0.0 < result.explained_variance.item() <= 1.0


def test_effective_rank_is_one_for_rank_one_data_and_near_full_for_isotropic():
    outer = torch.randn(40, dtype=torch.float64).unsqueeze(1) * torch.randn(6, dtype=torch.float64).unsqueeze(0)
    rank_one = top_spectral_subspace(outer, rank=6)
    assert rank_one.effective_rank.item() == pytest.approx(1.0, abs=1e-9)

    torch.manual_seed(3)
    isotropic = torch.randn(4000, 8, dtype=torch.float64)
    full = top_spectral_subspace(isotropic, rank=8)
    assert full.effective_rank.item() > 7.5


def test_condition_number_is_top_over_smallest_selected_eigenvalue():
    torch.manual_seed(4)
    features = torch.randn(70, 7, dtype=torch.float64)
    result = top_spectral_subspace(features, rank=4)

    expected = result.eigenvalues[0] / result.eigenvalues[-1]
    assert result.condition_number.item() == pytest.approx(expected.item(), rel=1e-10)


def test_block_energy_sums_selected_eigenvalues_per_block():
    torch.manual_seed(5)
    features = torch.randn(50, 6, dtype=torch.float64)
    slices = block_slices(4, 2)

    result = top_spectral_subspace(features, rank=4, block_slices=slices)

    assert torch.allclose(result.block_energy[0], result.eigenvalues[0:2].sum())
    assert torch.allclose(result.block_energy[1], result.eigenvalues[2:4].sum())


def test_top_spectral_subspace_validates_rank():
    features = torch.randn(10, 4, dtype=torch.float64)

    with pytest.raises(ValueError, match="rank must satisfy"):
        top_spectral_subspace(features, rank=0)
    with pytest.raises(ValueError, match="rank must satisfy"):
        top_spectral_subspace(features, rank=5)


# --------------------------------------------------------------------------
# orthogonal Procrustes
# --------------------------------------------------------------------------

def test_procrustes_recovers_a_known_rotation():
    torch.manual_seed(6)
    source = torch.randn(40, 5, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64))
    target = source @ q

    rotation = orthogonal_procrustes(source, target)

    assert torch.allclose(rotation, q, atol=1e-10)
    assert torch.allclose(rotation.transpose(0, 1) @ rotation, torch.eye(5, dtype=torch.float64), atol=1e-10)


def test_procrustes_reduces_the_alignment_error():
    torch.manual_seed(7)
    source = torch.randn(30, 4, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
    target = source @ q + 1e-6 * torch.randn(30, 4, dtype=torch.float64)

    rotation = orthogonal_procrustes(source, target)
    before = procrustes_error(source, target)
    after = procrustes_error(source, target, rotation)

    assert after < before
    assert after.item() < 1e-4


def test_procrustes_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="share a shape"):
        orthogonal_procrustes(torch.randn(4, 3), torch.randn(4, 2))


# --------------------------------------------------------------------------
# linear CKA
# --------------------------------------------------------------------------

def test_cka_of_a_matrix_with_itself_is_one():
    torch.manual_seed(8)
    features = torch.randn(20, 6, dtype=torch.float64)

    assert linear_cka(features, features).item() == pytest.approx(1.0, abs=1e-9)


def test_cka_is_invariant_to_orthogonal_transform_and_isotropic_scale():
    torch.manual_seed(9)
    left = torch.randn(25, 5, dtype=torch.float64)
    right = torch.randn(25, 5, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64))

    base = linear_cka(left, right).item()
    assert linear_cka(left @ q, right).item() == pytest.approx(base, abs=1e-9)
    assert linear_cka(left, 3.0 * right).item() == pytest.approx(base, abs=1e-9)


def test_cka_is_equivariant_to_a_shared_row_permutation():
    torch.manual_seed(10)
    left = torch.randn(15, 4, dtype=torch.float64)
    right = torch.randn(15, 4, dtype=torch.float64)
    permutation = torch.randperm(15)

    base = linear_cka(left, right).item()
    shuffled = linear_cka(left[permutation], right[permutation]).item()

    assert shuffled == pytest.approx(base, abs=1e-9)


def test_cka_one_dimensional_case_equals_squared_cosine():
    left = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    right = torch.tensor([[2.0], [1.0], [0.0]], dtype=torch.float64)

    centered_left = left - left.mean()
    centered_right = right - right.mean()
    expected = (torch.nn.functional.cosine_similarity(
        centered_left.flatten(), centered_right.flatten(), dim=0
    ) ** 2).item()

    assert linear_cka(left, right).item() == pytest.approx(expected, abs=1e-6)


def test_cka_stays_in_the_unit_interval_and_rejects_row_mismatch():
    torch.manual_seed(11)
    left = torch.randn(20, 3, dtype=torch.float64)
    right = torch.randn(20, 5, dtype=torch.float64)

    value = linear_cka(left, right).item()
    assert 0.0 <= value <= 1.0

    with pytest.raises(ValueError, match="same number of rows"):
        linear_cka(torch.randn(3, 2), torch.randn(4, 2))


# --------------------------------------------------------------------------
# block partition / stability
# --------------------------------------------------------------------------

def test_block_slices_partition_the_rank_contiguously_and_balancely():
    slices = block_slices(10, 3)

    assert slices == [(0, 4), (4, 7), (7, 10)]
    sizes = [end - start for start, end in slices]
    assert sum(sizes) == 10
    assert max(sizes) - min(sizes) <= 1


def test_block_slices_edge_cases_and_validation():
    assert block_slices(5, 1) == [(0, 5)]
    assert block_slices(4, 4) == [(0, 1), (1, 2), (2, 3), (3, 4)]

    with pytest.raises(ValueError, match="num_blocks must satisfy"):
        block_slices(4, 5)
    with pytest.raises(ValueError, match="num_blocks must satisfy"):
        block_slices(4, 0)


def test_block_stability_is_one_for_identical_views():
    torch.manual_seed(12)
    views = torch.randn(30, 6, dtype=torch.float64)

    scores = block_stability(views, views, block_slices(6, 3))

    assert scores.shape == (3,)
    assert torch.allclose(scores, torch.ones(3, dtype=torch.float64), atol=1e-9)


def test_block_stability_lower_for_independent_views_than_for_a_shared_signal():
    torch.manual_seed(13)
    signal = torch.randn(500, 4, dtype=torch.float64)
    correlated = torch.stack([block_stability(signal, signal + 0.05 * torch.randn_like(signal), [(0, 4)])[0]])

    independent = block_stability(
        torch.randn(500, 4, dtype=torch.float64),
        torch.randn(500, 4, dtype=torch.float64),
        [(0, 4)],
    )

    assert independent.item() < correlated.item()


def test_block_stability_rejects_mismatched_views():
    with pytest.raises(ValueError, match="share a shape"):
        block_stability(torch.randn(5, 3), torch.randn(5, 2), [(0, 3)])


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def test_gate_is_zero_at_or_below_the_margin_and_saturated_above():
    delta = torch.tensor([-0.5, 0.0, 0.05], dtype=torch.float64)

    gates = stability_gates(delta, margin=0.05, tau=0.02)

    assert torch.allclose(gates, torch.zeros(3, dtype=torch.float64))


def test_gate_follows_the_sigmoid_above_the_margin():
    margin, tau = 0.1, 0.05
    delta = torch.tensor([margin + tau * math.log(3.0)], dtype=torch.float64)

    gate = stability_gates(delta, margin=margin, tau=tau)

    assert gate.item() == pytest.approx(0.75, abs=1e-9)


def test_gate_is_monotonic_and_bounded():
    delta = torch.linspace(0.05, 1.0, 50, dtype=torch.float64)

    gates = stability_gates(delta, margin=0.05, tau=0.1)

    assert torch.all(gates[1:] >= gates[:-1])
    assert torch.all(gates >= 0.0) and torch.all(gates < 1.0)


def test_gate_rejects_non_positive_tau():
    with pytest.raises(ValueError, match="tau must be positive"):
        stability_gates(torch.tensor([0.5], dtype=torch.float64), margin=0.0, tau=0.0)


def test_gate_matrix_is_block_diagonal_with_the_gate_values():
    gates = torch.tensor([0.0, 0.75], dtype=torch.float64)
    slices = block_slices(4, 2)

    matrix = build_gate_matrix(gates, slices, rank=4)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.75, 0.0],
            [0.0, 0.0, 0.0, 0.75],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(matrix, expected)

    with pytest.raises(ValueError, match="gates for"):
        build_gate_matrix(torch.tensor([0.0], dtype=torch.float64), slices, rank=4)


# --------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------

def _orthonormal_subspace(dim, rank, seed):
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, rank, dtype=torch.float64))
    return q


def test_zero_gate_recovers_the_base_student_exactly():
    torch.manual_seed(14)
    h0 = torch.randn(12, 8, dtype=torch.float64)
    u0 = _orthonormal_subspace(8, 3, seed=15)
    delta = torch.randn(12, 3, dtype=torch.float64)
    gate = build_gate_matrix(torch.zeros(1, dtype=torch.float64), [(0, 3)], rank=3)

    target = reconstruct_target(h0, delta, gate, u0)

    assert torch.allclose(target, h0, atol=1e-12)


def test_identity_gate_matches_the_closed_form_and_preserves_the_residual():
    torch.manual_seed(16)
    h0 = torch.randn(10, 6, dtype=torch.float64)
    u0 = _orthonormal_subspace(6, 2, seed=17)
    z0 = h0 @ u0
    zt = torch.randn(10, 2, dtype=torch.float64)
    delta = zt - z0
    gate = build_gate_matrix(torch.ones(1, dtype=torch.float64), [(0, 2)], rank=2)

    target = reconstruct_target(h0, delta, gate, u0)
    closed_form = zt @ u0.transpose(0, 1) + h0 @ (torch.eye(6, dtype=torch.float64) - u0 @ u0.transpose(0, 1))

    assert torch.allclose(target, closed_form, atol=1e-10)
    # The component orthogonal to span(U0) is untouched.
    residual = target - h0
    orthogonal = residual @ (torch.eye(6, dtype=torch.float64) - u0 @ u0.transpose(0, 1))
    assert torch.allclose(orthogonal, torch.zeros_like(orthogonal), atol=1e-10)


def test_reconstruction_validates_dimensions():
    h0 = torch.randn(4, 5, dtype=torch.float64)
    u0 = torch.randn(5, 3, dtype=torch.float64)
    delta = torch.randn(4, 3, dtype=torch.float64)
    gate = build_gate_matrix(torch.ones(1, dtype=torch.float64), [(0, 3)], rank=3)

    with pytest.raises(ValueError, match="base embedding width"):
        reconstruct_target(torch.randn(4, 7), delta, gate, u0)
    with pytest.raises(ValueError, match="delta_latent width"):
        reconstruct_target(h0, torch.randn(4, 2), gate, u0)


# --------------------------------------------------------------------------
# criterion
# --------------------------------------------------------------------------

def test_fusion_loss_is_zero_for_aligned_and_one_for_orthogonal():
    criterion = OurMethodDistillation(w_task=0.0, w_fusion=1.0)
    aligned = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    orthogonal = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    loss, cosine = criterion.fusion_loss(aligned, aligned)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert cosine.item() == pytest.approx(1.0, abs=1e-6)

    loss, cosine = criterion.fusion_loss(aligned, orthogonal)
    assert loss.item() == pytest.approx(1.0, abs=1e-6)
    assert cosine.item() == pytest.approx(0.0, abs=1e-6)


def test_forward_combines_base_and_fusion_with_the_configured_weights():
    criterion = OurMethodDistillation(w_task=0.5, w_fusion=2.0)
    student = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0]])
    task_loss = torch.tensor(0.4)

    loss, metrics = criterion(student, target, task_loss)

    assert loss.item() == pytest.approx(0.5 * 0.4 + 2.0 * 1.0, abs=1e-6)
    assert set(metrics) == {"loss_total", "loss_base", "loss_fusion", "cos_target"}
    assert metrics["loss_base"] == pytest.approx(0.4, abs=1e-6)
    assert metrics["loss_fusion"] == pytest.approx(1.0, abs=1e-6)


def test_forward_averages_extra_pairs():
    criterion = OurMethodDistillation(w_task=0.0, w_fusion=1.0)
    student = torch.tensor([[1.0, 0.0]])
    target_aligned = torch.tensor([[1.0, 0.0]])
    target_orthogonal = torch.tensor([[0.0, 1.0]])
    task_loss = torch.tensor(0.0)

    loss, metrics = criterion(
        student, target_aligned, task_loss,
        extra_pairs=((student, target_orthogonal),),
    )

    assert metrics["loss_fusion"] == pytest.approx(0.5, abs=1e-6)
    assert loss.item() == pytest.approx(0.5, abs=1e-6)


def test_gradient_flows_to_student_but_target_is_stop_gradient():
    criterion = OurMethodDistillation(w_task=0.5, w_fusion=1.0)
    student = torch.tensor([[1.0, 0.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0]], requires_grad=True)

    loss, _ = criterion(student, target, torch.tensor(0.3))
    loss.backward()

    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert target.grad is None


def test_normalize_target_keeps_cosine_for_scaled_targets():
    criterion = OurMethodDistillation(w_task=0.0, w_fusion=1.0, normalize_target=True)

    loss, cosine = criterion.fusion_loss(torch.tensor([[1.0, 0.0]]), torch.tensor([[5.0, 0.0]]))

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert cosine.item() == pytest.approx(1.0, abs=1e-6)
