"""Student-anchored subspace fusion.

This module contains the label-free statistics that turn a frozen teacher and a
frozen base student into a fused target representation, plus the criterion that
trains the student toward that target. It is deliberately free of any model or
tokenizer plumbing so every quantity can be unit tested on synthetic tensors.

Notation follows ``docs/subspace_fusion_distillation_proposal_v4.md``:

* ``H_T`` / ``H_0`` -- teacher / base-student embeddings of the corpus,
* ``U_T`` / ``U_0`` -- top-``r`` spectral subspaces of their covariances,
* ``Z_T`` / ``Z_0`` -- latent coordinates in those subspaces,
* ``R*``           -- orthogonal Procrustes alignment of teacher to student,
* ``G``            -- block-diagonal gate built from cross-view stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "OurMethodDistillation",
    "SubspaceResult",
    "block_slices",
    "block_stability",
    "build_gate_matrix",
    "center_features",
    "covariance",
    "linear_cka",
    "orthogonal_procrustes",
    "procrustes_error",
    "reconstruct_target",
    "stability_gates",
    "top_spectral_subspace",
]

# Below this a covariance eigenvalue is treated as numerical zero.
_EIG_EPS = 1e-12


@dataclass
class SubspaceResult:
    """Top-``r`` spectral subspace of a feature matrix and its diagnostics."""

    U: torch.Tensor
    eigenvalues: torch.Tensor
    total_variance: torch.Tensor
    explained_variance: torch.Tensor
    effective_rank: torch.Tensor
    condition_number: torch.Tensor
    full_eigenvalues: torch.Tensor
    block_energy: torch.Tensor = field(default=None)


def center_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Center columns of ``features`` and return ``(centered, mean)``."""
    if features.dim() != 2:
        raise ValueError(f"expected a 2D [N, d] tensor, got shape {tuple(features.shape)}")
    mean = features.mean(dim=0, keepdim=True)
    return features - mean, mean


def covariance(features: torch.Tensor) -> torch.Tensor:
    """Unbiased-free covariance ``C = (1/N) Hc^T Hc`` (proposal Eq. 2)."""
    centered, _ = center_features(features)
    n = centered.shape[0]
    if n == 0:
        raise ValueError("cannot estimate a covariance from zero rows")
    return centered.transpose(0, 1) @ centered / n


def _effective_rank(eigenvalues: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    positive = eigenvalues[eigenvalues > eps]
    if positive.numel() == 0:
        return eigenvalues.new_zeros(())
    probabilities = positive / positive.sum()
    entropy = -(probabilities * torch.log(probabilities)).sum()
    return torch.exp(entropy)


def top_spectral_subspace(
    features: torch.Tensor,
    rank: int,
    block_slices: list[tuple[int, int]] | None = None,
) -> SubspaceResult:
    """Extract the top-``rank`` eigenvectors of the feature covariance.

    Eigenvalues are returned in descending order, matching the spectral blocks
    the gate is defined on.
    """
    if features.dim() != 2:
        raise ValueError(f"expected a 2D [N, d] tensor, got shape {tuple(features.shape)}")
    n, dim = features.shape
    if not 1 <= rank <= dim:
        raise ValueError(f"rank must satisfy 1 <= rank <= {dim}, got {rank}")

    feature_dtype = features.dtype
    work = features.to(torch.float64)
    centered, _ = center_features(work)
    # eigh expects a symmetric matrix and returns ascending eigenvalues.
    gram = centered.transpose(0, 1) @ centered / max(n, 1)
    gram = 0.5 * (gram + gram.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)

    full_eigenvalues = eigenvalues.clamp_min(0.0)
    top_eigenvalues = full_eigenvalues[-rank:].flip(0)
    top_eigenvectors = eigenvectors[:, -rank:].flip(1)

    total_variance = full_eigenvalues.sum()
    if float(total_variance) <= _EIG_EPS:
        explained_variance = torch.zeros((), dtype=torch.float64)
    else:
        explained_variance = top_eigenvalues.sum() / total_variance

    smallest_selected = top_eigenvalues[-1].clamp_min(_EIG_EPS)
    condition_number = top_eigenvalues[0].clamp_min(_EIG_EPS) / smallest_selected

    block_energy = None
    if block_slices is not None:
        block_energy = top_eigenvalues.new_zeros(len(block_slices))
        for index, (start, end) in enumerate(block_slices):
            block_energy[index] = top_eigenvalues[start:end].sum()

    return SubspaceResult(
        U=top_eigenvectors.to(feature_dtype),
        eigenvalues=top_eigenvalues.to(feature_dtype),
        total_variance=total_variance.to(feature_dtype),
        explained_variance=explained_variance.to(feature_dtype),
        effective_rank=_effective_rank(full_eigenvalues).to(feature_dtype),
        condition_number=condition_number.to(feature_dtype),
        full_eigenvalues=full_eigenvalues.to(feature_dtype),
        block_energy=block_energy.to(feature_dtype) if block_energy is not None else None,
    )


def orthogonal_procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return ``R* = argmin_{R^T R = I} ||source @ R - target||_F``.

    Closed form: if ``source^T target = U S V^T`` then ``R* = U V^T``.
    """
    if source.shape != target.shape:
        raise ValueError(
            f"source and target must share a shape, got "
            f"{tuple(source.shape)} and {tuple(target.shape)}"
        )
    if source.dim() != 2:
        raise ValueError("orthogonal_procrustes expects 2D [N, r] tensors")

    work_source = source.to(torch.float64)
    work_target = target.to(torch.float64)
    cross = work_source.transpose(0, 1) @ work_target
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return (u @ vh).to(source.dtype)


def procrustes_error(
    source: torch.Tensor,
    target: torch.Tensor,
    rotation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Relative Frobenius error ``||source R - target||_F / ||target||_F``."""
    denominator = target.norm().clamp_min(_EIG_EPS)
    aligned = source if rotation is None else source @ rotation
    return (aligned - target).norm() / denominator


def linear_cka(
    left: torch.Tensor,
    right: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Linear CKA between two ``[N, k]`` matrices (proposal Eq. 4)."""
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"CKA needs the same number of rows, got {left.shape[0]} and {right.shape[0]}"
        )
    work_left = left.to(torch.float64)
    work_right = right.to(torch.float64)
    work_left = work_left - work_left.mean(dim=0, keepdim=True)
    work_right = work_right - work_right.mean(dim=0, keepdim=True)

    numerator = (work_left.transpose(0, 1) @ work_right).norm() ** 2
    denominator = (work_left.transpose(0, 1) @ work_left).norm() * (
        work_right.transpose(0, 1) @ work_right
    ).norm()
    return (numerator / (denominator + eps)).to(left.dtype)


def block_slices(dim: int, num_blocks: int) -> list[tuple[int, int]]:
    """Partition ``[0, dim)`` into ``num_blocks`` contiguous balanced blocks."""
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    if not 1 <= num_blocks <= dim:
        raise ValueError(
            f"num_blocks must satisfy 1 <= num_blocks <= {dim}, got {num_blocks}"
        )
    base, remainder = divmod(dim, num_blocks)
    slices = []
    start = 0
    for index in range(num_blocks):
        size = base + (1 if index < remainder else 0)
        slices.append((start, start + size))
        start += size
    return slices


def block_stability(
    view1: torch.Tensor,
    view2: torch.Tensor,
    slices: list[tuple[int, int]],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-block linear CKA between two views of the same sentences."""
    if view1.shape != view2.shape:
        raise ValueError(
            f"views must share a shape, got {tuple(view1.shape)} and {tuple(view2.shape)}"
        )
    scores = [
        linear_cka(view1[:, start:end], view2[:, start:end], eps=eps)
        for start, end in slices
    ]
    return torch.stack(scores)


def stability_gates(
    delta_stability: torch.Tensor,
    margin: float,
    tau: float,
) -> torch.Tensor:
    """Gate teacher corrections from the relative stability margin (Eq. 5)."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    excess = (delta_stability - margin) / tau
    gates = torch.sigmoid(excess)
    return torch.where(delta_stability <= margin, torch.zeros_like(gates), gates)


def build_gate_matrix(
    gates: torch.Tensor,
    slices: list[tuple[int, int]],
    rank: int,
) -> torch.Tensor:
    """Assemble ``G = BlockDiag(g_1 I, ..., g_B I)`` of shape ``[rank, rank]``."""
    if len(gates) != len(slices):
        raise ValueError(
            f"got {len(gates)} gates for {len(slices)} blocks"
        )
    matrix = gates.new_zeros(rank, rank)
    for gate, (start, end) in zip(gates, slices):
        matrix[start:end, start:end] = torch.eye(
            end - start, dtype=gates.dtype, device=gates.device
        ) * gate
    return matrix


def reconstruct_target(
    base_embeddings: torch.Tensor,
    delta_latent: torch.Tensor,
    gate_matrix: torch.Tensor,
    student_subspace: torch.Tensor,
) -> torch.Tensor:
    """Fused target ``H* = H_0 + delta_Z G U_0^T`` (proposal Eq. 6, row form)."""
    if base_embeddings.dim() != 2:
        raise ValueError("base_embeddings must be [N, d_S]")
    if delta_latent.dim() != 2:
        raise ValueError("delta_latent must be [N, r]")
    if gate_matrix.dim() != 2 or gate_matrix.shape[0] != gate_matrix.shape[1]:
        raise ValueError("gate_matrix must be square [r, r]")
    if student_subspace.dim() != 2:
        raise ValueError("student_subspace must be [d_S, r]")
    if base_embeddings.shape[1] != student_subspace.shape[0]:
        raise ValueError(
            "base embedding width must match the student subspace row count"
        )
    if delta_latent.shape[1] != gate_matrix.shape[0]:
        raise ValueError("delta_latent width must match the gate matrix")
    if gate_matrix.shape[0] != student_subspace.shape[1]:
        raise ValueError("gate matrix must match the subspace rank")

    correction = (delta_latent.to(base_embeddings.dtype) @ gate_matrix.to(
        base_embeddings.dtype
    )) @ student_subspace.transpose(0, 1).to(base_embeddings.dtype)
    return base_embeddings + correction


class OurMethodDistillation(nn.Module):
    """Fusion objective ``w_task * L_base + w_fusion * L_fusion``.

    The target is treated as a constant (the proposal's stop-gradient), so the
    only parameters that ever receive gradient are the trainable student's.
    """

    def __init__(
        self,
        w_task: float = 0.5,
        w_fusion: float = 1.0,
        normalize_target: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.w_task = w_task
        self.w_fusion = w_fusion
        self.normalize_target = normalize_target
        self.eps = eps

    def fusion_loss(
        self,
        student_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean ``1 - cos`` and mean cosine for one (student, target) pair."""
        target = target_embeddings.detach().to(student_embeddings.dtype)
        if self.normalize_target:
            target = F.normalize(target, p=2, dim=-1, eps=self.eps)
        cosine = F.cosine_similarity(
            student_embeddings, target, dim=-1, eps=self.eps
        )
        return (1.0 - cosine).mean(), cosine.mean()

    def forward(
        self,
        student_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        task_loss: torch.Tensor,
        extra_pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...] = (),
    ) -> tuple[torch.Tensor, dict[str, float]]:
        fusion_terms = [self.fusion_loss(student_embeddings, target_embeddings)]
        for student_side, target_side in extra_pairs:
            fusion_terms.append(self.fusion_loss(student_side, target_side))

        fusion = torch.stack([term[0] for term in fusion_terms]).mean()
        cosine = torch.stack([term[1] for term in fusion_terms]).mean()

        loss = self.w_task * task_loss + self.w_fusion * fusion
        metrics = {
            "loss_total": float(loss.detach()),
            "loss_base": float(task_loss.detach()),
            "loss_fusion": float(fusion.detach()),
            "cos_target": float(cosine.detach()),
        }
        return loss, metrics
