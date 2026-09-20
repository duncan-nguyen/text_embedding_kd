"""Offline construction of the student-anchored fused target.

The teacher and the base student are both frozen, so every quantity the target
depends on -- their corpus embeddings, the spectral subspaces, the Procrustes
alignment, the block gates -- is deterministic given the corpus. This module
computes those quantities once and caches each stage on disk, mirroring the
TALAS teacher cache: the cache is keyed only by its path, so changing the
corpus or the models means deleting the file and rebuilding.

Cache files derived from ``config.cache_path``::

    <stem>_teacher.pt   teacher embeddings (one row per corpus item, per side)
    <stem>_base.pt      base-student embeddings
    <stem>_views.pt     two semantics-preserving views per model, for the CKA gate
    <stem>_targets.pt   fused targets H* plus all diagnostics
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.criterions.our_method import (
    block_slices,
    block_stability,
    build_gate_matrix,
    linear_cka,
    orthogonal_procrustes,
    procrustes_error,
    reconstruct_target,
    stability_gates,
    top_spectral_subspace,
)
from src.pooling import last_token_pool, mean_pooling

__all__ = ["FusedTargets", "OurMethodTargetBuilder", "sibling_cache_path"]

_MASK_DROP_RATE = 0.15


def sibling_cache_path(cache_path: str, suffix: str) -> str:
    """Return ``<cache stem>_<suffix>.pt`` next to ``cache_path``."""
    path = Path(cache_path)
    return str(path.with_name(f"{path.stem}_{suffix}").with_suffix(".pt"))


@dataclass
class FusedTargets:
    targets: torch.Tensor
    diagnostics: dict
    num_sides: int
    rank: int
    slices: list[tuple[int, int]]


def _has_dropout(model: nn.Module) -> bool:
    return any(
        isinstance(module, nn.Dropout) and module.p > 0 for module in model.modules()
    )


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().to(torch.float64) - left.detach().to(torch.float64).mean()
    right = right.detach().to(torch.float64) - right.detach().to(torch.float64).mean()
    denominator = left.norm() * right.norm()
    if float(denominator) < 1e-12:
        return 0.0
    return float((left * right).sum() / denominator)


class OurMethodTargetBuilder:
    def __init__(
        self,
        model_teacher: nn.Module,
        model_base: nn.Module,
        device_s: torch.device,
        device_t: torch.device,
        config,
        mask_token_ids: dict[str, int] | None = None,
    ):
        self.model_teacher = model_teacher
        self.model_base = model_base
        self.device_s = device_s
        self.device_t = device_t
        self.config = config
        both_sides = (
            config.target_view == "both" and getattr(config, "task_type", None) != "single_cls"
        )
        self.sides = [1, 2] if both_sides else [1]
        self.rank = int(config.subspace_rank)
        self.num_blocks = int(config.num_blocks)
        self.slices = block_slices(self.rank, self.num_blocks)
        self.seed = int(config.seed)
        self.cache_path = str(config.cache_path)
        self._mask_token_ids = dict(mask_token_ids or {})

        # Teacher and base student must use the *same* view mechanism, otherwise
        # the two CKA scores are not comparable and the gate is biased.
        self.teacher_view_mode = self.base_view_mode = self._resolve_view_mode()

    # ------------------------------------------------------------------
    # cache helpers
    # ------------------------------------------------------------------

    def _resolve_view_mode(self) -> str:
        requested = getattr(self.config, "stability_view", "auto")
        if requested in ("dropout", "augment"):
            return requested
        if _has_dropout(self.model_teacher) and _has_dropout(self.model_base):
            return "dropout"
        print(
            "[OurMethod] Dropout is absent in at least one frozen model; using "
            "token-masking augmentation as the shared stability view."
        )
        return "augment"

    @staticmethod
    def _save(path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(payload, path)

    @staticmethod
    def _load(path: str) -> dict:
        return torch.load(path, map_location="cpu")

    # ------------------------------------------------------------------
    # pooling
    # ------------------------------------------------------------------

    @staticmethod
    def _pool(hidden: torch.Tensor, mask: torch.Tensor, method: str) -> torch.Tensor:
        if method == "last_token":
            return last_token_pool(hidden, mask)
        if method == "mean":
            return mean_pooling(hidden, mask)
        if method == "cls":
            return hidden[:, 0, :]
        raise ValueError(f"Unknown pooling method: {method}")

    # ------------------------------------------------------------------
    # forwards
    # ------------------------------------------------------------------

    def _encode_deterministic(
        self,
        model: nn.Module,
        dataloader,
        suffix: str,
        pooling: str,
        device: torch.device,
    ) -> torch.Tensor:
        per_side = [[] for _ in self.sides]
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"embedding({suffix})", leave=False):
                for index, side in enumerate(self.sides):
                    input_ids = batch[f"input_ids{side}_{suffix}"].to(device)
                    attention_mask = batch[f"attention_mask{side}_{suffix}"].to(device)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                    pooled = self._pool(output.last_hidden_state, attention_mask, pooling)
                    per_side[index].append(pooled.float().cpu())
        side_tensors = [torch.cat(chunks, dim=0) for chunks in per_side]
        return torch.stack(side_tensors, dim=1)

    def _augment(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        mask_token_id: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        augmented = input_ids.clone()
        keep = special_tokens_mask.to(input_ids.device) == 0
        random = torch.rand(
            input_ids.shape, generator=generator, device=input_ids.device
        )
        positions = keep & (random < _MASK_DROP_RATE)
        augmented[positions] = mask_token_id
        return augmented

    def _encode_views(
        self,
        model: nn.Module,
        dataloader,
        suffix: str,
        pooling: str,
        device: torch.device,
        mode: str,
    ) -> torch.Tensor:
        per_side = [[] for _ in self.sides]
        mask_token_id = self._mask_token_ids.get(suffix, 0)
        was_training = model.training
        if mode == "dropout":
            model.train()
        else:
            model.eval()
        try:
            with torch.no_grad():
                for batch in tqdm(
                    dataloader, desc=f"views({suffix},{mode})", leave=False
                ):
                    for index, side in enumerate(self.sides):
                        input_ids = batch[f"input_ids{side}_{suffix}"].to(device)
                        attention_mask = batch[f"attention_mask{side}_{suffix}"].to(device)
                        if mode == "dropout":
                            view1 = self._pool(
                                model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=True,
                                ).last_hidden_state,
                                attention_mask,
                                pooling,
                            )
                            view2 = self._pool(
                                model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=True,
                                ).last_hidden_state,
                                attention_mask,
                                pooling,
                            )
                        else:
                            special = batch[
                                f"special_tokens_mask{side}_{suffix}"
                            ].to(device)
                            generator = torch.Generator(
                                device=input_ids.device.type
                            ).manual_seed(self.seed + 1000 * side)
                            ids1 = self._augment(
                                input_ids, special, mask_token_id, generator
                            )
                            generator = torch.Generator(
                                device=input_ids.device.type
                            ).manual_seed(self.seed + 5000 + 1000 * side)
                            ids2 = self._augment(
                                input_ids, special, mask_token_id, generator
                            )
                            view1 = self._pool(
                                model(
                                    input_ids=ids1,
                                    attention_mask=attention_mask,
                                    return_dict=True,
                                ).last_hidden_state,
                                attention_mask,
                                pooling,
                            )
                            view2 = self._pool(
                                model(
                                    input_ids=ids2,
                                    attention_mask=attention_mask,
                                    return_dict=True,
                                ).last_hidden_state,
                                attention_mask,
                                pooling,
                            )
                        per_side[index].append(
                            torch.stack([view1.float().cpu(), view2.float().cpu()], dim=1)
                        )
        finally:
            model.train(was_training)
        side_tensors = [torch.cat(chunks, dim=0) for chunks in per_side]
        return torch.stack(side_tensors, dim=1)

    # ------------------------------------------------------------------
    # cached stage builders
    # ------------------------------------------------------------------

    def _load_or_build_det(
        self,
        dataloader,
        num_samples: int,
        model: nn.Module,
        suffix: str,
        pooling: str,
        device: torch.device,
    ) -> torch.Tensor:
        path = sibling_cache_path(self.cache_path, suffix)
        if os.path.exists(path) and not self.config.force_recompute:
            payload = self._load(path)
            if payload["n"] != num_samples:
                raise ValueError(
                    f"Cached {suffix} embeddings have {payload['n']} rows but the "
                    f"corpus has {num_samples}. Remove {path} and rebuild."
                )
            print(f"[OurMethod] loaded {suffix} embeddings from {path}")
            return payload["tensor"]
        tensor = self._encode_deterministic(model, dataloader, suffix, pooling, device)
        self._save(path, {"tensor": tensor, "n": num_samples})
        print(f"[OurMethod] cached {suffix} embeddings to {path}")
        return tensor

    def _load_or_build_views(
        self,
        dataloader,
        num_samples: int,
        model: nn.Module,
        suffix: str,
        pooling: str,
        device: torch.device,
        mode: str,
    ) -> torch.Tensor:
        path = sibling_cache_path(self.cache_path, f"{suffix}_views")
        if os.path.exists(path) and not self.config.force_recompute:
            payload = self._load(path)
            if payload["n"] != num_samples:
                raise ValueError(
                    f"Cached {suffix} views have {payload['n']} rows but the corpus "
                    f"has {num_samples}. Remove {path} and rebuild."
                )
            print(f"[OurMethod] loaded {suffix} views from {path}")
            return payload["tensor"]
        tensor = self._encode_views(model, dataloader, suffix, pooling, device, mode)
        self._save(path, {"tensor": tensor, "n": num_samples, "mode": mode})
        print(f"[OurMethod] cached {suffix} views to {path}")
        return tensor

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def load_or_build(self, dataloader, num_samples: int) -> FusedTargets:
        if os.path.exists(self.cache_path) and not self.config.force_recompute:
            payload = self._load(self.cache_path)
            if payload["targets"].shape[0] != num_samples:
                raise ValueError(
                    f"Cached targets have {payload['targets'].shape[0]} rows but the "
                    f"corpus has {num_samples}. Remove {self.cache_path} and rebuild."
                )
            print(f"[OurMethod] loaded fused targets from {self.cache_path}")
            return FusedTargets(
                targets=payload["targets"],
                diagnostics=payload["diagnostics"],
                num_sides=payload["num_sides"],
                rank=payload["rank"],
                slices=[tuple(pair) for pair in payload["slices"]],
            )

        teacher = self._load_or_build_det(
            dataloader,
            num_samples,
            self.model_teacher,
            "tea",
            self.config.pooling_method,
            self.device_t,
        )
        base = self._load_or_build_det(
            dataloader,
            num_samples,
            self.model_base,
            "stu",
            "cls",
            self.device_s,
        )
        teacher_views = self._load_or_build_views(
            dataloader,
            num_samples,
            self.model_teacher,
            "tea",
            self.config.pooling_method,
            self.device_t,
            self.teacher_view_mode,
        )
        base_views = self._load_or_build_views(
            dataloader,
            num_samples,
            self.model_base,
            "stu",
            "cls",
            self.device_s,
            self.base_view_mode,
        )

        result = self._fuse(teacher, base, teacher_views, base_views)
        self._save(
            self.cache_path,
            {
                "targets": result.targets,
                "diagnostics": result.diagnostics,
                "num_sides": result.num_sides,
                "rank": result.rank,
                "slices": [list(pair) for pair in result.slices],
            },
        )
        print(f"[OurMethod] cached fused targets to {self.cache_path}")
        return result

    def _fuse(
        self,
        teacher: torch.Tensor,
        base: torch.Tensor,
        teacher_views: torch.Tensor,
        base_views: torch.Tensor,
    ) -> FusedTargets:
        num_samples = teacher.shape[0]
        max_estimate = getattr(self.config, "max_target_samples", None)
        side_count = len(self.sides)
        if max_estimate is not None and num_samples > max_estimate:
            generator = torch.Generator().manual_seed(self.seed)
            estimate_idx = torch.randperm(num_samples, generator=generator)[:max_estimate]
        else:
            estimate_idx = torch.arange(num_samples)
        expanded_idx = (
            estimate_idx[:, None] * side_count + torch.arange(side_count)
        ).reshape(-1)

        teacher_flat = teacher.reshape(num_samples * side_count, teacher.shape[-1])
        base_flat = base.reshape(num_samples * side_count, base.shape[-1])
        teacher_sub = top_spectral_subspace(
            teacher_flat[expanded_idx], self.rank, block_slices=self.slices
        )
        base_sub = top_spectral_subspace(
            base_flat[expanded_idx], self.rank, block_slices=self.slices
        )
        u_teacher = teacher_sub.U
        u_base = base_sub.U

        z_teacher = teacher @ u_teacher
        z_base = base @ u_base

        num_estimate = estimate_idx.numel()
        source = z_teacher[estimate_idx].reshape(num_estimate * len(self.sides), self.rank)
        target = z_base[estimate_idx].reshape(num_estimate * len(self.sides), self.rank)
        error_before = procrustes_error(source, target)
        rotation = orthogonal_procrustes(source, target)
        error_after = procrustes_error(source, target, rotation)
        aligned_teacher = z_teacher @ rotation

        teacher_stability = self._latent_stability(
            teacher_views, u_teacher, rotation, estimate_idx
        )
        base_stability = self._latent_stability(
            base_views, u_base, None, estimate_idx
        )
        delta_stability = teacher_stability - base_stability
        gates = stability_gates(
            delta_stability,
            margin=float(self.config.stability_margin),
            tau=float(self.config.stability_tau),
        )
        gate_matrix = build_gate_matrix(gates, self.slices, self.rank)

        target_sides = []
        cos_base_terms = []
        cos_teacher_terms = []
        correction_norms = []
        for index, _ in enumerate(self.sides):
            base_side = base[:, index, :]
            delta = aligned_teacher[:, index, :] - z_base[:, index, :]
            fused = reconstruct_target(base_side, delta, gate_matrix, u_base)
            target_sides.append(fused)
            cos_base_terms.append(F.cosine_similarity(fused, base_side, dim=-1).mean())
            z_star = z_base[:, index, :] + delta @ gate_matrix
            cos_teacher_terms.append(
                F.cosine_similarity(z_star, aligned_teacher[:, index, :], dim=-1).mean()
            )
            correction_norms.append((delta @ gate_matrix).norm())

        targets = torch.stack(target_sides, dim=1)
        if getattr(self.config, "normalize_target", False):
            targets = F.normalize(targets, p=2, dim=-1)

        delta_norm = torch.stack(
            [
                (aligned_teacher[:, i, :] - z_base[:, i, :]).norm()
                for i in range(len(self.sides))
            ]
        ).mean()
        inject_ratio = (
            torch.stack(correction_norms).mean() / delta_norm.clamp_min(1e-12)
        )

        diagnostics = {
            "subspace": {
                "teacher_eigenvalues": teacher_sub.eigenvalues,
                "student_eigenvalues": base_sub.eigenvalues,
                "teacher_explained_variance": float(teacher_sub.explained_variance),
                "student_explained_variance": float(base_sub.explained_variance),
                "teacher_effective_rank": float(teacher_sub.effective_rank),
                "student_effective_rank": float(base_sub.effective_rank),
                "teacher_condition_number": float(teacher_sub.condition_number),
                "student_condition_number": float(base_sub.condition_number),
                "teacher_block_energy": teacher_sub.block_energy,
                "student_block_energy": base_sub.block_energy,
            },
            "alignment": {
                "error_before": float(error_before),
                "error_after": float(error_after),
                "improvement": float(1.0 - error_after / error_before.clamp_min(1e-12)),
                "latent_cka": float(linear_cka(source, target)),
                "residual_norm": float(delta_norm),
            },
            "stability": {
                "teacher": teacher_stability,
                "student": base_stability,
                "delta": delta_stability,
                "teacher_mean": float(teacher_stability.mean()),
                "student_mean": float(base_stability.mean()),
                "delta_mean": float(delta_stability.mean()),
                "delta_std": float(delta_stability.std(unbiased=False)),
                "teacher_win_ratio": float((delta_stability > 0).float().mean()),
                "student_win_ratio": float((delta_stability <= 0).float().mean()),
                "margin_pass_ratio": float(
                    (delta_stability > float(self.config.stability_margin))
                    .float()
                    .mean()
                ),
            },
            "gate": {
                "values": gates,
                "mean": float(gates.mean()),
                "std": float(gates.std(unbiased=False)),
                "min": float(gates.min()),
                "max": float(gates.max()),
                "zero_ratio": float((gates == 0).float().mean()),
                "low_ratio": float((gates < 0.25).float().mean()),
                "mid_ratio": float(((gates >= 0.25) & (gates < 0.75)).float().mean()),
                "high_ratio": float((gates >= 0.75).float().mean()),
                "inject_ratio": float(inject_ratio),
                "score_delta_corr": _pearson(gates, delta_stability),
                "score_energy_corr": _pearson(gates, base_sub.block_energy),
            },
            "target": {
                "cos_target_base": float(torch.stack(cos_base_terms).mean()),
                "cos_target_teacher": float(torch.stack(cos_teacher_terms).mean()),
                "target_displacement": float(
                    1.0 - torch.stack(cos_base_terms).mean()
                ),
                "correction_norm": float(torch.stack(correction_norms).mean()),
            },
        }

        return FusedTargets(
            targets=targets,
            diagnostics=diagnostics,
            num_sides=len(self.sides),
            rank=self.rank,
            slices=self.slices,
        )

    def _latent_stability(
        self,
        views: torch.Tensor,
        subspace: torch.Tensor,
        rotation: torch.Tensor | None,
        estimate_idx: torch.Tensor,
    ) -> torch.Tensor:
        selected = views[estimate_idx]
        latent = selected @ subspace
        if rotation is not None:
            latent = latent @ rotation
        num_rows = selected.shape[0] * len(self.sides)
        latent = latent.reshape(num_rows, 2, self.rank)
        return block_stability(latent[:, 0, :], latent[:, 1, :], self.slices)
