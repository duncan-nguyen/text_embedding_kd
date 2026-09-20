"""Diagnostic plots and scalar logging for the OurMethod fused target.

The proposal lists five recommended plots. Four of them depend only on the
offline fused-target statistics and are produced here; the fifth
(``acquisition_vs_retention``) needs per-sample behaviour analysis on discrete
tasks, which is not part of this pass.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["flatten_ourmethod_diagnostics", "save_ourmethod_diagnostics"]


def _to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return [float(item) for item in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def flatten_ourmethod_diagnostics(diagnostics: dict) -> dict[str, Any]:
    """Flatten the nested diagnostics into ``namespace/name`` scalars."""
    subspace = diagnostics["subspace"]
    alignment = diagnostics["alignment"]
    stability = diagnostics["stability"]
    gate = diagnostics["gate"]
    target = diagnostics["target"]

    flat: dict[str, Any] = {
        "alignment/error_before": alignment["error_before"],
        "alignment/error_after": alignment["error_after"],
        "alignment/improvement": alignment["improvement"],
        "alignment/latent_cka": alignment["latent_cka"],
        "alignment/residual_norm": alignment["residual_norm"],
        "subspace/teacher_explained_variance": subspace["teacher_explained_variance"],
        "subspace/student_explained_variance": subspace["student_explained_variance"],
        "subspace/teacher_effective_rank": subspace["teacher_effective_rank"],
        "subspace/student_effective_rank": subspace["student_effective_rank"],
        "subspace/teacher_condition_number": subspace["teacher_condition_number"],
        "subspace/student_condition_number": subspace["student_condition_number"],
        "stability/teacher_mean": stability["teacher_mean"],
        "stability/student_mean": stability["student_mean"],
        "stability/delta_mean": stability["delta_mean"],
        "stability/delta_std": stability["delta_std"],
        "stability/teacher_win_ratio": stability["teacher_win_ratio"],
        "stability/student_win_ratio": stability["student_win_ratio"],
        "stability/margin_pass_ratio": stability["margin_pass_ratio"],
        "gate/mean": gate["mean"],
        "gate/std": gate["std"],
        "gate/min": gate["min"],
        "gate/max": gate["max"],
        "gate/zero_ratio": gate["zero_ratio"],
        "gate/low_ratio": gate["low_ratio"],
        "gate/mid_ratio": gate["mid_ratio"],
        "gate/high_ratio": gate["high_ratio"],
        "gate/inject_ratio": gate["inject_ratio"],
        "gate/score_delta_corr": gate["score_delta_corr"],
        "gate/score_energy_corr": gate["score_energy_corr"],
        "target/cos_target_base": target["cos_target_base"],
        "target/cos_target_teacher": target["cos_target_teacher"],
        "target/target_displacement": target["target_displacement"],
        "target/correction_norm": target["correction_norm"],
        "gate/values": _to_list(gate["values"]),
        "stability/delta_by_block": _to_list(stability["delta"]),
    }
    return flat


def save_ourmethod_diagnostics(diagnostics: dict, output_dir: str) -> list[str]:
    """Write the four offline diagnostics plots and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    weights = diagnostics["gate"]["values"]
    delta = diagnostics["stability"]["delta"]
    teacher_stability = diagnostics["stability"]["teacher"]
    student_stability = diagnostics["stability"]["student"]
    teacher_energy = diagnostics["subspace"]["teacher_block_energy"]
    student_energy = diagnostics["subspace"]["student_block_energy"]

    gate_values = _to_list(weights)
    delta_values = _to_list(delta)
    teacher_values = _to_list(teacher_stability)
    student_values = _to_list(student_stability)
    teacher_energy_values = _to_list(teacher_energy)
    student_energy_values = _to_list(student_energy)
    blocks = list(range(1, len(gate_values) + 1))
    paths = []

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(gate_values, bins=min(20, max(1, len(gate_values))), range=(0.0, 1.0))
    axis.set_xlabel("gate value")
    axis.set_ylabel("block count")
    axis.set_title("Gate histogram")
    paths.append(_save(figure, output_dir, "gate_histogram"))

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(blocks, teacher_values, marker="o", label="teacher")
    axis.plot(blocks, student_values, marker="s", label="base student")
    axis.plot(blocks, delta_values, marker="^", label="delta")
    axis.set_xlabel("spectral block")
    axis.set_ylabel("cross-view CKA")
    axis.set_title("Relative subspace stability by block")
    axis.legend()
    paths.append(_save(figure, output_dir, "delta_stability_by_block"))

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(delta_values, gate_values)
    axis.axvline(0.0, color="grey", linestyle="--", linewidth=1)
    axis.set_xlabel("delta stability (teacher - student)")
    axis.set_ylabel("gate value")
    axis.set_title("Gate vs delta stability")
    paths.append(_save(figure, output_dir, "gate_vs_delta_stability"))

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(blocks, teacher_energy_values, marker="o", label="teacher")
    axis.plot(blocks, student_energy_values, marker="s", label="base student")
    axis.set_xlabel("spectral block")
    axis.set_ylabel("spectral energy")
    axis.set_title("Spectral energy teacher vs student")
    axis.legend()
    paths.append(_save(figure, output_dir, "spectral_energy_teacher_vs_student"))

    return paths


def _save(figure, output_dir: str, name: str) -> str:
    import matplotlib.pyplot as plt

    path = os.path.join(output_dir, f"{name}.png")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
