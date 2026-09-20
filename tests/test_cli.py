"""Regression guards for the CLI -> config bridge in main.py."""

import sys

import pytest

from config import (
    CDMConfig,
    DSKDConfig,
    EMOConfig,
    OurMethodConfig,
    PKTConfig,
    RKDConfig,
    StellaConfig,
    TALASConfig,
)
from main import get_config, parse_args

METHOD_CONFIGS = [
    ("cdm", CDMConfig),
    ("dskd", DSKDConfig),
    ("emo", EMOConfig),
    ("ourmethod", OurMethodConfig),
    ("pkt", PKTConfig),
    ("rkd", RKDConfig),
    ("stella", StellaConfig),
    ("talas", TALASConfig),
]


def build_config(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    args = parse_args()
    return get_config(args.method, args)


@pytest.mark.parametrize("method,config_cls", METHOD_CONFIGS)
def test_method_selects_its_config_class(monkeypatch, method, config_cls):
    config = build_config(monkeypatch, "--method", method)

    assert isinstance(config, config_cls)
    assert config.distill_method == method


def test_default_method_is_cdm(monkeypatch):
    config = build_config(monkeypatch)

    assert config.distill_method == "cdm"


def test_every_flag_maps_to_its_config_attribute(monkeypatch):
    config = build_config(
        monkeypatch,
        "--method", "rkd",
        "--train_data", "train.csv",
        "--student_model", "student",
        "--teacher_model", "teacher",
        "--teacher_pooling", "mean",
        "--batch_size", "8",
        "--epochs", "2",
        "--lr", "1e-4",
        "--max_length", "64",
        "--w_task", "0.3",
        "--dist_ratio", "1.5",
        "--angle_ratio", "0.0",
        "--task_type", "pair_reg",
        "--save_dir", "out",
        "--weights_dir", "weights",
        "--cache_path", "cache.pt",
        "--seed", "7",
        "--num_workers", "3",
        "--wandb_project", "proj",
        "--wandb_run_name", "run",
        "--wandb_mode", "offline",
    )

    assert config.train_data_path == "train.csv"
    assert config.student_model_name == "student"
    assert config.teacher_model_name == "teacher"
    assert config.pooling_method == "mean"
    assert config.batch_size == 8
    assert config.epochs == 2
    assert config.learning_rate == 1e-4
    assert config.max_length == 64
    assert config.w_task == 0.3
    assert config.dist_ratio == 1.5
    assert config.angle_ratio == 0.0
    assert config.task_type == "pair_reg"
    assert config.save_dir == "out"
    assert config.weights_dir == "weights"
    assert config.cache_path == "cache.pt"
    assert config.seed == 7
    assert config.num_workers == 3
    assert config.wandb_project == "proj"
    assert config.wandb_run_name == "run"
    assert config.wandb_mode == "offline"


@pytest.mark.parametrize("method,flag,attribute", [
    ("pkt", "--w_pkt", "w_pkt"),
    ("pkt", "--pkt_kernel", "kernel"),
    ("cdm", "--alpha_dtw", "alpha_dtw"),
])
def test_method_specific_flags(monkeypatch, method, flag, attribute):
    config = build_config(monkeypatch, "--method", method, flag, "2.5" if flag != "--pkt_kernel" else "gaussian")

    expected = "gaussian" if flag == "--pkt_kernel" else 2.5
    assert getattr(config, attribute) == expected


def test_ourmethod_flags_map_to_config(monkeypatch):
    config = build_config(
        monkeypatch,
        "--method", "ourmethod",
        "--base_student_model", "base",
        "--teacher_pooling", "cls",
        "--teacher_dtype", "float32",
        "--subspace_rank", "32",
        "--num_blocks", "4",
        "--stability_margin", "0.1",
        "--stability_tau", "0.2",
        "--stability_view", "augment",
        "--target_view", "text1",
        "--w_fusion", "0.7",
        "--normalize_target",
        "--max_target_samples", "1000",
        "--target_batch_size", "64",
        "--eval_every", "2",
        "--force_recompute",
        "--no_diagnostics",
    )

    assert config.base_student_model_name == "base"
    assert config.pooling_method == "cls"
    assert config.teacher_dtype == "float32"
    assert config.subspace_rank == 32
    assert config.num_blocks == 4
    assert config.stability_margin == 0.1
    assert config.stability_tau == 0.2
    assert config.stability_view == "augment"
    assert config.target_view == "text1"
    assert config.w_fusion == 0.7
    assert config.normalize_target is True
    assert config.max_target_samples == 1000
    assert config.target_batch_size == 64
    assert config.eval_every == 2
    assert config.force_recompute is True
    assert config.diagnostics is False


def test_debug_flag_sets_debug_align(monkeypatch):
    config = build_config(monkeypatch, "--method", "cdm", "--debug")

    assert config.debug_align is True


def test_no_wandb_disables_logging(monkeypatch):
    config = build_config(monkeypatch, "--method", "cdm", "--no_wandb")

    assert config.use_wandb is False


def test_save_every_override(monkeypatch):
    config = build_config(monkeypatch, "--method", "talas", "--save_every", "3")

    assert config.save_every == 3


def test_non_positive_save_every_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--method", "talas", "--save_every", "0"])
    args = parse_args()

    with pytest.raises(ValueError, match="positive integer"):
        get_config(args.method, args)


def test_eval_every_zero_disables_in_training_eval(monkeypatch):
    config = build_config(monkeypatch, "--method", "ourmethod", "--eval_every", "0")

    assert config.eval_every == 0


def test_negative_eval_every_is_rejected(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--method", "ourmethod", "--eval_every", "-1"]
    )
    args = parse_args()

    with pytest.raises(ValueError, match="eval_every"):
        get_config(args.method, args)


def test_eval_data_flag_no_longer_exists(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--eval_data", "x"])

    with pytest.raises(SystemExit):
        parse_args()
