"""Regression guards for the method config classes.

These pin the config surface the trainer and the CLI depend on, so a cleanup
that drops or renames a field fails here instead of at training time.
"""

import pytest

from config import (
    BaseConfig,
    CDMConfig,
    DSKDConfig,
    EMOConfig,
    PKTConfig,
    RKDConfig,
    StellaConfig,
    TALASConfig,
)

ALL_CONFIGS = [CDMConfig, DSKDConfig, EMOConfig, PKTConfig, RKDConfig, StellaConfig, TALASConfig]


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_every_method_config_inherits_base(config_cls):
    assert issubclass(config_cls, BaseConfig)


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_distill_method_is_the_lowercase_class_prefix(config_cls):
    expected = config_cls.__name__.removesuffix("Config").lower()
    assert config_cls().distill_method == expected


def test_constructor_overrides_known_attributes():
    config = RKDConfig(dist_ratio=3.0, epochs=7)

    assert config.dist_ratio == 3.0
    assert config.epochs == 7


def test_constructor_ignores_unknown_attributes():
    config = RKDConfig(not_a_real_field=1)

    assert not hasattr(config, "not_a_real_field")


def test_to_dict_merges_class_defaults_and_instance_overrides():
    config = RKDConfig(dist_ratio=3.0)
    values = config.to_dict()

    assert values["distill_method"] == "rkd"
    assert values["dist_ratio"] == 3.0
    assert values["epochs"] == config.epochs


def test_to_dict_excludes_private_names_and_callables():
    values = BaseConfig().to_dict()

    assert all(not key.startswith("_") for key in values)
    assert all(not callable(value) for value in values.values())


def test_removed_dead_config_fields_stay_removed():
    assert not hasattr(BaseConfig(), "eval_data_path")
    assert not hasattr(BaseConfig(), "evaluate_test_each_epoch")

    dskd = DSKDConfig()
    assert not hasattr(dskd, "use_cross_attention")
    assert not hasattr(dskd, "bidirectional_align")

    assert not hasattr(TALASConfig(), "cache_teacher")


def test_pkt_paper_defaults():
    config = PKTConfig()

    assert config.w_task == 0.0
    assert config.kernel == "cosine"
    assert config.exclude_self is True
    assert config.reduction == "batchmean"


def test_rkd_paper_ratios():
    config = RKDConfig()

    assert config.dist_ratio == 1.0
    assert config.angle_ratio == 2.0
    assert config.huber_delta == 1.0
