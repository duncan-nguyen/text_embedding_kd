"""Regression guards for the public package surfaces.

The evaluation package previously advertised ``*_custom`` names whose import was
commented out; an ``__all__`` entry that does not resolve breaks ``import *``.
These tests keep every advertised name importable.
"""

import importlib

import pytest


def test_src_public_exports_resolve():
    import src

    for name in src.__all__:
        assert hasattr(src, name), name


def test_criterion_package_exports_resolve():
    import src.criterions as criterions

    for name in criterions.__all__:
        assert hasattr(criterions, name), name


def test_config_package_exports_resolve():
    import config

    for name in config.__all__:
        assert hasattr(config, name), name


def test_evaluation_package_exports_resolve():
    import src.evaluation as evaluation

    for name in evaluation.__all__:
        assert hasattr(evaluation, name), name


def test_evaluation_package_advertises_no_custom_aliases():
    import src.evaluation as evaluation

    assert not [name for name in evaluation.__all__ if name.endswith("_custom")]


def test_removed_dead_evaluation_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.evaluation.evaluation_model_define")


def test_removed_dead_helpers_are_gone():
    import src
    from src import cache_teacher, loss

    assert not hasattr(src, "compute_variance")
    assert not hasattr(src, "clear_cache_and_free_memory")
    assert not hasattr(loss, "compute_variance")
    assert not hasattr(cache_teacher, "clear_cache_and_free_memory")
