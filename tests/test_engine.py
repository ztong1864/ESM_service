"""The resident engine's bookkeeping: which files count as installed, what stays in memory, which
devices are accepted. No torch and no checkpoint -- loading is replaced by a stand-in."""

from __future__ import annotations

import pytest

from esm_service import config
from esm_service.engine import Engine, LoadedModel, ModelNotAvailable, installed_models, resolve_device


def test_installed_models_maps_file_stems_to_short_names(tmp_path):
    for name in ("esm2_t12_35M_UR50D.pt", "esm2_t12_35M_UR50D-contact-regression.pt", "mine.pt"):
        (tmp_path / name).touch()
    # The regression file belongs to the model, so it is not a model of its own; an unknown file keeps
    # its stem, because an administrator may have added it.
    assert installed_models(tmp_path) == ["esm2-35m", "mine"]


def test_installed_models_is_empty_without_the_folder(tmp_path):
    assert installed_models(tmp_path / "nothing-here") == []


def _stand_in(self, key, requested):
    return LoadedModel(name=key, model=None, alphabet=None, vocab=[], device="cpu", engine="stand-in")


def test_the_same_model_is_loaded_once(monkeypatch):
    monkeypatch.setattr(Engine, "_load_now", _stand_in)
    engine = Engine()
    assert engine.load("esm2-35m") is engine.load("ESM2-35M")  # the name is normalized


def test_the_least_recently_used_model_is_dropped(monkeypatch):
    monkeypatch.setattr(Engine, "_load_now", _stand_in)
    monkeypatch.setattr(config, "MAX_LOADED_MODELS", 1)
    engine = Engine()
    engine.load("esm2-35m")
    engine.load("esm2-650m")
    assert engine.loaded_names() == ["esm2-650m"]


def test_a_file_stem_resolves_to_its_short_name(monkeypatch):
    monkeypatch.setattr(Engine, "_load_now", _stand_in)
    engine = Engine()
    assert engine.load("esm2_t12_35M_UR50D").name == "esm2-35m"


def test_a_missing_file_says_what_is_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    with pytest.raises(ModelNotAvailable, match="No model is installed"):
        Engine().model_path("esm2-650m")
    (tmp_path / "esm2_t12_35M_UR50D.pt").touch()
    with pytest.raises(ModelNotAvailable, match="Installed: esm2-35m"):
        Engine().model_path("esm2-650m")


def test_a_name_that_is_not_a_model_never_becomes_a_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    with pytest.raises(ModelNotAvailable, match="is not a model name"):
        Engine().model_path("../../elsewhere/model")


def test_the_device_must_be_a_known_value():
    with pytest.raises(ModelNotAvailable, match="auto, cpu or cuda"):
        resolve_device("tpu")
    assert resolve_device("cpu") == "cpu"  # answered without importing torch
