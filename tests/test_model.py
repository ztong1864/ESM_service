"""The real checkpoint: the one check that the service and the skill's script agree on the model
itself, not only on the shape of the answer. Left out of the default run (pytest.ini); on a host with
a model file:

    pytest -m model

The numbers are the ones the script printed for the same sequence.
"""

from __future__ import annotations

import pytest

from esm_service import config
from esm_service.engine import installed_models

pytestmark = pytest.mark.model

WILD = "MKTFFVALVLLLSGALAAPVA"


@pytest.fixture()
def real(client, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("esm")
    if "esm2-35m" not in installed_models():
        pytest.skip(f"esm2-35m is not installed in {config.MODELS_DIR}")
    monkeypatch.setattr(config, "MOCK", False)  # read per call, so nothing has to be reimported
    return client


def test_the_wild_type_probability_is_the_scripts_number(real):
    body = real.post("/api/mask-probabilities",
                     json={"sequence": WILD, "positions": [7], "standard_only": True, "top": 3}).json()
    assert body["model"] == "esm2-35m"
    assert body["engine"].startswith("fair-esm 2.0.0, local ")
    position = body["positions"][0]
    assert position["wild_type"] == "A"
    assert position["wild_type_rank"] == 1
    assert position["probabilities"]["A"] == pytest.approx(0.2281482524887472, rel=1e-12)
    assert position["top_standard"][0] == pytest.approx(
        {"aa": "A", "probability": 0.2281482524887472, "mutation": "A7A", "log_ratio_vs_wild_type": 0.0},
        rel=1e-12, abs=1e-12)


def test_the_probabilities_are_a_distribution(real):
    body = real.post("/api/mask-probabilities", json={"sequence": "MKTF<mask>VA"}).json()
    position = body["positions"][0]
    assert 0 < position["selected_probability_mass"] <= 1
    assert sum(position["probabilities"].values()) == pytest.approx(
        position["selected_probability_mass"], rel=1e-12)


def test_the_model_stays_loaded_between_requests(real):
    real.post("/api/mask-probabilities", json={"sequence": "MK<mask>TF"})
    health = real.get("/api/health").json()
    assert health["loaded"] == ["esm2-35m"]
    assert health["mock"] is False
