"""api.md is the contract callers read, so a field cannot be added without documenting it."""

from __future__ import annotations

import re
from pathlib import Path

from esm_service.models import MODELS
from esm_service.schemas import MaskProbabilitiesRequest

ROOT = Path(__file__).resolve().parents[1]


def test_api_md_documents_every_request_field():
    table = (ROOT / "api.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^\| `(\w+)`", table, re.M))
    assert documented == set(MaskProbabilitiesRequest.model_fields)


def test_every_model_name_is_unique_and_lower_case():
    assert list(MODELS) == [name.lower() for name in MODELS]
    assert len({spec.file_stem for spec in MODELS.values()}) == len(MODELS)
