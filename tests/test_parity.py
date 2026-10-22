"""The service must keep answering exactly what the skill's script printed.

tests/golden/*.json were captured from the esm-directed-evolution script before it became a client of
this service, in mock mode: mock logits are seeded by the sha256 of the variant's own tokens, so the
numbers are reproducible on any machine. Those files are the frozen contract of the old implementation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden"
WILD = "MKTFFVALVLLLSGALAAPVA"

# Each case: the request body, and the script arguments that produced the same answer.
CASES = {
    "mask_in_sequence": (
        {"sequence": "MKTFFV<mask>LVLLLSGALAAPVA"},
        ["--sequence", "MKTFFV<mask>LVLLLSGALAAPVA"],
    ),
    "two_positions_standard_only": (
        {"sequence": WILD, "positions": [2, 7], "standard_only": True, "top": 3},
        ["--sequence", WILD, "--positions", "2,7", "--standard-only", "--top", "3"],
    ),
    "independent_renormalized": (
        {"sequence": WILD, "positions": [2, 7], "independent": True, "renormalize_selected": True},
        ["--sequence", WILD, "--positions", "2,7", "--independent", "--renormalize-selected"],
    ),
}


def golden(name: str) -> dict:
    return json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(CASES))
def test_answer_matches_the_script_field_for_field(client, name):
    body = client.post("/api/mask-probabilities", json=CASES[name][0]).json()
    assert body.pop("elapsed_s") >= 0  # the only field the script never had
    assert body == golden(name)


# The skill's script, so its output can be checked against the same golden files:
#   ESM_SKILL_SCRIPT=.../esm-directed-evolution/scripts/esm_mask_probabilities.py pytest -q
SCRIPT = os.environ.get("ESM_SKILL_SCRIPT")


@pytest.mark.skipif(not SCRIPT, reason="set ESM_SKILL_SCRIPT to also check the skill's script")
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_script_prints_what_the_service_answers(service_url, name):
    command = [sys.executable, SCRIPT, *CASES[name][1], "--base-url", service_url]
    printed = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    answer = json.loads(printed)
    answer.pop("elapsed_s", None)
    assert answer == golden(name)
