"""API behaviour in mock mode: the shape of the answer, and what each bad request returns."""

from __future__ import annotations

WILD = "MKTFFVALVLLLSGALAAPVA"


def test_health_reports_mock(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["mock"] is True
    assert body["default_model"]


def test_models_lists_the_known_ones(client):
    body = client.get("/api/models").json()
    names = [item["name"] for item in body["models"]]
    assert "esm2-35m" in names and "esm2-650m" in names
    assert all("installed" in item for item in body["models"])


def test_mask_in_the_sequence(client):
    body = client.post("/api/mask-probabilities", json={"sequence": "MKTFFV<mask>LVLLLSGALAAPVA"}).json()
    assert body["masking"] == "joint"
    assert body["sequence"] == "MKTFFV<mask>LVLLLSGALAAPVA"
    assert len(body["positions"]) == 1
    position = body["positions"][0]
    assert position["mask_index"] == 0 and position["position"] == 7
    assert "wild_type" not in position
    assert len(position["top_standard"]) == 5
    assert "<mask>" in body["token_order"]


def test_positions_give_mutations_and_wild_type_rank(client):
    body = client.post("/api/mask-probabilities",
                       json={"sequence": WILD, "positions": [7, 2], "standard_only": True, "top": 3}).json()
    assert body["token_order"] == list("ACDEFGHIKLMNPQRSTVWY")
    assert [item["position"] for item in body["positions"]] == [2, 7]
    first = body["positions"][0]
    assert first["wild_type"] == "K"
    assert first["top_standard"][0]["mutation"].startswith("K2")
    assert "log_ratio_vs_wild_type" in first["top_standard"][0]
    assert 1 <= first["wild_type_rank"] <= 20


def test_positions_accept_a_comma_string(client):
    body = client.post("/api/mask-probabilities", json={"sequence": WILD, "positions": "2,7"}).json()
    assert [item["position"] for item in body["positions"]] == [2, 7]


def test_independent_differs_from_joint(client):
    payload = {"sequence": WILD, "positions": [2, 7], "standard_only": True}
    joint = client.post("/api/mask-probabilities", json=payload).json()
    independent = client.post("/api/mask-probabilities", json={**payload, "independent": True}).json()
    assert joint["masking"] == "joint" and independent["masking"] == "independent"
    assert len(independent["positions"]) == 2
    # Mock logits are seeded by the variant's own tokens, so the two runs must not coincide.
    assert joint["positions"][0]["probabilities"] != independent["positions"][0]["probabilities"]


def test_renormalized_probabilities_sum_to_one(client):
    body = client.post("/api/mask-probabilities",
                       json={"sequence": WILD, "positions": [7], "standard_only": True,
                             "renormalize_selected": True}).json()
    position = body["positions"][0]
    assert position["renormalized"] is True
    assert abs(sum(position["probabilities"].values()) - 1.0) < 1e-9


def test_probabilities_are_not_renormalized_by_default(client):
    body = client.post("/api/mask-probabilities",
                       json={"sequence": WILD, "positions": [7], "standard_only": True}).json()
    position = body["positions"][0]
    assert position["renormalized"] is False
    assert sum(position["probabilities"].values()) <= position["selected_probability_mass"] + 1e-12


def test_fasta_text_is_read(client):
    body = client.post("/api/mask-probabilities",
                       json={"record_text": ">p1 test\nMKTF<mask>FVA\n"}).json()
    assert body["record"] == "p1"
    assert body["positions"][0]["position"] == 5


def test_bad_sequence_is_a_400_with_the_reason(client):
    response = client.post("/api/mask-probabilities", json={"sequence": "MK1TF<mask>"})
    assert response.status_code == 400
    assert "digits" in response.json()["error"]


def test_no_mask_is_a_400(client):
    response = client.post("/api/mask-probabilities", json={"sequence": WILD})
    assert response.status_code == 400
    assert "no position to predict" in response.json()["error"]


def test_unknown_field_is_refused(client):
    response = client.post("/api/mask-probabilities", json={"sequence": "MK<mask>", "modle": "esm2-35m"})
    assert response.status_code == 422


def test_top_out_of_range_is_refused(client):
    response = client.post("/api/mask-probabilities", json={"sequence": "MK<mask>", "top": 99})
    assert response.status_code == 422


def test_top_zero_gives_no_candidates(client):
    body = client.post("/api/mask-probabilities", json={"sequence": "MK<mask>", "top": 0}).json()
    assert body["positions"][0]["top_standard"] == []
