"""API behaviour in mock mode: the shape of the answer, and what each bad request returns."""

from __future__ import annotations

from esm_service import config

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


GENBANK = """LOCUS       TESTSEQ                   10 aa            linear   UNK 01-JAN-2026
DEFINITION  a test protein.
ACCESSION   TESTSEQ
ORIGIN
        1 mktffvalvl
//
"""


def test_genbank_text_is_read(client):
    body = client.post("/api/mask-probabilities",
                       json={"record_text": GENBANK, "record_format": "genbank", "positions": [3]}).json()
    assert body["record"] == "TESTSEQ"
    assert body["sequence"] == "MK<mask>FFVALVL"
    assert body["positions"][0]["wild_type"] == "T"


def test_genbank_text_read_as_fasta_is_a_400(client):
    response = client.post("/api/mask-probabilities", json={"record_text": GENBANK, "positions": [3]})
    assert response.status_code == 400
    # Biopython's own complaint comes through, wrapped in a sentence that names the format tried.
    assert response.json()["error"].startswith("cannot read the text as fasta:")


def test_exactly_the_longest_sequence_is_accepted(client):
    body = client.post("/api/mask-probabilities", json={"sequence": "A" * 1021 + "<mask>"}).json()
    assert body["positions"][0]["position"] == 1022


def test_one_residue_too_many_is_refused(client):
    response = client.post("/api/mask-probabilities", json={"sequence": "A" * 1022 + "<mask>"})
    assert response.status_code == 400
    assert "at most 1022" in response.json()["error"]


def test_custom_tokens_keep_their_order_and_drop_unknown_ones(client):
    body = client.post("/api/mask-probabilities",
                       json={"sequence": "MK<mask>", "tokens": ["W", "A", "Zzz"]}).json()
    assert body["token_order"] == ["W", "A"]
    assert list(body["positions"][0]["probabilities"]) == ["W", "A"]


def test_tokens_accept_a_comma_string(client):
    body = client.post("/api/mask-probabilities", json={"sequence": "MK<mask>", "tokens": "W,A"}).json()
    assert body["token_order"] == ["W", "A"]


def test_no_usable_token_is_a_400(client):
    response = client.post("/api/mask-probabilities", json={"sequence": "MK<mask>", "tokens": ["Zzz"]})
    assert response.status_code == 400
    assert "vocabulary" in response.json()["error"]


def test_a_non_standard_wild_type_has_no_rank(client):
    # X is in the model's vocabulary but not among the 20 ranked amino acids.
    body = client.post("/api/mask-probabilities", json={"sequence": "MKXTF", "positions": [3]}).json()
    position = body["positions"][0]
    assert position["wild_type"] == "X"
    assert position["wild_type_rank"] is None
    assert "wild_type_probability" in position


def test_translated_dna_reports_the_note(client):
    body = client.post("/api/mask-probabilities",
                       json={"sequence": "ATGAAAACC", "translate": True, "positions": [2]}).json()
    assert body["sequence"] == "M<mask>T"
    assert any("Translated" in note for note in body["notes"])


def test_a_missing_model_is_a_503_saying_what_is_installed(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MOCK", False)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    response = client.post("/api/mask-probabilities",
                           json={"sequence": "MK<mask>", "model": "esm2-650m"})
    assert response.status_code == 503  # answered before torch is even imported
    assert "No model is installed" in response.json()["error"]
    assert "fetch_models.py" in response.json()["error"]


def test_a_name_that_is_not_a_model_is_refused(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MOCK", False)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    response = client.post("/api/mask-probabilities",
                           json={"sequence": "MK<mask>", "model": "../../elsewhere/model"})
    assert response.status_code == 503
    assert "is not a model name" in response.json()["error"]


def test_an_oversized_record_is_refused(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_RECORD_BYTES", 100)
    response = client.post("/api/mask-probabilities",
                           json={"record_text": ">p\n" + "A" * 200 + "<mask>\n"})
    assert response.status_code == 422


def test_mock_mode_says_so_in_every_answer(client):
    body = client.post("/api/mask-probabilities", json={"sequence": "MK<mask>"}).json()
    assert body["model"] == "mock"
    assert "not a model result" in body["engine"]
