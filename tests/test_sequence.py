from __future__ import annotations

import pytest

from esm_service.sequence import (
    InputError,
    MAX_RESIDUES,
    prepare,
    token_order_for,
    tokens_of,
    variants_for,
)

WILD = "MKTFFVALVLLLSGALAAPVA"


def prepared(**kwargs):
    base = dict(sequence=None, record_text=None, record_format="fasta", record=None,
                translate_dna=False, positions=None)
    return prepare(**{**base, **kwargs})


def test_mask_token_is_one_residue():
    tokens = tokens_of("MK<mask>TF")
    assert tokens == ["M", "K", "<mask>", "T", "F"]


def test_positions_record_the_wild_type_and_mask_it():
    result = prepared(sequence=WILD, positions=[7, 2])
    assert result.wild_types == {7: "A", 2: "K"}
    assert result.mask_positions == [2, 7]  # sequence order, not the order given
    assert result.tokens[1] == "<mask>" and result.tokens[6] == "<mask>"


def test_mask_and_positions_together_are_refused():
    with pytest.raises(InputError, match="either <mask>"):
        prepared(sequence="MK<mask>TF", positions=[2])


def test_no_mask_at_all_is_refused():
    with pytest.raises(InputError, match="no position to predict"):
        prepared(sequence=WILD)


def test_position_outside_the_sequence():
    with pytest.raises(InputError, match="outside the sequence"):
        prepared(sequence=WILD, positions=[999])


def test_internal_stop_is_refused_but_a_terminal_one_is_dropped():
    with pytest.raises(InputError, match="stop symbol"):
        prepared(sequence="MK*TF<mask>")
    result = prepared(sequence="MKTF<mask>*")
    assert result.tokens[-1] == "<mask>"
    assert any("terminal" in note for note in result.notes)


def test_header_or_spaces_point_at_record_text():
    with pytest.raises(InputError, match="record_text"):
        prepared(sequence=">sp|P1|X\nMKT<mask>F")


def test_translate_needs_whole_codons():
    with pytest.raises(InputError, match="whole number of codons"):
        prepared(sequence="ATGAA", translate_dna=True)


def test_translate_then_mask_by_position():
    result = prepared(sequence="ATGAAAACC", translate_dna=True, positions=[2])
    assert result.wild_types == {2: "K"}
    assert "".join(result.tokens) == "M<mask>T"


def test_dna_looking_wild_type_gets_a_note():
    # Only the wild-type path can hit this: a sequence holding <mask> is not all ACGTUN anyway.
    result = prepared(sequence="ACGT" * 10, positions=[3])
    assert any("set translate" in note for note in result.notes)


def test_too_long_is_refused():
    with pytest.raises(InputError, match="at most"):
        prepared(sequence="A" * (MAX_RESIDUES + 1) + "<mask>")


def test_fasta_record_is_picked_by_id():
    text = ">one\nMKTF\n>two\nMKT<mask>F\n"
    result = prepared(record_text=text, record="two")
    assert result.record == "two"
    assert result.mask_positions == [4]


def test_unknown_fasta_record_lists_what_is_there():
    with pytest.raises(InputError, match="records: one, two"):
        prepared(record_text=">one\nMKTF\n>two\nMKTF\n", record="three")


def test_independent_variants_restore_the_other_positions():
    result = prepared(sequence=WILD, positions=[2, 7])
    variants = variants_for(result, independent=True)
    assert len(variants) == 2
    assert variants[0][1] == "<mask>" and variants[0][6] == "A"
    assert variants[1][1] == "K" and variants[1][6] == "<mask>"


def test_independent_without_positions_is_refused():
    result = prepared(sequence="MK<mask>TF")
    with pytest.raises(InputError, match="independent needs positions"):
        variants_for(result, independent=True)


def test_token_order_choices():
    assert len(token_order_for(None, standard_only=True)) == 20
    assert "<mask>" in token_order_for(None, standard_only=False)
    assert token_order_for(["A", " C "], standard_only=False) == ["A", "C"]
