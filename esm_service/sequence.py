"""Turning a request's sequence into the tokens the model scores.

Ported from the esm-directed-evolution skill's script so the rules and the messages stay the same; the
difference is that a FASTA or GenBank record arrives as text (the caller uploaded the file), not as a path
on this machine.
"""

from __future__ import annotations

import io

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_SPECIAL = ["X", "B", "Z", "U", "O", "-", ".", "<unk>", "<mask>"]
DEFAULT_TOKEN_ORDER = STANDARD_AA + DEFAULT_SPECIAL
RESIDUE_LETTERS = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
MASK = "<mask>"
MAX_RESIDUES = 1022  # ESM-2 was trained on at most 1022 residues (1024 tokens with the start and end tokens)

GENBANK_SUFFIXES = {".gb", ".gbk", ".genbank", ".gbff"}


class InputError(ValueError):
    """Something about the request's sequence is wrong; the API answers 400 with this message."""


class PreparedSequence:
    """What the model needs, plus what the answer reports back about the input."""

    def __init__(self, tokens: list[str], wild_types: dict[int, str], notes: list[str], record: str | None):
        self.tokens = tokens
        self.wild_types = wild_types
        self.notes = notes
        self.record = record

    @property
    def mask_positions(self) -> list[int]:
        """1-based residue numbers of the masked positions, in sequence order."""
        return [index + 1 for index, token in enumerate(self.tokens) if token == MASK]


def read_record(text: str, record_id: str | None, file_format: str = "fasta") -> tuple[str, str]:
    """(id, sequence) of one record in FASTA or GenBank text."""
    from Bio import SeqIO

    try:
        records = list(SeqIO.parse(io.StringIO(text), file_format))
    except ValueError as exc:
        raise InputError(f"cannot read the text as {file_format}: {exc}") from exc
    if not records:
        raise InputError(f"no {file_format} records in the text")
    if record_id:
        matches = [record for record in records if record.id == record_id]
        if not matches:
            listed = ", ".join(record.id for record in records[:20])
            raise InputError(f"no record {record_id!r} in the text; records: {listed}")
        record = matches[0]
    else:
        record = records[0]
    return record.id, str(record.seq)


def translate(dna: str) -> str:
    from Bio.Seq import Seq

    cleaned = "".join(dna.split()).upper().replace("U", "T")
    if set(cleaned) - set("ACGTN"):
        raise InputError("translate needs a DNA/RNA sequence (A, C, G, T/U, N)")
    if len(cleaned) % 3:
        raise InputError(f"the DNA is {len(cleaned)} nt long, not a whole number of codons; "
                         "give the coding sequence only")
    return str(Seq(cleaned).translate())


def tokens_of(sequence: str) -> list[str]:
    """One token per residue: letters and <mask>. Rejects anything else with a clear message."""
    tokens: list[str] = []
    index = 0
    while index < len(sequence):
        if sequence.startswith(MASK, index):
            tokens.append(MASK)
            index += len(MASK)
            continue
        letter = sequence[index]
        if letter.upper() in RESIDUE_LETTERS:
            tokens.append(letter.upper())
        elif letter == "*":
            raise InputError(f"stop symbol '*' inside the sequence (residue {len(tokens) + 1}); "
                             "give one protein chain")
        elif letter.isspace() or letter.isdigit() or letter in ">;":
            raise InputError("the sequence contains spaces, digits or a FASTA header; pass only residue "
                             "letters and <mask>, or send the file's text as record_text")
        else:
            raise InputError(f"unexpected character {letter!r} at residue {len(tokens) + 1}; residues are "
                             f"{''.join(sorted(RESIDUE_LETTERS))} and masks are written {MASK}")
        index += 1
    return tokens


def prepare(
    *,
    sequence: str | None,
    record_text: str | None,
    record_format: str,
    record: str | None,
    translate_dna: bool,
    positions: list[int] | None,
) -> PreparedSequence:
    notes: list[str] = []
    record_id = None
    if record_text:
        if sequence:
            raise InputError("give sequence or record_text, not both")
        record_id, raw = read_record(record_text, record, record_format)
    elif sequence:
        raw = sequence.strip()
    else:
        raise InputError("give sequence (with <mask>, or a wild type with positions) or record_text")

    if translate_dna:
        raw = translate(raw)
        notes.append("Translated the DNA sequence to protein (standard genetic code).")
    if raw.endswith("*"):
        raw = raw[:-1]
        notes.append("Removed the terminal '*' stop symbol.")
    if not translate_dna and len(raw) >= 30 and set(raw.upper()) <= set("ACGTUN"):
        notes.append("The sequence contains only A, C, G, T/U and N: if it is DNA, set translate.")

    tokens = tokens_of(raw)
    wild_types: dict[int, str] = {}
    if positions:
        if MASK in tokens:
            raise InputError("use either <mask> in the sequence or positions on a wild-type sequence, not both")
        for position in positions:
            if not 1 <= position <= len(tokens):
                raise InputError(f"position {position} is outside the sequence (1-{len(tokens)})")
            wild_types[position] = tokens[position - 1]
            tokens[position - 1] = MASK
    if MASK not in tokens:
        raise InputError("no position to predict: put <mask> in the sequence or pass positions")
    if len(tokens) > MAX_RESIDUES:
        raise InputError(f"the sequence has {len(tokens)} residues; ESM-2 handles at most {MAX_RESIDUES}. "
                         "Pass the domain around the positions of interest")
    return PreparedSequence(tokens, wild_types, notes, record_id)


def variants_for(prepared: PreparedSequence, independent: bool) -> list[list[str]]:
    """The token lists to run: one (all masks together), or one per position with the others restored."""
    if not independent:
        return [prepared.tokens]
    if not prepared.wild_types:
        raise InputError("independent needs positions: the other masked positions need their "
                         "wild-type residues")
    variants = []
    for position in prepared.mask_positions:
        variant = list(prepared.tokens)
        for other in prepared.mask_positions:
            if other != position:
                variant[other - 1] = prepared.wild_types[other]
        variants.append(variant)
    return variants


def token_order_for(tokens: list[str] | None, standard_only: bool) -> list[str]:
    if tokens:
        chosen = [token.strip() for token in tokens if token and token.strip()]
        if not chosen:
            raise InputError("tokens was given but no tokens were parsed")
        return chosen
    return list(STANDARD_AA) if standard_only else list(DEFAULT_TOKEN_ORDER)
