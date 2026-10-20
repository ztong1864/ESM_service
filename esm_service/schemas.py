"""Request bodies. The answer is the same JSON the esm-directed-evolution script printed, so it is built
as a plain dict rather than a model that would have to be kept in step with it."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from esm_service import config


def _as_list(value: Any, what: str) -> list[str] | None:
    """Accepts a list, or one string with commas/semicolons, so curl and a form field both work."""
    if value is None or isinstance(value, list):
        return value
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        return parts or None
    raise ValueError(f"{what} must be a list or a comma-separated string")


class MaskProbabilitiesRequest(BaseModel):
    # "model" is a field name here; without this pydantic warns about its own model_ namespace.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    sequence: str | None = Field(
        default=None,
        description="Protein sequence with one or more <mask>, or a wild type together with positions.",
    )
    record_text: str | None = Field(
        default=None,
        description="Contents of a FASTA or GenBank file, instead of sequence.",
    )
    record_format: Literal["fasta", "genbank"] = "fasta"
    record: str | None = Field(default=None, description="Record id in record_text (default: the first).")
    translate: bool = Field(default=False, description="The input is a DNA/RNA coding sequence: translate it first.")
    positions: list[int] | None = Field(
        default=None,
        description="1-based residues of the wild type to mask, e.g. [45, 120]. A string like \"45,120\" also works.",
    )
    model: str | None = Field(default=None, description=f"Installed model (default: {config.DEFAULT_MODEL}).")
    independent: bool = Field(
        default=False,
        description="With positions: score each position with only that one masked. Default: all together.",
    )
    tokens: list[str] | None = Field(default=None, description="Output tokens (default: the 20 amino acids plus ESM's special tokens).")
    standard_only: bool = Field(default=False, description="Output only the 20 standard amino acids.")
    renormalize_selected: bool = Field(default=False, description="Renormalize over the output tokens so they sum to 1.")
    top: int = Field(default=5, ge=0, le=20, description="How many best standard amino acids to list per position.")

    @field_validator("positions", mode="before")
    @classmethod
    def _positions(cls, value: Any) -> Any:
        parts = _as_list(value, "positions")
        if parts is None:
            return None
        try:
            return [int(part) for part in parts]
        except (TypeError, ValueError) as exc:
            raise ValueError("positions must be 1-based residue numbers such as [45, 120]") from exc

    @field_validator("tokens", mode="before")
    @classmethod
    def _tokens(cls, value: Any) -> Any:
        return _as_list(value, "tokens")

    @field_validator("record_text")
    @classmethod
    def _record_size(cls, value: str | None) -> str | None:
        if value is not None and len(value.encode("utf-8")) > config.MAX_RECORD_BYTES:
            raise ValueError(f"record_text is larger than {config.MAX_RECORD_BYTES} bytes")
        return value
