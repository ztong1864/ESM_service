"""The ESM-2 models this service knows, and where their files come from.

`sha256` is the checksum of Meta's published file. Only checksums that were verified against a real
download are recorded here; `None` means the file is fetched without a content check (scripts/fetch_models.py
says so when it downloads one). Add the checksum here once a download has been verified.
"""

from __future__ import annotations

from dataclasses import dataclass

BASE_URL = "https://dl.fbaipublicfiles.com/fair-esm"


@dataclass(frozen=True)
class ModelSpec:
    """One ESM-2 model: the short name used in requests and the file fair-esm loads."""

    name: str
    file_stem: str
    parameters: str
    layers: int
    embedding_dim: int
    sha256: str | None = None
    regression_sha256: str | None = None

    @property
    def weights_url(self) -> str:
        return f"{BASE_URL}/models/{self.file_stem}.pt"

    @property
    def regression_url(self) -> str:
        return f"{BASE_URL}/regression/{self.file_stem}-contact-regression.pt"


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec("esm2-8m", "esm2_t6_8M_UR50D", "8M", 6, 320),
        ModelSpec(
            "esm2-35m",
            "esm2_t12_35M_UR50D",
            "35M",
            12,
            480,
            sha256="7f21e80e61d16a71735163ef555d3009afb0c98da74c48e29df08606973cc55e",
            regression_sha256="16641e05d830d0ce863dd152dbb8c2f3ddfa3c3ec2a66080152c8abad01d8585",
        ),
        ModelSpec("esm2-150m", "esm2_t30_150M_UR50D", "150M", 30, 640),
        ModelSpec("esm2-650m", "esm2_t33_650M_UR50D", "650M", 33, 1280),
        ModelSpec("esm2-3b", "esm2_t36_3B_UR50D", "3B", 36, 2560),
        ModelSpec("esm2-15b", "esm2_t48_15B_UR50D", "15B", 48, 5120),
    )
}

# A file stem back to the short name, so an installed file is reported the way requests name it.
BY_FILE_STEM: dict[str, str] = {spec.file_stem: name for name, spec in MODELS.items()}


def resolve(name: str) -> ModelSpec | None:
    """The spec for a short name (esm2-35m) or a file stem (esm2_t12_35M_UR50D); None if unknown."""
    key = name.strip().lower()
    if key in MODELS:
        return MODELS[key]
    stem_match = BY_FILE_STEM.get(name.strip())
    return MODELS[stem_match] if stem_match else None
