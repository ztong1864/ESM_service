"""Settings, all from environment variables (docker-compose passes them with `env_file`, run_local.ps1
reads the same .env). Read once at import; the service is restarted to change them."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, "").strip() or default))
    except ValueError:
        return default


# Where the .pt files live. scripts/fetch_models.py fills this folder.
MODELS_DIR = Path(os.getenv("ESM_MODELS_DIR", "").strip() or REPO_ROOT / "models").resolve()

# The model loaded at startup and used when a request does not name one.
DEFAULT_MODEL = os.getenv("ESM_SERVICE_DEFAULT_MODEL", "").strip().lower() or "esm2-35m"

# cpu, cuda, or auto (cuda when torch reports a usable GPU, else cpu).
DEVICE = os.getenv("ESM_SERVICE_DEVICE", "").strip().lower() or "auto"

# Load DEFAULT_MODEL when the service starts, so the first request does not pay for it.
PRELOAD = _flag("ESM_SERVICE_PRELOAD", True)

# How many models stay in memory. A 650M model is about 2.6 GB, so raise this only with the RAM for it;
# the least recently used one is dropped when the limit is passed.
MAX_LOADED_MODELS = _int("ESM_SERVICE_MAX_LOADED_MODELS", 1)

# Inference runs in a worker thread; this caps how many run at once (each one holds the model's memory
# for its activations). Requests above the limit wait.
MAX_CONCURRENCY = _int("ESM_SERVICE_MAX_CONCURRENCY", 2)

# Sequences per forward pass when several positions are scored independently.
BATCH_SIZE = _int("ESM_SERVICE_BATCH_SIZE", 8)

# torch's intra-op threads. 0 leaves torch's own default (usually every core), which is wrong on a shared
# machine: a single large request would take the whole CPU.
TORCH_THREADS = _int("ESM_SERVICE_TORCH_THREADS", 0, minimum=0)

# Made-up logits instead of a model: lets the API, its validation and its output format be tested
# without torch or a model file. Never serve real work with this on; every response says it is mock.
MOCK = _flag("ESM_SERVICE_MOCK", False)

# Largest accepted FASTA/GenBank text, in bytes. A protein record is a few KB; this only stops a whole
# genome being posted.
MAX_RECORD_BYTES = _int("ESM_SERVICE_MAX_RECORD_BYTES", 2 * 1024 * 1024)
