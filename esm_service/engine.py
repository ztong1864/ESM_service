"""Loading ESM-2 once and scoring masked positions with it.

This is the whole point of the service: the skill's script loaded a 134 MB checkpoint on every question,
so each call paid for that. Here a model is loaded on first use and stays in memory, and requests queue on
a capacity limiter instead of each starting their own interpreter.

torch and fair-esm are imported inside the functions that need them, so the module (and the API's input
validation, and the mock mode) still work where they are not installed.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esm_service import config
from esm_service.models import BY_FILE_STEM, MODELS, resolve
from esm_service.sequence import MASK

# The vocabulary of every ESM-2 model, in its order (fair-esm Alphabet.all_toks); used by mock mode only.
ESM2_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
              "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]

MOCK_ENGINE = "mock (made-up logits; not a model result)"


class ModelNotAvailable(RuntimeError):
    """The requested model is not installed, or could not be loaded. The API answers 503 with this."""


@dataclass
class LoadedModel:
    name: str
    model: Any
    alphabet: Any
    vocab: list[str]
    device: str
    engine: str


def installed_models(folder: Path | None = None) -> list[str]:
    """Short names of the model files present, e.g. ["esm2-35m"]. An unknown file keeps its own stem."""
    directory = folder or config.MODELS_DIR
    if not directory.is_dir():
        return []
    stems = {path.stem for path in directory.glob("*.pt") if not path.stem.endswith("-contact-regression")}
    return sorted(BY_FILE_STEM.get(stem, stem) for stem in stems)


def _not_installed_message(name: str, folder: Path) -> str:
    installed = installed_models(folder)
    kind = ("ESM-1v is not installed here" if name.lower().startswith("esm1v")
            else f"model {name} is not installed")
    choice = f"Installed: {', '.join(installed)}; use one of those." if installed else "No model is installed."
    return (f"{kind} (looked in {folder}). {choice} An administrator can add it with "
            f"`python scripts/fetch_models.py {name}` on the server running this service.")


def resolve_device(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ModelNotAvailable(f"ESM_SERVICE_DEVICE must be auto, cpu or cuda, not {requested!r}")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - only without torch installed
        raise ModelNotAvailable(f"torch is not installed: {exc}") from exc
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise ModelNotAvailable("ESM_SERVICE_DEVICE=cuda but torch reports no usable GPU")
    return "cpu"


class Engine:
    """Holds the loaded models. One instance per process, created in api.py."""

    def __init__(self) -> None:
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.Lock()  # one load at a time: two requests must not load the same model twice
        self._device: str | None = None

    # --- model files ------------------------------------------------------------------------------------

    def model_path(self, name: str) -> Path:
        spec = resolve(name)
        stem = spec.file_stem if spec else name
        path = config.MODELS_DIR / f"{stem}.pt"
        if not path.is_file():
            raise ModelNotAvailable(_not_installed_message(name, config.MODELS_DIR))
        return path

    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(config.DEVICE)
        return self._device

    # --- loading ----------------------------------------------------------------------------------------

    def load(self, name: str) -> LoadedModel:
        """The loaded model for a short name, loading it on first use. Safe to call from several threads."""
        spec = resolve(name)
        key = spec.name if spec else name.strip().lower()
        with self._lock:
            existing = self._loaded.get(key)
            if existing is not None:
                self._loaded.move_to_end(key)
                return existing
            loaded = self._load_now(key, name)
            self._loaded[key] = loaded
            while len(self._loaded) > config.MAX_LOADED_MODELS:
                self._loaded.popitem(last=False)  # least recently used
            return loaded

    def _load_now(self, key: str, requested: str) -> LoadedModel:
        path = self.model_path(requested)
        device = self.device()
        try:
            import torch
            import esm
            from importlib.metadata import version
        except ImportError as exc:
            raise ModelNotAvailable(f"this service needs torch and fair-esm installed: {exc}") from exc

        if config.TORCH_THREADS:
            torch.set_num_threads(config.TORCH_THREADS)
        try:
            # fair-esm's own loader calls torch.load with torch's default weights_only=True since torch 2.6,
            # which cannot read these checkpoints; the files are Meta's, checked against their sha256 when
            # scripts/fetch_models.py downloaded them.
            model_data = torch.load(str(path), map_location="cpu", weights_only=False)
            regression_path = path.with_name(f"{path.stem}-contact-regression.pt")
            regression_data = (torch.load(str(regression_path), map_location="cpu", weights_only=False)
                               if regression_path.is_file() else None)
            model, alphabet = esm.pretrained.load_model_and_alphabet_core(path.stem, model_data, regression_data)
            model.eval()
            model.to(device)
        except Exception as exc:  # a corrupt or truncated checkpoint, or not enough memory
            raise ModelNotAvailable(f"cannot load {path.name}: {exc}") from exc

        label = "CUDA" if device == "cuda" else "CPU"
        return LoadedModel(
            name=key if key in MODELS else path.stem,
            model=model,
            alphabet=alphabet,
            vocab=list(alphabet.all_toks),
            device=device,
            engine=f"fair-esm {version('fair-esm')}, local {label}",
        )

    def loaded_names(self) -> list[str]:
        with self._lock:
            return list(self._loaded)

    # --- inference --------------------------------------------------------------------------------------

    def masked_logits(self, loaded: LoadedModel, variants: list[list[str]]) -> list[list[list[float]]]:
        """For each variant (tokens with masks), the logits over the vocabulary at each mask, in order."""
        import torch

        converter = loaded.alphabet.get_batch_converter()
        results: list[list[list[float]]] = []
        for start in range(0, len(variants), config.BATCH_SIZE):
            chunk = variants[start:start + config.BATCH_SIZE]
            _, _, batch = converter([(str(index), "".join(tokens)) for index, tokens in enumerate(chunk)])
            batch = batch.to(loaded.device)
            with torch.no_grad():
                logits = loaded.model(batch)["logits"]
            for row in range(len(chunk)):
                where = (batch[row] == loaded.alphabet.mask_idx).nonzero(as_tuple=True)[0].tolist()
                results.append([logits[row, index].tolist() for index in where])
        return results

    def score(self, variants: list[list[str]], model_name: str) -> tuple[list[str], str, str, list[list[list[float]]]]:
        """(vocabulary, engine label, model name, logits per variant). Blocking: call it in a worker thread."""
        if config.MOCK:
            return ESM2_VOCAB, MOCK_ENGINE, "mock", mock_logits(variants)
        loaded = self.load(model_name)
        return loaded.vocab, loaded.engine, loaded.name, self.masked_logits(loaded, variants)


def mock_logits(variants: list[list[str]]) -> list[list[list[float]]]:
    """Deterministic made-up logits, for checking the output format without torch or a model."""
    out = []
    for tokens in variants:
        rows = []
        for index, token in enumerate(tokens):
            if token == MASK:
                seed = hashlib.sha256(f"{''.join(tokens)}:{index}".encode()).digest()
                rows.append([(seed[i % len(seed)] - 128) / 40 for i in range(len(ESM2_VOCAB))])
        out.append(rows)
    return out
