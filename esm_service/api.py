"""The HTTP API.

Three endpoints: health, the installed/loaded models, and the masked-position scoring that does the work.
Scoring is blocking (torch), so it runs in a worker thread behind a capacity limiter; the event loop stays
free to answer health checks while a long sequence is running.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import anyio
import anyio.to_thread
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from esm_service import config
from esm_service.engine import Engine, ModelNotAvailable, installed_models
from esm_service.models import MODELS
from esm_service.schemas import MaskProbabilitiesRequest
from esm_service.scoring import available_tokens, position_result
from esm_service.sequence import InputError, prepare, token_order_for, variants_for

logger = logging.getLogger("esm_service")

VERSION = "0.1.0"

engine = Engine()
_limiter: anyio.CapacityLimiter | None = None
_startup: dict[str, Any] = {"preloaded": None, "error": None}


def limiter() -> anyio.CapacityLimiter:
    global _limiter
    if _limiter is None:
        _limiter = anyio.CapacityLimiter(config.MAX_CONCURRENCY)
    return _limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.MOCK:
        logger.warning("ESM_SERVICE_MOCK is on: every answer is made up, not a model result")
    elif config.PRELOAD:
        # A failed preload must not stop the service: /api/health then reports why, which is how an
        # administrator sees a missing model file without reading the logs.
        try:
            loaded = await anyio.to_thread.run_sync(engine.load, config.DEFAULT_MODEL, limiter=limiter())
            _startup["preloaded"] = loaded.name
            logger.info("loaded %s on %s", loaded.name, loaded.device)
        except ModelNotAvailable as exc:
            _startup["error"] = str(exc)
            logger.error("could not load %s at startup: %s", config.DEFAULT_MODEL, exc)
    yield


app = FastAPI(
    title="ESM service",
    version=VERSION,
    summary="Amino-acid probabilities at masked protein positions from a resident ESM-2 model.",
    lifespan=lifespan,
)


@app.exception_handler(InputError)
async def _input_error(_request, exc: InputError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(ModelNotAvailable)
async def _model_error(_request, exc: ModelNotAvailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Reachable, and what it can serve. Never fails because a model is missing: that is in `error`."""
    body: dict[str, Any] = {
        "status": "ok" if (config.MOCK or not _startup["error"]) else "degraded",
        "version": VERSION,
        "mock": config.MOCK,
        "default_model": config.DEFAULT_MODEL,
        "models_dir": str(config.MODELS_DIR),
        "installed": installed_models(),
        "loaded": engine.loaded_names(),
    }
    if not config.MOCK:
        try:
            body["device"] = engine.device()
        except ModelNotAvailable as exc:
            body["status"] = "degraded"
            body["device"] = None
            body.setdefault("error", str(exc))
    if _startup["error"]:
        body["error"] = _startup["error"]
    return body


@app.get("/api/models")
async def models() -> dict[str, Any]:
    installed = set(installed_models())
    return {
        "default": config.DEFAULT_MODEL,
        "loaded": engine.loaded_names(),
        "models": [
            {
                "name": spec.name,
                "parameters": spec.parameters,
                "layers": spec.layers,
                "embedding_dim": spec.embedding_dim,
                "file": f"{spec.file_stem}.pt",
                "installed": spec.name in installed,
            }
            for spec in MODELS.values()
        ],
    }


@app.post("/api/mask-probabilities")
async def mask_probabilities(request: MaskProbabilitiesRequest) -> dict[str, Any]:
    started = time.monotonic()
    prepared = prepare(
        sequence=request.sequence,
        record_text=request.record_text,
        record_format=request.record_format,
        record=request.record,
        translate_dna=request.translate,
        positions=request.positions,
    )
    variants = variants_for(prepared, request.independent)
    requested_tokens = token_order_for(request.tokens, request.standard_only)
    model_name = (request.model or config.DEFAULT_MODEL).strip()

    vocab, engine_label, resolved_name, per_variant = await anyio.to_thread.run_sync(
        engine.score, variants, model_name, limiter=limiter()
    )

    logits_by_mask = ([rows[0] for rows in per_variant] if request.independent else per_variant[0])
    if len(logits_by_mask) != len(prepared.mask_positions):  # a model whose masking behaved unexpectedly
        raise HTTPException(
            status_code=500,
            detail=f"the model returned {len(logits_by_mask)} masked positions, expected "
                   f"{len(prepared.mask_positions)}",
        )

    token_order = available_tokens(requested_tokens, vocab)
    positions = [
        position_result(index, position, prepared.wild_types.get(position), logits, vocab, token_order,
                        request.renormalize_selected, request.top)
        for index, (position, logits) in enumerate(zip(prepared.mask_positions, logits_by_mask))
    ]
    answer: dict[str, Any] = {
        "model": resolved_name,
        "engine": engine_label,
        "sequence": "".join(prepared.tokens),
        "masking": "independent" if request.independent else "joint",
        "token_order": token_order,
        "positions": positions,
    }
    if prepared.record:
        answer["record"] = prepared.record
    if prepared.notes:
        answer["notes"] = prepared.notes
    answer["elapsed_s"] = round(time.monotonic() - started, 3)
    return answer
