"""Logits over the model's vocabulary turned into the per-position answer.

Same arithmetic and same field names as the esm-directed-evolution skill's script, so a caller that read
its output keeps working.
"""

from __future__ import annotations

import math
from typing import Any

from esm_service.sequence import STANDARD_AA, InputError


def softmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    exps = [math.exp(value - maximum) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def position_result(
    mask_index: int,
    position: int,
    wild_type: str | None,
    logits: list[float],
    vocab: list[str],
    token_order: list[str],
    renormalize: bool,
    top: int,
) -> dict[str, Any]:
    full = dict(zip(vocab, softmax([float(value) for value in logits])))
    selected = {token: full[token] for token in token_order if token in full}
    mass = sum(selected.values())
    if renormalize:
        if mass <= 0:
            raise InputError("the selected tokens have no probability mass; cannot renormalize")
        selected = {token: probability / mass for token, probability in selected.items()}
    result: dict[str, Any] = {"mask_index": mask_index, "position": position}
    if wild_type:
        result["wild_type"] = wild_type
    result.update(probabilities=selected, selected_probability_mass=mass, renormalized=renormalize)
    ranked = sorted(STANDARD_AA, key=lambda aa: full.get(aa, 0.0), reverse=True)
    best = []
    for aa in ranked[:top]:
        item: dict[str, Any] = {"aa": aa, "probability": selected.get(aa, full.get(aa, 0.0))}
        if wild_type:
            item["mutation"] = f"{wild_type}{position}{aa}"
            if full.get(wild_type, 0) > 0 and full.get(aa, 0) > 0:
                # masked-marginal score: log p(mutant) - log p(wild type) at the masked position
                item["log_ratio_vs_wild_type"] = math.log(full[aa]) - math.log(full[wild_type])
        best.append(item)
    result["top_standard"] = best
    if wild_type and wild_type in full:
        result["wild_type_probability"] = selected.get(wild_type, full[wild_type])
        result["wild_type_rank"] = ranked.index(wild_type) + 1 if wild_type in ranked else None
    return result


def available_tokens(token_order: list[str], vocab: list[str]) -> list[str]:
    available = [token for token in token_order if token in vocab]
    if not available:
        raise InputError("none of the requested output tokens is in the model's vocabulary")
    return available
