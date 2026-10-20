# ESM service API

Base URL: `http://127.0.0.1:8789` by default. All bodies are JSON.

Errors carry one readable sentence: `{"error": "..."}` with status **400** for anything wrong about the
input, **503** when the model is not installed or cannot be loaded, and **422** for a malformed body
(FastAPI's own validation, e.g. an unknown field or `top` out of range).

---

## `GET /api/health`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "mock": false,
  "default_model": "esm2-35m",
  "models_dir": "/app/models",
  "installed": ["esm2-35m"],
  "loaded": ["esm2-35m"],
  "device": "cpu"
}
```

`status` is `degraded` when the default model could not be loaded; `error` then says why. The endpoint
itself still answers, so a health check tells an administrator what is wrong instead of only timing out.

## `GET /api/models`

Every model this service knows, with `installed` saying whether its file is present, plus `default` and the
models currently in memory (`loaded`).

## `POST /api/mask-probabilities`

Scores the masked positions of one protein chain.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `sequence` | string | — | Protein sequence with one or more `<mask>`, or a wild type together with `positions`. |
| `record_text` | string | — | Contents of a FASTA or GenBank file, instead of `sequence`. |
| `record_format` | `fasta` \| `genbank` | `fasta` | How to read `record_text`. |
| `record` | string | first record | Which record in `record_text`. |
| `translate` | bool | `false` | The input is a DNA/RNA coding sequence: translate it first. |
| `positions` | list of int | — | 1-based residues of the wild type to mask. `"45,120"` also works. |
| `model` | string | `esm2-35m` | An installed model. |
| `independent` | bool | `false` | With `positions`: score each position with only that one masked. |
| `tokens` | list of string | — | Output tokens. Default: the 20 amino acids plus ESM's special tokens. |
| `standard_only` | bool | `false` | Output only the 20 standard amino acids. |
| `renormalize_selected` | bool | `false` | Renormalize over the output tokens so they sum to 1. |
| `top` | int 0–20 | `5` | How many best standard amino acids to list per position. |

Give `sequence` or `record_text`, not both; and `<mask>` or `positions`, not both. At most 1022 residues.

```bash
curl -s -X POST http://127.0.0.1:8789/api/mask-probabilities \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTFFVALVLLLSGALAAPVA", "positions": [7], "standard_only": true, "top": 3}'
```

```json
{
  "model": "esm2-35m",
  "engine": "fair-esm 2.0.0, local CPU",
  "sequence": "MKTFFV<mask>LVLLLSGALAAPVA",
  "masking": "joint",
  "token_order": ["A", "C", "D", "..."],
  "positions": [
    {
      "mask_index": 0,
      "position": 7,
      "wild_type": "A",
      "probabilities": {"A": 0.2281482524887472, "C": 0.02541706843957601, "...": 0.0},
      "selected_probability_mass": 0.9999804848175206,
      "renormalized": false,
      "top_standard": [
        {"aa": "A", "probability": 0.2281, "mutation": "A7A", "log_ratio_vs_wild_type": 0.0},
        {"aa": "L", "probability": 0.1932, "mutation": "A7L", "log_ratio_vs_wild_type": -0.166}
      ],
      "wild_type_probability": 0.2281482524887472,
      "wild_type_rank": 1
    }
  ],
  "elapsed_s": 0.041
}
```

`record` is present when the sequence came from `record_text`, and `notes` when the input was changed
(translated, a terminal `*` removed) or looks like DNA.

**Reading the numbers.** `probabilities` is a softmax over the model's whole vocabulary, then filtered to
`token_order`, so it may sum to slightly less than 1 (`selected_probability_mass`) unless
`renormalize_selected` was set. `log_ratio_vs_wild_type` is log p(mutant) − log p(wild type) at that
masked position: a ranking aid, not a measurement of stability or activity.

These fields are exactly those of the `esm-directed-evolution` skill's script, apart from `elapsed_s`,
which the script did not have.
