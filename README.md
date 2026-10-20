# ESM service

A resident ESM-2 protein language model behind a small HTTP API: it scores the amino acids at masked
positions of a protein, for directed-evolution candidate ranking.

It replaces the per-call script of FounDryClaw's `esm-directed-evolution` skill, which loaded a 134 MB
checkpoint on every question. Here the model is loaded once and stays in memory, so a request is answered
in tens of milliseconds instead of tens of seconds. The answer's JSON is the same as that script's, field
for field, and was checked to the last digit against it.

Nothing is sent anywhere: the model runs on this machine's CPU (or GPU) and no key is involved.

## Quick start (Windows, no Docker)

Double-click `start_esm_service.cmd`, or:

```powershell
.\run_local.ps1
```

It creates `.venv`, installs the tested versions, downloads the default model into `models\` and starts
the service on <http://127.0.0.1:8789>. The download site is slow and sometimes drops the connection;
running the script again continues where it stopped.

Check it:

```bash
curl http://127.0.0.1:8789/api/health
```

## Quick start (Docker)

```bash
cp .env.example .env
python scripts/fetch_models.py          # once, into ./models
docker compose up -d --build
```

The model files are mounted read-only rather than baked into the image. Nothing else is shared with the
host, because sequences arrive over HTTP and results go back in the reply.

## Models

```bash
python scripts/fetch_models.py --list          # what is known, and what is here
python scripts/fetch_models.py esm2-650m       # add one
```

`esm2-35m` is the default: the smallest useful ESM-2 model, so its rankings are coarse. `esm2-150m` and
`esm2-650m` are better and heavier; `esm2-650m` needs about 2.6 GB of memory. The model's checksum is
verified when one is recorded in `esm_service/models.py`; the command says so when it downloads a file
without one.

A GPU is used when there is one (`ESM_SERVICE_DEVICE=auto`), but it is not needed: esm2-35m answers a
20-residue request on a CPU in well under a second.

## Using it from FounDryClaw

Set the service's address in FounDryClaw's `api/.env`:

```
ESM_SERVICE_BASE_URL=http://127.0.0.1:8789
```

and call it from the `esm-protein` tool package (`tools/Device/iBioFoundry/.claude/tools/`), the same way
`trace-collab` calls TRACE Lab. The chat side is a tool, not a skill script, so the model's calls are one
HTTP request each, with no Python environment or model files on the FounDryClaw side.

## Settings

Every setting is an environment variable with a working default; `.env.example` lists them, and both
`run_local.ps1` and docker compose read `.env`. The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `ESM_MODELS_DIR` | `./models` | Where the `.pt` files are. |
| `ESM_SERVICE_DEFAULT_MODEL` | `esm2-35m` | Loaded at startup, used when a request names no model. |
| `ESM_SERVICE_DEVICE` | `auto` | `auto`, `cpu` or `cuda`. |
| `ESM_SERVICE_MAX_LOADED_MODELS` | `1` | How many models stay in memory. |
| `ESM_SERVICE_MAX_CONCURRENCY` | `2` | How many requests are scored at once. |
| `ESM_SERVICE_MOCK` | `false` | Made-up numbers, no model: for testing the API without torch. |

## API

See [api.md](api.md). Three endpoints: `GET /api/health`, `GET /api/models` and
`POST /api/mask-probabilities`.

## Tests

The tests run in mock mode, so neither torch nor a model file is needed:

```bash
pip install -r requirements-dev.txt
pytest
```

They cover the sequence rules (masks, positions, translation, length, FASTA records) and the API's
answers and error codes. To check against a real model, start the service and compare one request with
the `esm-directed-evolution` script's output for the same sequence: the numbers must be identical.

## Layout

```
esm_service/        config, model registry, sequence rules, scoring, the resident engine, the FastAPI app
scripts/            run_esm_api.py (start it), fetch_models.py (download model files)
tests/              mock-mode tests
```
