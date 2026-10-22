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

FounDryClaw passes that address to each chat (`_claude_env` in `api/app/services/chat_runtime_service.py`),
where the `esm-directed-evolution` skill's `scripts/esm_mask_probabilities.py` reads it. That script is a
thin client: it keeps its command line, reads a FASTA or GenBank file, posts the sequence here and prints
the answer. Nothing on the FounDryClaw side holds torch, fair-esm or a model file any more.

## Running it for months

The service is meant to stay up: a model is loaded once and answers in about 0.05 s, which is the whole
point of it being a service rather than a script. [deploy/](deploy/) holds the unit files and the step-by-step
commands; in short, three ways to keep it running:

- **Docker** (`docker compose up -d`), the normal way: `restart: unless-stopped` brings it back after a
  crash or a reboot, the healthcheck polls `/api/health` every 30 s, and the log is capped at 3 x 10 MB.
  Note that Docker restarts a container that *exits*, not one the healthcheck marks unhealthy -- watch
  `/api/health` from outside too, where `status` is `degraded` when the model cannot be loaded.
  [deploy/esm-service-docker.service](deploy/esm-service-docker.service) hands the container to systemd
  when you want `systemctl` to manage it.
- **Linux without Docker**: [deploy/esm-service.service](deploy/esm-service.service), which runs the
  service from the host's own `.venv` with `Restart=always` and `MALLOC_ARENA_MAX=2` (glibc's per-thread
  malloc arenas make a long-running torch process creep in RSS).
- **Windows without Docker**: `start_esm_service.cmd` is a foreground window -- it dies with the session
  and restarts nothing, so it is for trying the service out, not for running it. Use Task Scheduler
  ("at startup", "run whether user is logged on or not") calling
  `.venv/Scripts/python.exe scripts/run_esm_api.py`, and redirect the output to a file you rotate
  yourself.

What to expect from a long run, measured over 400 mixed requests and 160 model loads:

- **Steady state is flat.** With esm2-35m resident, 400 requests of mixed length left the process
  oscillating between 390 and 490 MB with no upward trend.
- **Swapping models costs about 1 MB a time** in fragmentation (the models themselves are released; this
  is the allocator, not a leak). It only happens when requests ask for different models, because
  `ESM_SERVICE_MAX_LOADED_MODELS` is 1 by default. If two models are in regular use, raise it to 2 so
  nothing is unloaded (esm2-8m plus esm2-35m is about 0.7 GB together); otherwise a scheduled weekly
  restart costs 0.5 s of preload.
- **Never run more than one worker.** Each would load its own copy of the model; `run_esm_api.py` starts
  a single process on purpose, and concurrency is handled inside it by `ESM_SERVICE_MAX_CONCURRENCY`.
- **Check the model files now and then**: `python scripts/fetch_models.py --verify` reads what is on disk
  and compares it with the recorded checksums (a monthly scheduled task is enough). An existing file is
  never re-read while fetching, so this is what catches a truncated download.

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
| `ESM_SERVICE_MAX_RECORD_BYTES` | `2097152` | Largest accepted FASTA/GenBank text. |

## API

See [api.md](api.md). Three endpoints: `GET /api/health`, `GET /api/models` and
`POST /api/mask-probabilities`.

## Tests

The tests run in mock mode, so neither torch nor a model file is needed:

```bash
pip install -r requirements-dev.txt
pytest
```

They cover the sequence rules (masks, positions, translation, length, FASTA and GenBank records), the
API's answers and error codes, the resident engine's bookkeeping, and `tests/golden/`: the output the
`esm-directed-evolution` script printed before it became a client of this service. That answer, field for
field, is what this service must keep returning.

Two sets are left out of the default run:

```bash
pytest -m model                                     # loads a real checkpoint; needs torch and a model file
ESM_SKILL_SCRIPT=<path to esm_mask_probabilities.py> pytest -q tests/test_parity.py
```

The first checks the numbers against the ones the script printed for the same sequence. The second serves
this app on a free port and runs the skill's script against it, so the script's output and the API's answer
cannot drift apart.

## Layout

```
esm_service/        config, model registry, sequence rules, scoring, the resident engine, the FastAPI app
scripts/            run_esm_api.py (start it), fetch_models.py (download and verify model files)
deploy/             systemd units and the deployment steps
tests/              mock-mode tests, plus golden/ (the skill script's frozen output)
```

## Licence

This service is MIT (see [LICENSE](LICENSE)); `fair-esm` is MIT too. Meta's model weights keep their own
terms and are not redistributed here: `scripts/fetch_models.py` downloads them and git ignores them.
