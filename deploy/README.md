# Deploying the ESM service

Docker is the normal way to run this. The two unit files here are for systemd hosts: one owns the
container, the other runs the service directly without Docker.

| File | When |
|---|---|
| `esm-service-docker.service` | Docker, and you want `systemctl` to manage it. Optional — see below. |
| `esm-service.service` | No Docker: the service runs from the host's own `.venv`. |

## Docker (recommended)

```bash
git clone <this repository> /opt/esm-service
cd /opt/esm-service
cp .env.example .env                    # every setting has a working default; an empty file is fine
python3 scripts/fetch_models.py         # once, into ./models (needs only the standard library)
docker compose up -d --build
curl -s http://127.0.0.1:8789/api/health
```

`restart: unless-stopped` brings the container back after a crash and after a reboot, provided the Docker
daemon itself starts at boot (`systemctl enable docker`). That is why the systemd unit is optional. Add it
when you want `systemctl restart esm-service` to be the one way to manage the service:

```bash
install -m 644 deploy/esm-service-docker.service /etc/systemd/system/esm-service.service
systemctl daemon-reload
systemctl enable --now esm-service
```

Two things to know:

- **An unhealthy container is not restarted.** Docker restarts a container that *exits*; the healthcheck
  only marks it unhealthy. If the model files disappear, the container keeps running while
  `/api/health` reports `status: degraded`. Watch the endpoint, not just `docker compose ps`.
- **Model files are mounted read-only.** Fetch new ones on the host (`python3 scripts/fetch_models.py
  esm2-150m`) and restart. Checking them from inside the container works, because `--verify` only reads:
  `docker compose exec esm-service python scripts/fetch_models.py --verify`.

Logs: `docker compose logs -f`, capped at 3 x 10 MB by `docker-compose.yml`.

## Without Docker (systemd)

```bash
useradd --system --home /opt/esm-service --shell /usr/sbin/nologin esm
git clone <this repository> /opt/esm-service
cd /opt/esm-service
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
cp .env.example .env
.venv/bin/python scripts/fetch_models.py
chown -R esm:esm /opt/esm-service

install -m 644 deploy/esm-service.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now esm-service
systemctl status esm-service
curl -s http://127.0.0.1:8789/api/health
```

The `--extra-index-url` is not optional: without it pip installs the CUDA build of torch and pulls in
several GB of NVIDIA libraries that a CPU server never uses.

Logs: `journalctl -u esm-service -f`.

## Pointing FounDryClaw at it

In FounDryClaw's `api/.env`, then restart the backend:

```
ESM_SERVICE_BASE_URL=http://127.0.0.1:8789
```

The backend passes that address to every chat, where the `esm-directed-evolution` script reads it. If the
service runs on another host, use that host's address and bind the service accordingly — the port is
published on `127.0.0.1` only, both in compose and in the unit above.

## Checking the model files on a schedule

An existing file is never re-read while fetching, so a download that was truncated months ago stays
unnoticed until a request hits it. A monthly check:

```ini
# /etc/systemd/system/esm-verify.service
[Unit]
Description=Check the ESM model files against their checksums

[Service]
Type=oneshot
WorkingDirectory=/opt/esm-service
ExecStart=/opt/esm-service/.venv/bin/python scripts/fetch_models.py --verify
```

```ini
# /etc/systemd/system/esm-verify.timer
[Unit]
Description=Monthly ESM model file check

[Timer]
OnCalendar=monthly
Persistent=true

[Install]
WantedBy=timers.target
```

`systemctl enable --now esm-verify.timer`. It exits non-zero on a mismatch, so a failed run shows up in
`systemctl list-units --failed`. With Docker, point `ExecStart` at
`/usr/bin/docker compose exec -T esm-service python scripts/fetch_models.py --verify` instead.

## Restarting on a schedule (optional)

Repeatedly switching between models leaves about 1 MB of allocator fragmentation behind each time. It only
happens when requests name different models, and raising `ESM_SERVICE_MAX_LOADED_MODELS` so nothing is
unloaded avoids it entirely. If you would rather just restart: `systemctl edit --force --full
esm-restart.timer` with `OnCalendar=weekly`, calling `systemctl restart esm-service`. Preloading
esm2-35m again costs about half a second.

## Windows

`start_esm_service.cmd` (double-click) runs `run_local.ps1`, which creates or completes `.venv`, checks the
model files and starts the service in that window. It is for trying the service out and for testing
locally: it stops when the window closes and restarts nothing. For a Windows host that must keep it
running, use Task Scheduler at startup, with "run whether user is logged on or not", calling
`.venv/Scripts/python.exe scripts/run_esm_api.py`.
