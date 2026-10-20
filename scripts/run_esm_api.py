#!/usr/bin/env python3
"""Start the ESM service.

    python scripts/run_esm_api.py --host 127.0.0.1 --port 8789

Settings other than the address come from the environment (esm_service/config.py); .env is read first when
python-dotenv is installed, so a native run behaves like the container's `env_file`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ESM service API.")
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind (0.0.0.0 in a container).")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--reload", action="store_true", help="Reload on code changes (development only).")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(REPO_ROOT / ".env")

    import uvicorn

    # The settings module reads the environment at import, so it must not be imported before load_dotenv
    # above; passing the app by name leaves importing it to uvicorn, after this point.
    uvicorn.run("esm_service.api:app", host=args.host, port=args.port, reload=args.reload,
                log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
