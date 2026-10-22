#!/usr/bin/env python3
"""Download ESM-2 model files into the service's models folder.

    python scripts/fetch_models.py              # the default model (ESM_SERVICE_DEFAULT_MODEL)
    python scripts/fetch_models.py esm2-650m    # one or more named models
    python scripts/fetch_models.py --list       # what is known and what is already here
    python scripts/fetch_models.py --verify     # check the files already here against their checksums

A file that is already there is kept (delete it to fetch it again), and is not re-read: use --verify for
that. The download site (dl.fbaipublicfiles.com) is slow and has dropped transfers part-way, so an
interrupted download continues where it stopped when the command is run again, and a freshly downloaded
file whose checksum is recorded is checked before it is put in place; the resumable download is the same
approach FounDryClaw's venv/build_envs.py uses.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from esm_service import config  # noqa: E402
from esm_service.models import MODELS, resolve  # noqa: E402

ATTEMPTS = 5  # attempts in a row that fetch nothing


def download(url: str, partial: Path) -> None:
    """Fetch url into partial, continuing where an earlier attempt (or run) stopped. A partial file the
    server will not continue (no 206 reply) is started again. Gives up after ATTEMPTS attempts that
    fetched nothing; attempts that make progress do not count."""
    stalled = 0
    while True:
        have = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                total = response.headers.get("Content-Length")
                resumed = have and response.status == 206
                with partial.open("ab" if resumed else "wb") as handle:
                    for chunk in iter(lambda: response.read(1 << 20), b""):
                        handle.write(chunk)
                if total is not None and partial.stat().st_size != (have if resumed else 0) + int(total):
                    raise OSError("the connection ended before the whole file arrived")
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:  # the partial file is already complete
                return
            raise RuntimeError(f"downloading {url} failed: HTTP {exc.code}") from exc
        except OSError as exc:
            now = partial.stat().st_size if partial.exists() else 0
            stalled = 0 if now > have else stalled + 1
            if stalled == ATTEMPTS:
                raise RuntimeError(f"downloading {url} failed: {exc} ({stalled} attempts in a row fetched "
                                   f"nothing); run again to continue from the {now} bytes already "
                                   f"fetched") from exc


def digest_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, target: Path, expected: str | None) -> str:
    """Download one file; returns a one-line report. An existing file is kept."""
    if target.is_file():
        return f"{target.name}: already there"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print(f"  downloading {url}", flush=True)
    download(url, partial)
    if expected:
        actual = digest_of(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"{target.name}: the download's sha256 {actual} is not the recorded one "
                               f"({expected}); the incomplete file was removed")
        checked = "sha256 checked"
    else:
        checked = f"no recorded sha256 (actual {digest_of(partial)}; record it in esm_service/models.py)"
    os.replace(partial, target)
    return f"{target.name}: downloaded, {checked}"


def verify_model(name: str, folder: Path) -> tuple[list[str], int]:
    """Check what is on disk against the recorded checksums. Returns the report and how many are wrong.

    A multi-GB download from a site that drops transfers can end up truncated, and an existing file is
    never re-read while fetching, so this is the only way to find that out before a request does.
    """
    spec = resolve(name)
    if spec is None:
        known = ", ".join(MODELS)
        raise SystemExit(f"fetch_models: unknown model {name!r}. Known models: {known}")
    lines: list[str] = []
    wrong = 0
    for path, expected in ((folder / f"{spec.file_stem}.pt", spec.sha256),
                           (folder / f"{spec.file_stem}-contact-regression.pt", spec.regression_sha256)):
        if not path.is_file():
            lines.append(f"{path.name}: not here")
            continue
        actual = digest_of(path)
        if expected is None:
            lines.append(f"{path.name}: here, no recorded sha256 (actual {actual}; record it in "
                         "esm_service/models.py)")
        elif actual == expected:
            lines.append(f"{path.name}: sha256 ok")
        else:
            lines.append(f"{path.name}: sha256 {actual} is NOT the recorded one ({expected}); "
                         "delete the file and fetch it again")
            wrong += 1
    return lines, wrong


def fetch_model(name: str, folder: Path) -> list[str]:
    spec = resolve(name)
    if spec is None:
        known = ", ".join(MODELS)
        raise SystemExit(f"fetch_models: unknown model {name!r}. Known models: {known}")
    lines = [fetch(spec.weights_url, folder / f"{spec.file_stem}.pt", spec.sha256)]
    # The contact-regression file is small and only needed for contact prediction; a missing one is not
    # an error, the model still loads without it.
    try:
        lines.append(fetch(spec.regression_url, folder / f"{spec.file_stem}-contact-regression.pt",
                           spec.regression_sha256))
    except RuntimeError as exc:
        lines.append(f"contact-regression file skipped: {exc}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download ESM-2 model files for the ESM service.")
    parser.add_argument("models", nargs="*", help=f"Models to fetch (default: {config.DEFAULT_MODEL}).")
    parser.add_argument("--models-dir", help="Where to put them (default: ESM_MODELS_DIR, else ./models).")
    parser.add_argument("--list", action="store_true", help="Show the known models and exit.")
    parser.add_argument("--verify", action="store_true",
                        help="Check the files already here against their recorded sha256 instead of downloading.")
    args = parser.parse_args(argv)

    folder = Path(args.models_dir).resolve() if args.models_dir else config.MODELS_DIR
    if args.list:
        print(f"models folder: {folder}")
        for spec in MODELS.values():
            here = (folder / f"{spec.file_stem}.pt").is_file()
            print(f"  {spec.name:<10} {spec.parameters:>4} params  "
                  f"{'installed' if here else 'not installed'}  {spec.file_stem}.pt")
        return 0

    if args.verify:
        wrong = 0
        for name in (args.models or list(MODELS)):
            lines, bad = verify_model(name, folder)
            if lines == ["not here"] or all(line.endswith("not here") for line in lines):
                continue  # a model that was never installed is not a problem
            print(f"{name}:", flush=True)
            for line in lines:
                print(f"  {line}", flush=True)
            wrong += bad
        return 1 if wrong else 0

    for name in (args.models or [config.DEFAULT_MODEL]):
        print(f"{name}:", flush=True)
        for line in fetch_model(name, folder):
            print(f"  {line}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        raise SystemExit(f"fetch_models: {exc}")
