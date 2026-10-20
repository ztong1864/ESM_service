"""Test setup: mock mode, chosen before esm_service.config reads the environment at import."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ESM_SERVICE_MOCK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from esm_service.api import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
