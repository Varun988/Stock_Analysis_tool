import os

import pytest
from fastapi.testclient import TestClient

# Keep tests on safe/local defaults unless explicitly overridden.
os.environ.setdefault("AI_EXPLANATION_PROVIDER", "MOCK")
os.environ.setdefault("RESEARCH_PROVIDER", "MOCK")

from app.main import app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)