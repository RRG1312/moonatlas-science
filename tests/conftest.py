"""Shared paths for the test suite. No downloads: everything here runs on a clean clone."""

from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="session")
def smoke_dataset() -> Path:
    """The redistributable example build (examples/science-smoke)."""
    return EXAMPLES / "science-smoke"
