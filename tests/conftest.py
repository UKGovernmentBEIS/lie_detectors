"""Shared test fixtures.

The integration tests download a real probe from HuggingFace; they skip gracefully when
there is no network (or ``HF_HUB_OFFLINE=1``).
"""

from __future__ import annotations

import os
import socket

import pytest

# A small repo from the did-you-lie collection used by the integration tests.
REPO_ID = "ai-safety-institute/dyl-qwen-qwen3.5-122b-a10b-fp8"
# Smallest checkpoint in the repo (~14 KB, a difference-in-means probe).
SMALL_DIM_FILENAME = "l_28_ar_dim.pt"


def _online(
    host: str = "huggingface.co", port: int = 443, timeout: float = 3.0
) -> bool:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def hf_online() -> bool:
    if not _online():
        pytest.skip("No network access to huggingface.co (or HF_HUB_OFFLINE=1)")
    return True
