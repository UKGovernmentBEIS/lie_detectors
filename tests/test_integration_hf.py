"""Integration tests: download a real probe from HuggingFace and run a forward pass.

Run with::

    uv run pytest -m integration

These are skipped by default (see addopts in pyproject.toml) and skip gracefully offline.
"""

from __future__ import annotations

import pytest
import torch

from conftest import REPO_ID, SMALL_DIM_FILENAME

from lie_detectors import MeanDifferenceProbe, ProbeArchitecture, get_probe


def _infer_d_model(probe) -> int:
    if hasattr(probe, "direction"):
        return probe.direction.numel()
    if hasattr(probe, "linear"):
        return probe.linear.in_features
    return probe.net[0].in_features


@pytest.mark.integration
def test_download_and_forward_dim_probe(hf_online, tmp_path):
    probe = get_probe(REPO_ID, filename=SMALL_DIM_FILENAME, cache_dir=str(tmp_path))

    assert isinstance(probe, MeanDifferenceProbe)
    assert probe.hyperparams is not None
    assert probe.hyperparams.probe_architecture is ProbeArchitecture.DIFFERENCE_IN_MEANS
    assert probe.threshold.ndim == 0 and probe.threshold.dtype.is_floating_point

    d_model = _infer_d_model(probe)
    scores = probe(torch.randn(8, d_model))
    assert scores.shape == (8,)
    assert torch.isfinite(scores).all()
    assert scores.dtype.is_floating_point


@pytest.mark.integration
def test_get_probe_default_checkpoint(hf_online, tmp_path):
    # No filename => the repo's default "probe.pt" (the best sweep checkpoint).
    probe = get_probe(REPO_ID, cache_dir=str(tmp_path))

    d_model = _infer_d_model(probe)
    scores = probe(torch.randn(2, d_model))
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
