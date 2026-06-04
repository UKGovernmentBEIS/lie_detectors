"""Fast, offline tests: construct each probe in memory and run a forward pass."""

from __future__ import annotations

import torch

from lie_detectors import (
    LogisticRegressionProbe,
    MeanDifferenceProbe,
    MLPProbe,
    ProbeArchitecture,
)

D_MODEL = 16


def _randomise(probe):
    """Replace the NaN-initialised buffers with finite values so forward() runs."""
    state = {}
    for key, value in probe.state_dict().items():
        if not value.dtype.is_floating_point:
            state[key] = value
        elif key == "dataset_std":
            state[key] = torch.rand_like(value) + 0.5  # strictly positive
        else:
            state[key] = torch.randn_like(value)
    probe.load_state_dict(state)
    return probe


def test_mean_difference_forward():
    probe = _randomise(MeanDifferenceProbe(d_model=D_MODEL))
    assert probe.arch is ProbeArchitecture.DIFFERENCE_IN_MEANS

    batched = probe(torch.randn(4, D_MODEL))
    assert batched.shape == (4,)
    assert torch.isfinite(batched).all()
    assert batched.dtype.is_floating_point

    # A single (unbatched) activation gives a scalar score.
    assert probe(torch.randn(D_MODEL)).shape == ()
    assert probe.threshold.ndim == 0


def test_logistic_regression_forward():
    probe = _randomise(LogisticRegressionProbe(d_model=D_MODEL))
    assert probe.arch is ProbeArchitecture.LOGISTIC_REGRESSION

    batched = probe(torch.randn(4, D_MODEL))
    assert batched.shape == (4,)
    assert torch.isfinite(batched).all()
    assert probe(torch.randn(D_MODEL)).shape == ()


def test_mlp_forward():
    probe = _randomise(MLPProbe(d_model=D_MODEL))
    assert probe.arch is ProbeArchitecture.MULTI_LAYER_PERCEPTRON

    batched = probe(torch.randn(4, D_MODEL))
    assert batched.shape == (4,)
    assert torch.isfinite(batched).all()
    assert probe(torch.randn(D_MODEL)).shape == ()


def test_from_checkpoint_roundtrip_infers_shape():
    """from_checkpoint should rebuild the right shape purely from the state_dict."""
    original = _randomise(MLPProbe(d_model=D_MODEL, d_hidden=8))
    rebuilt = MLPProbe.from_checkpoint(original.state_dict())

    assert rebuilt.net[0].in_features == D_MODEL
    assert rebuilt.net[0].out_features == 8
    x = torch.randn(3, D_MODEL)
    assert torch.allclose(rebuilt(x), original(x))


def test_unset_probe_raises():
    """A freshly constructed (NaN) probe refuses to run."""
    import pytest

    with pytest.raises(RuntimeError):
        MeanDifferenceProbe(d_model=D_MODEL)(torch.randn(2, D_MODEL))
