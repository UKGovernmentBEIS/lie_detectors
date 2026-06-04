"""Probe model definitions.

Only the three architectures actually published in the AISI lie-detection collection are
included: difference-in-means, logistic regression and a one-hidden-layer MLP. Each class
knows how to (a) run a forward pass and (b) rebuild itself from a checkpoint ``state_dict``.
Loading from disk / the Hub lives in :mod:`lie_detectors.loading`.
"""

from __future__ import annotations

from typing import Any, Self

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .types import ProbeArchitecture, TrainHyperparameters


class BaseProbe(nn.Module):
    """Base class for loadable probes (load + inference only).

    Subclasses set the ``arch`` class attribute, register their buffers/parameters in
    ``__init__``, implement :meth:`forward`, and implement :meth:`from_checkpoint` to infer
    their shape from a ``state_dict`` and rebuild themselves.
    """

    arch: ProbeArchitecture
    threshold: Tensor

    def __init__(self, hyperparams: TrainHyperparameters | None = None) -> None:
        super().__init__()
        self.hyperparams = hyperparams

    @property
    def layer(self) -> int | None:
        """The model layer whose activations this probe reads (``None`` if unknown)."""
        return self.hyperparams.layer if self.hyperparams is not None else None

    def forward(self, x: Float[Tensor, "*batch d_model"]) -> Float[Tensor, "*batch"]:
        raise NotImplementedError

    @classmethod
    def from_checkpoint(
        cls,
        state_dict: dict[str, Any],
        hyperparams: TrainHyperparameters | None = None,
    ) -> Self:
        raise NotImplementedError


class MeanDifferenceProbe(BaseProbe):
    """Score along the mean-difference direction between the two classes."""

    arch = ProbeArchitecture.DIFFERENCE_IN_MEANS
    direction: Tensor
    threshold: Tensor

    def __init__(
        self,
        d_model: int,
        threshold: float | None = None,
        hyperparams: TrainHyperparameters | None = None,
    ) -> None:
        super().__init__(hyperparams=hyperparams)
        self.register_buffer("direction", torch.full((d_model,), float("nan")))
        self.register_buffer(
            "threshold", torch.tensor(threshold if threshold is not None else 0.0)
        )

    @classmethod
    def from_checkpoint(
        cls,
        state_dict: dict[str, Any],
        hyperparams: TrainHyperparameters | None = None,
    ) -> Self:
        d_model = state_dict["direction"].shape[0]
        probe = cls(d_model=d_model, hyperparams=hyperparams)
        probe.load_state_dict(state_dict)
        probe.eval()
        return probe

    def forward(self, x: Float[Tensor, "*batch d_model"]) -> Float[Tensor, "*batch"]:
        if self.direction.isnan().any():
            raise RuntimeError("direction is unset; load a trained checkpoint first.")
        return x @ self.direction


class LogisticRegressionProbe(BaseProbe):
    """Logistic-regression classifier over standardised activations."""

    arch = ProbeArchitecture.LOGISTIC_REGRESSION
    dataset_mean: Tensor
    dataset_std: Tensor
    threshold: Tensor

    def __init__(
        self,
        d_model: int,
        threshold: float | None = None,
        hyperparams: TrainHyperparameters | None = None,
    ) -> None:
        super().__init__(hyperparams=hyperparams)
        self.linear = nn.Linear(d_model, 1)
        self.register_buffer("dataset_mean", torch.full((d_model,), float("nan")))
        self.register_buffer("dataset_std", torch.full((d_model,), float("nan")))
        self.register_buffer(
            "threshold", torch.tensor(threshold if threshold is not None else 0.0)
        )

    @classmethod
    def from_checkpoint(
        cls,
        state_dict: dict[str, Any],
        hyperparams: TrainHyperparameters | None = None,
    ) -> Self:
        d_model = state_dict["linear.weight"].shape[1]
        probe = cls(d_model=d_model, hyperparams=hyperparams)
        probe.load_state_dict(state_dict)
        probe.eval()
        return probe

    def forward(self, x: Float[Tensor, "*batch d_model"]) -> Float[Tensor, "*batch"]:
        if self.dataset_mean.isnan().any() or self.dataset_std.isnan().any():
            raise RuntimeError(
                "dataset_mean/dataset_std are unset; load a trained checkpoint first."
            )
        x = (x - self.dataset_mean) / self.dataset_std
        return self.linear(x).squeeze(-1)


class MLPProbe(BaseProbe):
    """One-hidden-layer MLP over standardised activations."""

    arch = ProbeArchitecture.MULTI_LAYER_PERCEPTRON
    dataset_mean: Tensor
    dataset_std: Tensor
    threshold: Tensor

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        threshold: float | None = None,
        hyperparams: TrainHyperparameters | None = None,
    ) -> None:
        super().__init__(hyperparams=hyperparams)
        if d_hidden is None:
            d_hidden = max(d_model // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )
        self.register_buffer("dataset_mean", torch.full((d_model,), float("nan")))
        self.register_buffer("dataset_std", torch.full((d_model,), float("nan")))
        self.register_buffer(
            "threshold", torch.tensor(threshold if threshold is not None else 0.0)
        )

    @classmethod
    def from_checkpoint(
        cls,
        state_dict: dict[str, Any],
        hyperparams: TrainHyperparameters | None = None,
    ) -> Self:
        d_hidden, d_model = state_dict["net.0.weight"].shape
        probe = cls(d_model=d_model, d_hidden=d_hidden, hyperparams=hyperparams)
        probe.load_state_dict(state_dict)
        probe.eval()
        return probe

    def forward(self, x: Float[Tensor, "*batch d_model"]) -> Float[Tensor, "*batch"]:
        if self.dataset_mean.isnan().any() or self.dataset_std.isnan().any():
            raise RuntimeError(
                "dataset_mean/dataset_std are unset; load a trained checkpoint first."
            )
        x = (x - self.dataset_mean) / self.dataset_std
        return self.net(x).squeeze(-1)


PROBE_REGISTRY: dict[ProbeArchitecture, type[BaseProbe]] = {
    ProbeArchitecture.DIFFERENCE_IN_MEANS: MeanDifferenceProbe,
    ProbeArchitecture.LOGISTIC_REGRESSION: LogisticRegressionProbe,
    ProbeArchitecture.MULTI_LAYER_PERCEPTRON: MLPProbe,
}
