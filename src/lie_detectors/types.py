"""Probe metadata types.

This module deliberately avoids importing :mod:`torch` so it stays cheap to import. It is
also the target of the legacy import-path alias installed in :mod:`lie_detectors.loading`,
so that probes pickled under the original ``deception.detectors.utils.types`` path resolve
to the :class:`ProbeArchitecture` defined here.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class ProbeArchitecture(str, Enum):
    """Probe architectures published in the AISI lie-detection collection.

    The string *values* are significant: published checkpoints pickle this enum by value,
    so ``"dim"`` / ``"lr"`` / ``"mlp"`` must not change.
    """

    DIFFERENCE_IN_MEANS = "dim"
    LOGISTIC_REGRESSION = "lr"
    MULTI_LAYER_PERCEPTRON = "mlp"


class TrainHyperparameters(BaseModel):
    """Hyperparameters stored alongside a probe checkpoint.

    ``extra="allow"`` keeps loading forward-compatible: checkpoints written with additional
    fields still parse, and every known field is optional so older checkpoints load too.
    """

    model_config = ConfigDict(extra="allow")

    layer: int | None = None
    layer_pct: int | None = None
    l2_lambda: float | None = None
    train_dataset: str | None = None
    probe_architecture: ProbeArchitecture | None = None
    weight_decay: float | None = None
    learning_rate: float | None = None
    epochs: int | None = None
