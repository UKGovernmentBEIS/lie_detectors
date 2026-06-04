"""Load pre-trained lie-detection probes from the AI Safety Institute HuggingFace collection.

>>> from lie_detectors import get_probe
>>> probe = get_probe("ai-safety-institute/dyl-qwen-qwen3.5-122b-a10b-fp8")
>>> scores = probe(activations)   # per-sample deception scores
>>> probe.threshold               # calibrated decision threshold
"""

from __future__ import annotations

from .loading import get_probe, load_probe
from .probes import (
    PROBE_REGISTRY,
    BaseProbe,
    LogisticRegressionProbe,
    MeanDifferenceProbe,
    MLPProbe,
)
from .types import ProbeArchitecture, TrainHyperparameters

__version__ = "0.1.0"

__all__ = [
    "get_probe",
    "load_probe",
    "BaseProbe",
    "MeanDifferenceProbe",
    "LogisticRegressionProbe",
    "MLPProbe",
    "PROBE_REGISTRY",
    "ProbeArchitecture",
    "TrainHyperparameters",
    "__version__",
]
