"""Load probes from a local checkpoint or the HuggingFace Hub.

Published checkpoints pickle their ``probe_architecture`` as a reference to the original
(private) module path ``deception.detectors.utils.types.ProbeArchitecture``. So that
``torch.load`` can resolve that reference in this standalone package, we alias that import
path to :mod:`lie_detectors.types` once, on import. This is simpler and more robust than a
custom unpickler, and :func:`load_probe` additionally normalises the architecture to our own
enum so registry lookups work even if a real ``deception`` package is installed.
"""

from __future__ import annotations

import sys
import types as _stdlib_types
from pathlib import Path
from typing import Any

import torch

from . import types as _types_module
from .probes import PROBE_REGISTRY, BaseProbe
from .types import ProbeArchitecture, TrainHyperparameters

_LEGACY_MODULE = "deception.detectors.utils.types"


def _install_legacy_alias() -> None:
    """Alias the original probe module path to this package (no-op if it already exists)."""
    if _LEGACY_MODULE in sys.modules:
        return
    for parent in ("deception", "deception.detectors", "deception.detectors.utils"):
        sys.modules.setdefault(parent, _stdlib_types.ModuleType(parent))
    sys.modules[_LEGACY_MODULE] = _types_module


_install_legacy_alias()


def _read_checkpoint(
    path: Path | str,
) -> tuple[dict[str, Any], TrainHyperparameters | None]:
    """Load a ``.pt`` checkpoint, returning ``(state_dict, hyperparams)``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    # weights_only=False: the checkpoint stores a ProbeArchitecture enum object. Only load
    # probes from trusted sources (e.g. the AISI HuggingFace repos).
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        hp = (
            TrainHyperparameters(**raw["hyperparams"])
            if raw.get("hyperparams")
            else None
        )
        return raw["state_dict"], hp
    return raw, None


def load_probe(path: Path | str) -> BaseProbe:
    """Load a probe from a local ``.pt`` checkpoint, dispatching on its stored architecture."""
    state_dict, hyperparams = _read_checkpoint(path)
    if hyperparams is None or hyperparams.probe_architecture is None:
        raise ValueError(
            f"Cannot determine probe architecture from {path}; "
            "the checkpoint has no 'probe_architecture' in its hyperparams."
        )
    # Normalise to our enum (handles a foreign enum or a plain string value).
    arch = ProbeArchitecture(hyperparams.probe_architecture)
    probe_cls = PROBE_REGISTRY.get(arch)
    if probe_cls is None:
        raise ValueError(f"Unsupported probe architecture: {arch!r}")
    return probe_cls.from_checkpoint(state_dict, hyperparams)


def get_probe(
    repo_id: str,
    *,
    filename: str = "probe.pt",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
) -> BaseProbe:
    """Download a probe from a HuggingFace model repo and load it.

    Args:
        repo_id: HuggingFace repo, e.g. ``"ai-safety-institute/dyl-qwen-qwen3.5-122b-a10b-fp8"``.
        filename: File within the repo. Defaults to ``"probe.pt"`` (the best sweep
            checkpoint). Pass a specific sweep filename (see ``sweep.json``) to pick another.
        revision: Git revision (branch / tag / commit SHA) on HuggingFace.
        cache_dir: Where to cache downloaded files.
        token: HuggingFace auth token (for gated/private repos).
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    return load_probe(path)
