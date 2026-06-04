"""The crux test (offline): a checkpoint pickled under the *original* private module path
``deception.detectors.utils.types.ProbeArchitecture`` must still load in this standalone
package, resolving to our own enum.
"""

from __future__ import annotations

import enum
import sys
import types

import pytest
import torch

from lie_detectors import MeanDifferenceProbe, ProbeArchitecture, load_probe

LEGACY_MODULE = "deception.detectors.utils.types"


def _save_legacy_checkpoint(path) -> None:
    """Save a checkpoint whose ``probe_architecture`` pickles under the legacy module path."""
    # A stand-in enum that pickles as deception.detectors.utils.types.ProbeArchitecture.
    legacy_enum = enum.Enum(
        "ProbeArchitecture", {"DIFFERENCE_IN_MEANS": "dim"}, type=str
    )
    legacy_enum.__module__ = LEGACY_MODULE

    legacy_mod = types.ModuleType(LEGACY_MODULE)
    setattr(legacy_mod, "ProbeArchitecture", legacy_enum)

    saved = sys.modules.get(LEGACY_MODULE)
    sys.modules[LEGACY_MODULE] = legacy_mod
    for parent in ("deception", "deception.detectors", "deception.detectors.utils"):
        sys.modules.setdefault(parent, types.ModuleType(parent))
    try:
        checkpoint = {
            "state_dict": {
                "direction": torch.randn(8),
                "threshold": torch.tensor(0.5),
            },
            "hyperparams": {
                "probe_architecture": legacy_enum.DIFFERENCE_IN_MEANS,
                "layer": 28,
            },
        }
        torch.save(checkpoint, path)
    finally:
        # Restore the alias our package installed at import time.
        if saved is not None:
            sys.modules[LEGACY_MODULE] = saved
        else:
            sys.modules.pop(LEGACY_MODULE, None)


def test_load_probe_resolves_legacy_module_path(tmp_path):
    path = tmp_path / "legacy_dim.pt"
    _save_legacy_checkpoint(path)

    probe = load_probe(path)

    assert isinstance(probe, MeanDifferenceProbe)
    assert probe.hyperparams is not None
    # Resolved to *our* enum, by identity — so PROBE_REGISTRY lookups work.
    assert probe.hyperparams.probe_architecture is ProbeArchitecture.DIFFERENCE_IN_MEANS
    assert probe.hyperparams.layer == 28

    scores = probe(torch.randn(4, 8))
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()


def test_without_alias_plain_torch_load_fails(tmp_path):
    """Documents *why* the alias is needed: without it, torch.load can't find the module."""
    path = tmp_path / "legacy_dim.pt"
    _save_legacy_checkpoint(path)

    removed = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "deception" or name.startswith("deception.")
    }
    try:
        with pytest.raises(ModuleNotFoundError):
            torch.load(path, map_location="cpu", weights_only=False)
    finally:
        sys.modules.update(removed)
