"""The forecaster registry.

Each entry is a factory that takes no arguments and returns the callable  f(Y, h, period) -> F.
Factories keep heavy models (checkpoints, GPUs) from loading until a benchmark asks for them.
"""

import importlib
import pkgutil
from typing import Callable

import numpy as np

from horizoneval.prediction import Prediction

Forecast = Callable[[np.ndarray, int, int | None], "np.ndarray | Prediction"]

_REGISTRY: dict[str, Callable[[], Forecast]] = {}
_DISCOVERED = False


def forecaster(name: str):
    """Register a function f(Y, h, period), or a class whose instances are such callables (built with no arguments)."""

    def deco(obj):
        if name in _REGISTRY:
            raise ValueError(f"forecaster {name!r} is registered twice")
        _REGISTRY[name] = obj if isinstance(obj, type) else (lambda: obj)
        return obj

    return deco


def register_factory(name: str, factory: Callable[[], Forecast]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"forecaster {name!r} is registered twice")
    _REGISTRY[name] = factory


def _discover() -> None:
    """Import the baselines, the TimeCopilot wrappers, and every module in horizoneval.models."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    import horizoneval.baselines  # noqa: F401
    import horizoneval.tsfm  # noqa: F401
    import horizoneval.models

    for mod in pkgutil.iter_modules(horizoneval.models.__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"horizoneval.models.{mod.name}")


def available() -> list[str]:
    _discover()
    return sorted(_REGISTRY)


def load(name: str) -> Forecast:
    _discover()
    if name not in _REGISTRY:
        raise KeyError(f"unknown forecaster {name!r}; available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]()
