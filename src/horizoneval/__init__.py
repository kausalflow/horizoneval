"""horizoneval: invent forecasters and benchmark them.

A forecaster is any callable  f(Y, h, period) -> F  with Y of shape (N, n) and F of shape (N, h).
Register it with @forecaster("name") in a module under horizoneval/models/ and it shows up in the benchmark.
"""

from horizoneval.registry import available, forecaster, load

__all__ = ["available", "forecaster", "load"]
