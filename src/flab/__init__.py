"""flab: invent forecasters and benchmark them.

A forecaster is any callable  f(Y, h, period) -> F  with Y of shape (N, n) and F of shape (N, h).
Register it with @forecaster("name") in a module under flab/models/ and it shows up in the benchmark.
"""

from flab.registry import available, forecaster, load

__all__ = ["available", "forecaster", "load"]
