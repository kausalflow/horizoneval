"""Datasets by name: 'synthetic:<name>' or 'gift:<dataset>/<freq>', plus named suites of them."""

from dataclasses import dataclass

import numpy as np

from horizoneval.data.synthetic import SYNTHETIC

SUITES = {
    "synthetic": [f"synthetic:{k}" for k in SYNTHETIC],
    "gift-small": ["gift:electricity/H", "gift:solar/H", "gift:m4_hourly", "gift:m4_daily", "gift:jena_weather/H"],
}


@dataclass
class Dataset:
    name: str
    Y: np.ndarray  # (N, n)
    period: int | None


def load_dataset(spec: str, N: int, n: int, rng: np.random.Generator) -> Dataset:
    source, _, name = spec.partition(":")
    if source == "synthetic":
        gen, period = SYNTHETIC[name]
        return Dataset(spec, gen(n, N, rng), period)
    if source == "gift":
        from horizoneval.data.gift import load_gift

        Y, period = load_gift(name, N, n, rng)
        return Dataset(spec, Y, period)
    raise ValueError(f"unknown dataset {spec!r}: use synthetic:<name> or gift:<dataset>/<freq>")


def expand(specs: list[str]) -> list[str]:
    """Suite names become their datasets; anything else passes through."""
    return [d for s in specs for d in SUITES.get(s, [s])]
