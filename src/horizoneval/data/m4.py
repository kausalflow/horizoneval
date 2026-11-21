"""The M4 competition data and its published point-forecast results (Makridakis, Spiliotis & Assimakopoulos, 2020).

Everything comes from the organisers' repository, github.com/Mcompetitions/M4-methods, and is cached under
data/m4 (or $HORIZONEVAL_DATA/m4):

- Dataset/Train/<Freq>-train.csv, Dataset/Test/<Freq>-test.csv   the 100,000 series
- Point Forecasts/submission-<id>.rar                             the forecasts each participant submitted
- Evaluation and Ranks.xlsx                                       the official sMAPE and MASE per frequency; its
                                                                  'Point Forecasts-Frequency' sheet is copied into
                                                                  m4_published.csv next to this file

Unpacking a submission needs the `replicate` extra (libarchive-c, with the system libarchive).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master"
FREQS = {"Yearly": (6, 1), "Quarterly": (8, 4), "Monthly": (18, 12), "Weekly": (13, 1), "Daily": (14, 1),
         "Hourly": (48, 24)}  # horizon, seasonal period used by the official MASE
# submissions rescored by `horizoneval replicate m4`: file id -> column name
SUBMISSIONS = {"Naive": "m4_Naive", "Naive2": "m4_Naive2", "118": "m4_118_Smyl", "245": "m4_245_MonteroManso",
               "237": "m4_237_Pawlikowski"}
PUBLISHED = Path(__file__).with_name("m4_published.csv")


def root() -> Path:
    return Path(os.environ.get("HORIZONEVAL_DATA", "data")) / "m4"


def _fetch(rel: str) -> Path:
    path = root() / rel
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/{rel.replace(' ', '%20')}"
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def _wide(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    df = pd.read_csv(path, index_col=0)
    vals = df.to_numpy(float)
    return df.index.to_numpy(str), [r[np.isfinite(r)] for r in vals]


def load(freq: str):
    """(ids, histories as a list of arrays, test (N, h)) for one M4 frequency."""
    h, _ = FREQS[freq]
    ids, train = _wide(_fetch(f"Dataset/Train/{freq}-train.csv"))
    tids, test = _wide(_fetch(f"Dataset/Test/{freq}-test.csv"))
    assert (ids == tids).all() and all(len(t) == h for t in test)
    return ids, train, np.stack(test)


def submission(name: str) -> pd.DataFrame:
    """A participant's submitted point forecasts: index id, columns F1..F48 (NaN past each series' horizon)."""
    csv = root() / "submissions" / f"submission-{name}.csv"
    if not csv.exists():
        import libarchive

        rar = _fetch(f"Point Forecasts/submission-{name}.rar")
        cwd = os.getcwd()
        csv.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chdir(csv.parent)
            libarchive.extract_file(str(rar.resolve()))
        finally:
            os.chdir(cwd)
    return pd.read_csv(csv, index_col=0)


def published() -> pd.DataFrame:
    """The official results: one row per method, columns smape_<Freq> and mase_<Freq> (and _Total)."""
    return pd.read_csv(PUBLISHED, dtype={"method": str})
