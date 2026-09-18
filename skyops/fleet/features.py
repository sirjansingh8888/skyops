"""NASA C-MAPSS loading and feature engineering for Remaining Useful Life (RUL) estimation.

Each subset (FD001..FD004) has one row per engine per cycle: 3 operating settings + 21 sensors.
Training engines run to failure; test engines are truncated and the true RUL is given separately.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from skyops import config

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
# sensors that are flat in FD001/FD003 carry no information; the rest trend with degradation
USEFUL_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
SENSOR_NAMES = {
    "s1": "Fan inlet temp (T2)", "s2": "LPC outlet temp (T24)", "s3": "HPC outlet temp (T30)", "s4": "LPT outlet temp (T50)",
    "s5": "Fan inlet pressure (P2)", "s6": "Bypass duct pressure (P15)", "s7": "HPC outlet pressure (P30)",
    "s8": "Physical fan speed (Nf)", "s9": "Physical core speed (Nc)", "s10": "Engine pressure ratio (epr)",
    "s11": "HPC outlet static pressure (Ps30)", "s12": "Fuel flow / Ps30 (phi)", "s13": "Corrected fan speed (NRf)",
    "s14": "Corrected core speed (NRc)", "s15": "Bypass ratio (BPR)", "s16": "Burner fuel-air ratio (farB)",
    "s17": "Bleed enthalpy (htBleed)", "s18": "Demanded fan speed (Nf_dmd)", "s19": "Demanded corrected fan speed (PCNfR_dmd)",
    "s20": "HPT coolant bleed (W31)", "s21": "LPT coolant bleed (W32)",
}
RUL_CAP = 125  # piecewise-linear target: engines are 'healthy' until ~125 cycles before failure
WINDOW = 10


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :26]
    df.columns = COLUMNS
    df[COLUMNS[2:]] = df[COLUMNS[2:]].astype(float)  # some sensors are integer-valued in the files
    return df


@lru_cache(maxsize=16)
def load_subset(subset: str = "FD001", split: str = "train", data_dir: Path = config.CMAPSS_DIR) -> pd.DataFrame:
    df = _read(Path(data_dir) / f"{split}_{subset}.txt")
    df["subset"] = subset
    df["engine_id"] = df["subset"] + "-" + df["unit"].astype(int).astype(str).str.zfill(3)
    return df


@lru_cache(maxsize=8)
def load_rul(subset: str = "FD001", data_dir: Path = config.CMAPSS_DIR) -> np.ndarray:
    return pd.read_csv(Path(data_dir) / f"RUL_{subset}.txt", header=None).iloc[:, 0].values.astype(float)


def add_rul_labels(train: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """RUL = cycles until the engine's last cycle, clipped at `cap` (standard piecewise-linear label)."""
    last = train.groupby("engine_id")["cycle"].transform("max")
    out = train.copy()
    out["rul"] = (last - out["cycle"]).clip(upper=cap).astype(float)
    return out


def op_condition(df: pd.DataFrame) -> pd.Series:
    """Discrete operating-condition id from the settings (6 regimes in FD002/FD004, 1 otherwise)."""
    return (df["op1"].round(0).astype(int).astype(str) + "_" + df["op3"].round(0).astype(int).astype(str))


def normalise_by_condition(df: pd.DataFrame, stats: dict | None = None, sensors: list[str] = USEFUL_SENSORS):
    """Z-score each sensor within its operating condition. Returns (df, stats) so test data reuses train stats."""
    out = df.copy()
    out["cond"] = op_condition(out)
    if stats is None:
        stats = {}
        for c, g in out.groupby("cond"):
            stats[c] = {s: (float(g[s].mean()), float(g[s].std() or 1.0)) for s in sensors}
    for c, g in out.groupby("cond"):
        st = stats.get(c)
        if st is None:  # unseen condition: fall back to global stats
            st = {s: (float(out[s].mean()), float(out[s].std() or 1.0)) for s in sensors}
        for s in sensors:
            mu, sd = st[s]
            out.loc[g.index, s] = (g[s] - mu) / (sd if sd > 1e-9 else 1.0)
    return out, stats


def make_features(df: pd.DataFrame, window: int = WINDOW, sensors: list[str] = USEFUL_SENSORS) -> pd.DataFrame:
    """Per-engine rolling statistics: current value, rolling mean/std, slope over the window, cycle count."""
    out = df.sort_values(["engine_id", "cycle"]).copy()
    g = out.groupby("engine_id", sort=False)
    feats = {"cycle": out["cycle"].astype(float)}
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float((x ** 2).sum())
    for s in sensors:
        roll = g[s].rolling(window, min_periods=1)
        feats[f"{s}"] = out[s].astype(float)
        feats[f"{s}_mean"] = roll.mean().reset_index(level=0, drop=True)
        feats[f"{s}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        # slope of a least-squares line over the window (0 until the window is full)
        slope = g[s].rolling(window).apply(lambda v: float(np.dot(x, v - v.mean()) / denom), raw=True)
        feats[f"{s}_slope"] = slope.reset_index(level=0, drop=True).fillna(0.0)
    F = pd.DataFrame(feats, index=out.index)
    F["engine_id"] = out["engine_id"].values
    F["subset"] = out["subset"].values
    F["unit"] = out["unit"].values
    if "rul" in out.columns:
        F["rul"] = out["rul"].values
    return F


def feature_columns(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in ("engine_id", "subset", "unit", "rul")]


def build_training_table(subsets: tuple[str, ...] = SUBSETS) -> tuple[pd.DataFrame, dict]:
    """Stack labelled, normalised, featurised training data for the chosen subsets."""
    parts, stats = [], {}
    for sub in subsets:
        tr = add_rul_labels(load_subset(sub, "train"))
        tr, st = normalise_by_condition(tr)
        stats[sub] = st
        parts.append(make_features(tr))
    return pd.concat(parts, ignore_index=True), stats


def build_test_table(subset: str, stats: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Featurised test rows for one subset plus a per-engine table of the last row with true RUL."""
    te = load_subset(subset, "test")
    te, _ = normalise_by_condition(te, stats.get(subset) if stats else None)
    F = make_features(te)
    last = F.groupby("engine_id").tail(1).copy()
    last["rul_true"] = load_rul(subset)[last["unit"].astype(int).values - 1]
    return F, last
