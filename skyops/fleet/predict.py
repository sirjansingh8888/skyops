"""Fleet health: Remaining-Useful-Life estimates per engine with uncertainty and a maintenance verdict.

Uses the LightGBM models from ``skyops.fleet.train`` when present (models/rul_lgbm/). Until they are
trained, a training-free k-nearest-neighbour estimator over the C-MAPSS training windows keeps the API
and UI fully functional (it is also a useful transparent baseline for the pitch).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from skyops import config
from skyops.fleet.features import (SENSOR_NAMES, SUBSETS, USEFUL_SENSORS, build_test_table, build_training_table,
                                   feature_columns)

HEALTH_BANDS = [  # (max RUL cycles, state, action)
    (15, "ground", "Ground immediately: predicted failure within 15 cycles"),
    (40, "maintenance", "Schedule shop visit within the next 10 cycles"),
    (80, "watch", "Monitor: trending sensors, plan inspection"),
    (10 ** 9, "healthy", "No action"),
]


def health_state(rul: float) -> tuple[str, str]:
    for cap, state, action in HEALTH_BANDS:
        if rul <= cap:
            return state, action
    return "healthy", "No action"


class KNNFallback:
    """Median RUL of the k most similar training windows (Euclidean on standardised features)."""

    name = "knn-fallback"

    def __init__(self, subsets: tuple[str, ...] = SUBSETS, k: int = 25, max_rows: int = 60_000, seed: int = 0):
        F, self.stats = build_training_table(subsets)
        self.cols = feature_columns(F)
        rng = np.random.default_rng(seed)
        if len(F) > max_rows:
            F = F.iloc[np.sort(rng.choice(len(F), max_rows, replace=False))]
        X = F[self.cols].values.astype(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.X = (X - self.mu) / self.sd
        self.y = F["rul"].values.astype(float)
        self.k = k

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z = (X.astype(float) - self.mu) / self.sd
        p50, p10, p90 = [], [], []
        for z in Z:
            d = np.sqrt(((self.X - z) ** 2).sum(1))
            nn = self.y[np.argpartition(d, self.k)[:self.k]]
            p50.append(np.median(nn))
            p10.append(np.percentile(nn, 10))
            p90.append(np.percentile(nn, 90))
        return np.array(p50), np.array(p10), np.array(p90)


class LightGBMRUL:
    name = "lightgbm"

    def __init__(self, model_dir: Path):
        import lightgbm as lgb

        self.manifest = json.loads((model_dir / "manifest.json").read_text())
        self.cols = self.manifest["features"]
        self.stats = self.manifest["stats"]
        # Load from a string with line endings normalised: a checkout with CRLF conversion (Git on Windows without
        # the repo's .gitattributes) would otherwise make LightGBM abort with "Model format error".
        self.b = {q: lgb.Booster(model_str=(model_dir / f"{q}.txt").read_text(encoding="utf-8").replace("\r\n", "\n"))
                  for q in ("p50", "p10", "p90")}

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p50 = self.b["p50"].predict(X)
        lo, hi = self.b["p10"].predict(X), self.b["p90"].predict(X)
        if self.manifest.get("quantile_mode") == "offset":  # quantile boosters predict offsets around p50
            q = float(self.manifest.get("conformal_q", 0.0))  # conformal widening to reach nominal coverage
            lo, hi = p50 + lo - q, p50 + hi + q
        p10, p90 = np.minimum(lo, p50), np.maximum(hi, p50)
        return np.clip(p50, 0, None), np.clip(p10, 0, None), np.clip(p90, 0, 130.0)


@lru_cache(maxsize=1)
def get_model():
    model_dir = config.ROOT / "models" / "rul_lgbm"
    if (model_dir / "manifest.json").exists():
        return LightGBMRUL(model_dir)
    return KNNFallback()


_TEST_TABLES: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}


def _test_table(subset: str, model) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = f"{model.name}:{subset}"
    if key not in _TEST_TABLES:
        _TEST_TABLES[key] = build_test_table(subset, model.stats)
    return _TEST_TABLES[key]


def fleet_status(subset: str = "FD001") -> dict:
    """Per-engine RUL estimate for every test engine in `subset` plus a fleet summary."""
    model = get_model()
    F, last = _test_table(subset, model)
    p50, p10, p90 = model.predict(last[model.cols].values)
    engines = []
    for (_, row), a, lo, hi in zip(last.iterrows(), p50, p10, p90):
        state, action = health_state(float(a))
        trending = sorted(((s, float(row[f"{s}_slope"])) for s in USEFUL_SENSORS), key=lambda t: -abs(t[1]))[:3]
        engines.append(dict(
            engine_id=row["engine_id"], unit=int(row["unit"]), cycles_observed=int(row["cycle"]),
            rul_p50=round(float(a), 1), rul_p10=round(float(lo), 1), rul_p90=round(float(hi), 1),
            rul_true=float(row["rul_true"]), state=state, action=action,
            trending=[dict(sensor=s, name=SENSOR_NAMES[s], slope=round(v, 4)) for s, v in trending],
        ))
    order = {"ground": 0, "maintenance": 1, "watch": 2, "healthy": 3}
    engines.sort(key=lambda e: (order[e["state"]], e["rul_p50"]))
    yt = np.minimum(np.array([e["rul_true"] for e in engines]), 125.0)
    yp = np.array([e["rul_p50"] for e in engines])
    counts = {s: sum(e["state"] == s for e in engines) for s in order}
    return dict(subset=subset, model=model.name, n_engines=len(engines), counts=counts,
                rmse_vs_truth=round(float(np.sqrt(np.mean((yp - yt) ** 2))), 2), engines=engines)


def engine_history(subset: str, unit: int) -> dict:
    """Raw sensor trace (normalised) and rolling RUL estimate for one test engine, for the detail chart."""
    model = get_model()
    F, _ = _test_table(subset, model)
    g = F[F["unit"] == unit].sort_values("cycle")
    if g.empty:
        return dict(subset=subset, unit=unit, cycles=[], rul=[], sensors={})
    p50, p10, p90 = model.predict(g[model.cols].values)
    sensors = {s: [round(float(v), 3) for v in g[s].values] for s in ("s2", "s3", "s4", "s7", "s11", "s12", "s15", "s21")}
    return dict(subset=subset, unit=unit, cycles=g["cycle"].astype(int).tolist(),
                rul=dict(p50=np.round(p50, 1).tolist(), p10=np.round(p10, 1).tolist(), p90=np.round(p90, 1).tolist()),
                sensors=sensors, sensor_names={s: SENSOR_NAMES[s] for s in sensors})
