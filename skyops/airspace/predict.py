"""Short-horizon trajectory prediction for ADS-B state vectors.

Baseline: curvilinear dead reckoning (constant ground speed, constant turn rate estimated from the
last two headings, constant vertical rate). A learned model can replace it through the same
``Predictor`` interface; ``evaluate()`` measures any predictor against the captured future.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from skyops import config
from skyops.airspace.loader import Airspace


def step_great_circle(lat: np.ndarray, lon: np.ndarray, bearing_deg: np.ndarray, dist_m: np.ndarray):
    """Move points along a bearing over the sphere. All arguments are arrays in degrees / metres."""
    lat1, lon1 = np.radians(lat), np.radians(lon)
    brg = np.radians(bearing_deg)
    d = dist_m / config.R_EARTH_M
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(brg))
    lon2 = lon1 + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), (np.degrees(lon2) + 540.0) % 360.0 - 180.0


def wrap_deg(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


class Predictor(Protocol):
    name: str

    def predict(self, history: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
        """history: last k states for many aircraft (sorted by icao24, snapshot_time).
        Returns rows (icao24, horizon_s, latitude, longitude, altitude_m, true_track)."""


class DeadReckoningPredictor:
    """Constant speed / turn-rate / vertical-rate extrapolation in 10-second sub-steps."""

    name = "dead-reckoning"

    def __init__(self, substep_s: float = 10.0, max_turn_rate_dps: float = 3.0, use_turn_rate: bool = True):
        self.substep_s = substep_s
        self.max_turn_rate_dps = max_turn_rate_dps
        self.use_turn_rate = use_turn_rate

    def _turn_rates(self, history: pd.DataFrame) -> pd.Series:
        """deg/s per aircraft from the last two headings (0 when unknown or implausible)."""
        rates = {}
        for icao, g in history.groupby("icao24", sort=False):
            if len(g) < 2 or not self.use_turn_rate:
                rates[icao] = 0.0
                continue
            a, b = g.iloc[-2], g.iloc[-1]
            dt = float(b.snapshot_time - a.snapshot_time)
            if dt <= 0 or b.velocity < 30:
                rates[icao] = 0.0
                continue
            r = float(wrap_deg(np.array([b.true_track - a.true_track]))[0]) / dt
            rates[icao] = float(np.clip(r, -self.max_turn_rate_dps, self.max_turn_rate_dps))
        return pd.Series(rates)

    def predict(self, history: pd.DataFrame, horizons: tuple[int, ...] = config.PREDICTION_HORIZONS_S) -> pd.DataFrame:
        last = history.groupby("icao24", sort=False).tail(1).set_index("icao24")
        turn = self._turn_rates(history).reindex(last.index).fillna(0.0).values
        lat, lon = last.latitude.values.astype(float), last.longitude.values.astype(float)
        hdg = last.true_track.values.astype(float)
        spd = last.velocity.values.astype(float)
        vr = last.vertical_rate.values.astype(float)
        alt = last.altitude_m.values.astype(float)
        grounded = last.on_ground.values.astype(bool)
        spd = np.where(grounded, 0.0, spd)

        rows = []
        t = 0.0
        for h in sorted(horizons):
            while t < h:
                dt = min(self.substep_s, h - t)
                hdg = (hdg + turn * dt) % 360.0
                lat, lon = step_great_circle(lat, lon, hdg, spd * dt)
                alt = np.clip(alt + vr * dt, 0.0, None)
                t += dt
            rows.append(pd.DataFrame(dict(icao24=last.index.values, horizon_s=int(h), latitude=lat, longitude=lon,
                                          altitude_m=alt, true_track=hdg)))
        return pd.concat(rows, ignore_index=True)


def predict_at(airspace: Airspace, t_idx: int, predictor: Predictor | None = None,
               horizons: tuple[int, ...] = config.PREDICTION_HORIZONS_S, history_n: int = 3) -> pd.DataFrame:
    predictor = predictor or DeadReckoningPredictor()
    hist = airspace.history(t_idx, n=history_n)
    return predictor.predict(hist, horizons)


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * config.R_EARTH_M * np.arcsin(np.sqrt(a))


def _interp_track(track: pd.DataFrame, t: float) -> tuple[float, float] | None:
    """Linearly interpolate an aircraft's (lat, lon) at absolute time t using position-fix times.

    Returns None when t is outside the track or the bracketing fixes are more than 90 s apart.
    """
    tp = track["time_position"].values.astype(float)
    if t < tp[0] or t > tp[-1]:
        return None
    j = int(np.searchsorted(tp, t))
    if j == 0 or tp[j] == tp[j - 1]:
        return float(track["latitude"].values[j]), float(track["longitude"].values[j])
    if tp[j] - tp[j - 1] > 90:
        return None
    w = (t - tp[j - 1]) / (tp[j] - tp[j - 1])
    lat = track["latitude"].values[j - 1] * (1 - w) + track["latitude"].values[j] * w
    lon = track["longitude"].values[j - 1] * (1 - w) + track["longitude"].values[j] * w
    return float(lat), float(lon)


def evaluate(airspace: Airspace, predictor: Predictor | None = None, horizons: tuple[int, ...] = (60, 120, 180, 300),
             start_idx: int = 2) -> dict:
    """Score a predictor against the real future in the capture.

    For every snapshot index >= start_idx, predict each airborne aircraft at the horizons and compare with
    the aircraft's actual position interpolated at exactly (fix time + horizon) from later fixes.
    Returns error statistics in metres per horizon plus the number of comparisons.
    """
    predictor = predictor or DeadReckoningPredictor()
    errs: dict[int, list[float]] = {h: [] for h in horizons}
    tracks = {k: g.drop_duplicates("time_position").sort_values("time_position") for k, g in airspace.df.groupby("icao24")}
    for t_idx in range(start_idx, airspace.n_snapshots - 1):
        hist = airspace.history(t_idx, n=3)
        pred = predictor.predict(hist, tuple(horizons))
        last = hist.groupby("icao24").tail(1).set_index("icao24")
        for h in horizons:
            p = pred[pred.horizon_s == h].set_index("icao24")
            for icao, row in p.iterrows():
                st = last.loc[icao]
                if bool(st.on_ground):
                    continue
                target = float(st.time_position) + h
                actual = _interp_track(tracks[icao], target)
                if actual is None:
                    continue
                errs[h].append(float(haversine_m(np.array([row.latitude]), np.array([row.longitude]),
                                                 np.array([actual[0]]), np.array([actual[1]]))[0]))
    out = {}
    for h, e in errs.items():
        if e:
            a = np.array(e)
            out[int(h)] = dict(n=int(a.size), mean_m=float(a.mean()), median_m=float(np.median(a)),
                               p90_m=float(np.percentile(a, 90)))
    return dict(predictor=getattr(predictor, "name", "unknown"), horizons=out)
