"""Pairwise loss-of-separation detection on current and predicted positions.

An aircraft pair is in conflict when horizontal distance < 5 NM (3 NM in terminal areas) AND vertical
distance < 1000 ft at the same instant. We evaluate the current snapshot (horizon 0) plus every
prediction horizon, then report the earliest horizon at which separation is lost, the closest point
of approach and whether the pair is converging.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from skyops import config


def _pairwise_haversine_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    p = np.radians(lat)[:, None]
    q = np.radians(lat)[None, :]
    dl = np.radians(lon)[:, None] - np.radians(lon)[None, :]
    a = np.sin((p - q) / 2) ** 2 + np.cos(p) * np.cos(q) * np.sin(dl / 2) ** 2
    return 2 * config.R_EARTH_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def detect_conflicts(current: pd.DataFrame, predictions: pd.DataFrame | None,
                     h_sep_nm: float = config.HORIZONTAL_SEP_NM, v_sep_ft: float = config.VERTICAL_SEP_FT,
                     terminal_h_sep_nm: float = config.TERMINAL_HORIZONTAL_SEP_NM, min_alt_m: float = 300.0,
                     horizon_limit_s: int = 300, altitude_tolerance_ft: float = 150.0) -> list[dict]:
    """Return conflict records sorted by severity then time-to-loss.

    current:     one snapshot (rows per aircraft, from Airspace.snapshot)
    predictions: rows (icao24, horizon_s, latitude, longitude, altitude_m) from a Predictor

    Barometric altitude is quantised to 25 ft and noisy, so two aircraft correctly separated by 1000 ft
    often report 950-1000 ft. A pair only counts as losing vertical separation below
    (v_sep_ft - altitude_tolerance_ft); pairs inside the tolerance band are reported as 'marginal'.
    """
    air = current[(~current["on_ground"]) & (current["altitude_m"] >= min_alt_m)].drop_duplicates("icao24")
    if len(air) < 2:
        return []
    air = air.set_index("icao24")
    ids = air.index.values
    n = len(ids)

    # per-horizon position arrays aligned to `ids`
    frames: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = [
        (0, air.latitude.values.astype(float), air.longitude.values.astype(float), air.altitude_m.values.astype(float))
    ]
    if predictions is not None and len(predictions):
        for h, g in predictions.groupby("horizon_s"):
            if h > horizon_limit_s:
                continue
            g = g.drop_duplicates("icao24").set_index("icao24").reindex(ids)
            frames.append((int(h), g.latitude.values.astype(float), g.longitude.values.astype(float),
                           g.altitude_m.values.astype(float)))
    frames.sort(key=lambda f: f[0])

    # terminal-area pairs (either aircraft near an airport) use the tighter minimum
    near = air["near_airport"].values.astype(bool)
    h_sep_m = np.where(near[:, None] | near[None, :], terminal_h_sep_nm, h_sep_nm) * config.NM_TO_M
    v_loss_m = (v_sep_ft - altitude_tolerance_ft) * config.FT_TO_M
    v_sep_m = v_sep_ft * config.FT_TO_M
    iu = np.triu_indices(n, k=1)

    first_loss = np.full((n, n), np.inf)      # earliest horizon with a genuine loss of separation
    first_marginal = np.full((n, n), np.inf)  # earliest horizon inside the altitude tolerance band
    min_dist = np.full((n, n), np.inf)
    t_cpa = np.zeros((n, n))
    dist0 = None
    for h, lat, lon, alt in frames:
        D = _pairwise_haversine_m(lat, lon)
        V = np.abs(alt[:, None] - alt[None, :])
        valid = ~(np.isnan(D) | np.isnan(V))
        if dist0 is None:
            dist0 = D.copy()
        close = valid & (D < h_sep_m)
        loss = close & (V < v_loss_m)
        marginal = close & (V >= v_loss_m) & (V < v_sep_m)
        first_loss = np.where(loss & np.isinf(first_loss), h, first_loss)
        first_marginal = np.where(marginal & np.isinf(first_marginal), h, first_marginal)
        better = valid & (D < min_dist)
        t_cpa = np.where(better, h, t_cpa)
        min_dist = np.where(better, D, min_dist)

    out = []
    for i, j in zip(*iu):
        genuine = not np.isinf(first_loss[i, j])
        if not genuine and np.isinf(first_marginal[i, j]):
            continue
        t_loss = float(first_loss[i, j]) if genuine else float(first_marginal[i, j])
        a, b = air.iloc[i], air.iloc[j]
        converging = bool(min_dist[i, j] < dist0[i, j] - 1.0)
        if not genuine:
            severity = "marginal"
        elif t_loss == 0:
            severity = "critical"
        elif t_loss <= 120:
            severity = "alert"
        else:
            severity = "warning"
        # positions at the loss horizon for drawing
        fr = next(f for f in frames if f[0] == t_loss)
        out.append(dict(
            id=f"{ids[i]}-{ids[j]}",
            a=dict(icao24=ids[i], callsign=a.callsign, altitude_ft=round(float(a.altitude_ft)), speed_kt=round(float(a.speed_kt)),
                   lat=float(fr[1][i]), lon=float(fr[2][i])),
            b=dict(icao24=ids[j], callsign=b.callsign, altitude_ft=round(float(b.altitude_ft)), speed_kt=round(float(b.speed_kt)),
                   lat=float(fr[1][j]), lon=float(fr[2][j])),
            severity=severity,
            t_loss_s=t_loss,
            t_cpa_s=float(t_cpa[i, j]),
            cpa_nm=round(float(min_dist[i, j]) / config.NM_TO_M, 2),
            now_nm=round(float(dist0[i, j]) / config.NM_TO_M, 2),
            v_sep_ft=round(abs(float(a.altitude_ft) - float(b.altitude_ft))),
            terminal=bool(near[i] or near[j]),
            converging=converging,
            h_sep_nm_applied=float(h_sep_m[i, j] / config.NM_TO_M),
        ))
    order = {"critical": 0, "alert": 1, "warning": 2, "marginal": 3}
    out.sort(key=lambda r: (order[r["severity"]], r["t_loss_s"], r["cpa_nm"]))
    return out


def conflict_summary(conflicts: list[dict]) -> dict:
    return dict(
        total=len(conflicts),
        critical=sum(c["severity"] == "critical" for c in conflicts),
        alert=sum(c["severity"] == "alert" for c in conflicts),
        warning=sum(c["severity"] == "warning" for c in conflicts),
        marginal=sum(c["severity"] == "marginal" for c in conflicts),
        converging=sum(c["converging"] for c in conflicts),
    )
