"""Rule-based and statistical anomaly flags for a snapshot of ADS-B states.

Every flag is a dict: type, severity (info | warning | alert | critical), icao24, callsign, detail, value.
Rules are deliberately explainable: an operator must be able to read *why* a track was flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from skyops import config

EMERGENCY_SQUAWKS = {"7500": "hijack (7500)", "7600": "radio failure (7600)", "7700": "general emergency (7700)"}


def _flag(row: pd.Series, type_: str, severity: str, detail: str, value: float | str | None = None) -> dict:
    return dict(type=type_, severity=severity, icao24=str(row.icao24), callsign=str(row.callsign),
                detail=detail, value=value, lat=float(row.latitude), lon=float(row.longitude),
                altitude_ft=round(float(row.altitude_ft)))


def rule_flags(snapshot: pd.DataFrame, previous: pd.DataFrame | None = None) -> list[dict]:
    """Deterministic rules on one snapshot (and the previous one for jump detection)."""
    flags: list[dict] = []
    prev = previous.drop_duplicates("icao24").set_index("icao24") if previous is not None else None

    for _, r in snapshot.iterrows():
        alt_m, spd, vs, grounded = float(r.altitude_m), float(r.velocity), float(r.vs_fpm), bool(r.on_ground)
        sq = str(r.squawk or "")
        if sq in EMERGENCY_SQUAWKS:
            flags.append(_flag(r, "EMERGENCY_SQUAWK", "critical", f"transponder squawking {EMERGENCY_SQUAWKS[sq]}", sq))
        if not grounded:
            if abs(vs) > 6000:
                flags.append(_flag(r, "EXTREME_VERTICAL_RATE", "critical", f"vertical rate {vs:+.0f} ft/min", round(vs)))
            elif abs(vs) > 4000:
                flags.append(_flag(r, "HIGH_VERTICAL_RATE", "alert", f"vertical rate {vs:+.0f} ft/min", round(vs)))
            if alt_m > 3000 and spd < 60:
                flags.append(_flag(r, "SLOW_AT_ALTITUDE", "alert",
                                   f"{spd*config.MS_TO_KT:.0f} kt at {alt_m/config.FT_TO_M:.0f} ft (possible stall or bad data)",
                                   round(spd * config.MS_TO_KT)))
            if spd > 300:
                flags.append(_flag(r, "OVERSPEED", "warning", f"ground speed {spd*config.MS_TO_KT:.0f} kt", round(spd * config.MS_TO_KT)))
            if alt_m < 1000 and spd > 180 and not r.near_airport:
                flags.append(_flag(r, "LOW_FAST_AWAY_FROM_AIRPORT", "warning",
                                   f"{spd*config.MS_TO_KT:.0f} kt below 3300 ft, {r.airport_km:.0f} km from nearest airport ({r.airport_icao})",
                                   round(alt_m / config.FT_TO_M)))
            if alt_m < 600 and not r.near_airport and abs(vs) < 300 and spd > 50:
                flags.append(_flag(r, "LOW_LEVEL_CRUISE", "info",
                                   f"level at {alt_m/config.FT_TO_M:.0f} ft, {r.airport_km:.0f} km from {r.airport_icao}",
                                   round(alt_m / config.FT_TO_M)))
        else:
            if spd > 80:
                flags.append(_flag(r, "GROUND_FLAG_MISMATCH", "warning",
                                   f"reported on ground but moving at {spd*config.MS_TO_KT:.0f} kt", round(spd * config.MS_TO_KT)))
        if float(r.stale_s) > 60:
            flags.append(_flag(r, "STALE_CONTACT", "info", f"last contact {r.stale_s:.0f} s ago (coverage gap)", int(r.stale_s)))

        if prev is not None and r.icao24 in prev.index:
            p = prev.loc[r.icao24]
            dt = float(r.snapshot_time - p.snapshot_time)
            if dt > 0 and not grounded:
                dalt = float(r.altitude_m - p.altitude_m)
                if abs(dalt) > 1000:
                    flags.append(_flag(r, "ALTITUDE_JUMP", "warning", f"altitude changed {dalt/config.FT_TO_M:+.0f} ft in {dt:.0f} s",
                                       round(dalt / config.FT_TO_M)))
                dh = (float(r.true_track - p.true_track) + 180.0) % 360.0 - 180.0
                if abs(dh) > 60 and spd > 100:
                    flags.append(_flag(r, "HEADING_JUMP", "warning", f"heading changed {dh:+.0f} deg in {dt:.0f} s at {spd*config.MS_TO_KT:.0f} kt",
                                       round(dh)))
    return flags


def statistical_flags(snapshot: pd.DataFrame, z_thresh: float = 3.0) -> list[dict]:
    """Speed outliers within altitude bands (robust z-score using median / MAD)."""
    flags: list[dict] = []
    air = snapshot[~snapshot.on_ground].copy()
    if len(air) < 20:
        return flags
    bands = pd.cut(air.altitude_m, bins=[-1, 1500, 4500, 7500, 10500, 20000], labels=["<5k", "5-15k", "15-25k", "25-35k", ">35k"])
    air["band"] = bands.astype(str)
    for band, g in air.groupby("band"):
        if len(g) < 8:
            continue
        med = g.velocity.median()
        mad = (g.velocity - med).abs().median() * 1.4826 + 1e-6
        z = (g.velocity - med) / mad
        for (_, r), zi in zip(g.iterrows(), z):
            if abs(zi) > z_thresh:
                flags.append(_flag(r, "SPEED_OUTLIER", "warning",
                                   f"{r.speed_kt:.0f} kt vs typical {med*config.MS_TO_KT:.0f} kt for band {band} ft (z={zi:+.1f})",
                                   round(float(zi), 1)))
    return flags


def detect_anomalies(snapshot: pd.DataFrame, previous: pd.DataFrame | None = None) -> list[dict]:
    flags = rule_flags(snapshot, previous) + statistical_flags(snapshot)
    order = {"critical": 0, "alert": 1, "warning": 2, "info": 3}
    flags.sort(key=lambda f: (order[f["severity"]], f["type"], f["callsign"]))
    return flags


def anomaly_summary(flags: list[dict]) -> dict:
    by_type: dict[str, int] = {}
    for f in flags:
        by_type[f["type"]] = by_type.get(f["type"], 0) + 1
    return dict(total=len(flags), critical=sum(f["severity"] == "critical" for f in flags),
                alert=sum(f["severity"] == "alert" for f in flags), warning=sum(f["severity"] == "warning" for f in flags),
                info=sum(f["severity"] == "info" for f in flags), by_type=by_type)
