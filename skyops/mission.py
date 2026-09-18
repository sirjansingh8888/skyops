"""Mission risk brief: "is it safe to fly a drone here, now?"

Combines the airspace layer (live manned traffic, predicted intrusions, conflicts and anomalies near
the site), India's DigitalSky-style airport zoning (red < 5 km, yellow 5-12 km, green beyond) and the
120 m / 400 ft VLOS ceiling into one explainable score. Perception and land-use assessments can be
attached by the caller when a drone frame or map tile for the site is available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from skyops import config
from skyops.airspace.airports import nearest_airport
from skyops.airspace.anomalies import detect_anomalies
from skyops.airspace.conflicts import detect_conflicts
from skyops.airspace.loader import Airspace, get_airspace
from skyops.airspace.predict import haversine_m, predict_at

RED_ZONE_KM = 5.0        # no drone operations without special permission
YELLOW_ZONE_KM = 12.0    # permission required from ATC
GREEN_CEILING_M = 120.0  # 400 ft AGL ceiling for routine operations
LOW_TRAFFIC_ALT_M = 1500.0  # manned traffic below this altitude matters to a drone


def _within(df: pd.DataFrame, lat: float, lon: float, radius_km: float) -> pd.DataFrame:
    if df.empty:
        return df
    d = haversine_m(df["latitude"].values, df["longitude"].values, np.full(len(df), lat), np.full(len(df), lon)) / 1000.0
    out = df.copy()
    out["dist_km"] = np.round(d, 2)
    return out[out["dist_km"] <= radius_km].sort_values("dist_km")


def mission_risk(lat: float, lon: float, alt_m: float = 100.0, radius_km: float = 10.0, t_idx: int | None = None,
                 airspace: Airspace | None = None, perception: dict | None = None, landuse: dict | None = None) -> dict:
    A = airspace or get_airspace()
    t_idx = A.n_snapshots - 1 if t_idx is None else int(np.clip(t_idx, 0, A.n_snapshots - 1))
    snap = A.snapshot(t_idx)
    pred = predict_at(A, t_idx)
    reasons: list[dict] = []
    score = 0.0

    # --- airport zoning -------------------------------------------------------------------
    ap, ap_km = nearest_airport(lat, lon)
    if ap_km < RED_ZONE_KM:
        zone = "red"
        score += 70
        reasons.append(dict(level="critical", text=f"Inside the {RED_ZONE_KM:.0f} km red zone of {ap['name']} ({ap_km:.1f} km): drone flight prohibited without special permission"))
    elif ap_km < YELLOW_ZONE_KM:
        zone = "yellow"
        score += 30
        reasons.append(dict(level="warning", text=f"Yellow zone: {ap_km:.1f} km from {ap['name']}, ATC permission required"))
    else:
        zone = "green"
        reasons.append(dict(level="info", text=f"Green zone: nearest airport {ap['name']} is {ap_km:.0f} km away"))
    if alt_m > GREEN_CEILING_M:
        score += 25
        reasons.append(dict(level="warning", text=f"Planned altitude {alt_m:.0f} m exceeds the {GREEN_CEILING_M:.0f} m routine ceiling"))

    # --- live traffic --------------------------------------------------------------------
    airborne = snap[~snap["on_ground"]]
    near = _within(airborne, lat, lon, radius_km)
    low = near[near["altitude_m"] < LOW_TRAFFIC_ALT_M]
    if len(low):
        score += min(40, 15 * len(low))
        cs = ", ".join(f"{r.callsign} ({r.altitude_ft:.0f} ft, {r.dist_km:.1f} km)" for r in low.head(4).itertuples())
        reasons.append(dict(level="alert", text=f"{len(low)} manned aircraft below {LOW_TRAFFIC_ALT_M/config.FT_TO_M:.0f} ft within {radius_km:.0f} km: {cs}"))
    elif len(near):
        score += min(15, 3 * len(near))
        reasons.append(dict(level="info", text=f"{len(near)} aircraft within {radius_km:.0f} km, all above {LOW_TRAFFIC_ALT_M/config.FT_TO_M:.0f} ft"))
    else:
        reasons.append(dict(level="info", text=f"No manned traffic within {radius_km:.0f} km"))

    # --- predicted intrusions (next 5 minutes) -------------------------------------------
    intr = _within(pred, lat, lon, radius_km)
    intr = intr[intr["altitude_m"] < LOW_TRAFFIC_ALT_M]
    intruders = sorted(set(intr["icao24"]) - set(low["icao24"]))
    if intruders:
        score += min(25, 10 * len(intruders))
        callsigns = A.df.drop_duplicates("icao24").set_index("icao24")["callsign"]
        names = [str(callsigns.get(i, i)) for i in intruders]
        soonest = int(intr[intr["icao24"].isin(intruders)]["horizon_s"].min())
        reasons.append(dict(level="warning", text=f"{len(intruders)} aircraft predicted to enter the area at low level within {soonest} s: {', '.join(names[:4])}"))

    # --- conflicts and anomalies near the site --------------------------------------------
    conflicts = [c for c in detect_conflicts(snap, pred)
                 if haversine_m(np.array([c["a"]["lat"]]), np.array([c["a"]["lon"]]), np.array([lat]), np.array([lon]))[0] / 1000 <= radius_km * 2]
    if conflicts:
        worst = conflicts[0]
        score += 10 if worst["severity"] in ("critical", "alert") else 5
        reasons.append(dict(level="warning", text=f"{len(conflicts)} separation conflict(s) within {radius_km*2:.0f} km, worst: {worst['a']['callsign']} / {worst['b']['callsign']} ({worst['severity']})"))
    anomalies = [a for a in detect_anomalies(snap, A.snapshot(t_idx - 1) if t_idx > 0 else None)
                 if a["severity"] in ("critical", "alert") and haversine_m(np.array([a["lat"]]), np.array([a["lon"]]), np.array([lat]), np.array([lon]))[0] / 1000 <= radius_km * 2]
    if anomalies:
        score += 10
        reasons.append(dict(level="warning", text=f"{len(anomalies)} abnormal track(s) nearby: " + "; ".join(f"{a['callsign']} {a['detail']}" for a in anomalies[:2])))

    # --- optional perception / land-use layers ---------------------------------------------
    if perception:
        lz = perception.get("landing_zone", {})
        if lz.get("verdict") == "NO-GO":
            score += 25
            reasons.append(dict(level="alert", text=f"Camera: landing zone NO-GO (score {lz.get('score')}), people or vehicles below the drone"))
        elif lz.get("verdict") == "CAUTION":
            score += 10
            reasons.append(dict(level="warning", text=f"Camera: landing zone CAUTION (score {lz.get('score')})"))
        else:
            reasons.append(dict(level="info", text=f"Camera: landing zone clear (score {lz.get('score')})"))
    if landuse:
        if landuse.get("verdict") == "NO-GO":
            score += 15
            reasons.append(dict(level="warning", text=f"Land use: mostly buildings/water below ({landuse.get('hazard_fraction', 0)*100:.0f}% hazard)"))
        else:
            reasons.append(dict(level="info", text=f"Land use: {landuse.get('verdict')} (suitability {landuse.get('score')})"))

    score = float(min(100.0, score))
    verdict = "NO-GO" if score >= 60 else "CAUTION" if score >= 25 else "GO"
    order = {"critical": 0, "alert": 1, "warning": 2, "info": 3}
    reasons.sort(key=lambda r: order[r["level"]])
    cols = ["icao24", "callsign", "altitude_ft", "speed_kt", "dist_km", "latitude", "longitude", "vs_fpm"]
    return dict(
        site=dict(lat=lat, lon=lon, alt_m=alt_m, radius_km=radius_km), t_idx=t_idx, snapshot_time=int(A.times[t_idx]),
        score=round(score, 1), verdict=verdict, zone=zone,
        airport=dict(icao=ap["icao"], name=ap["name"], distance_km=round(ap_km, 1), lat=ap["lat"], lon=ap["lon"]),
        traffic=dict(within_radius=int(len(near)), low_level=int(len(low)), predicted_intruders=len(intruders),
                     aircraft=[{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}
                               for r in near[cols].head(12).to_dict("records")]),
        conflicts_nearby=len(conflicts), anomalies_nearby=len(anomalies), reasons=reasons,
    )
