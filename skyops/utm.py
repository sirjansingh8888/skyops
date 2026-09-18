"""UTM desk: registered drone missions as geofences, with live intrusion monitoring.

A mission is a cylinder (centre, radius, ceiling). Registration runs the mission risk brief; a NO-GO brief
is rejected unless forced. While a mission is active, every replay tick checks manned traffic against the
geofence: inside the cylinder (plus buffers) now -> INTRUSION, predicted inside within the look-ahead ->
PREDICTED_INTRUSION with time to entry. Missions persist to data/processed/missions.json.
"""
from __future__ import annotations

import json
import threading
import time
import uuid

import numpy as np
import pandas as pd

from skyops import config
from skyops.airspace.loader import Airspace
from skyops.airspace.predict import haversine_m
from skyops.mission import mission_risk

HORIZONTAL_BUFFER_KM = 1.0   # treat aircraft this close to the fence as inside
VERTICAL_BUFFER_M = 300.0    # 1000 ft above the mission ceiling
_STORE = config.DATA_PROCESSED / "missions.json"
_LOCK = threading.RLock()
_MISSIONS: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _MISSIONS
    with _LOCK:
        if _MISSIONS is None:
            try:
                _MISSIONS = {m["id"]: m for m in json.loads(_STORE.read_text())} if _STORE.exists() else {}
            except Exception:  # noqa: BLE001
                _MISSIONS = {}
        return _MISSIONS


def _save() -> None:
    with _LOCK:
        _STORE.parent.mkdir(parents=True, exist_ok=True)
        _STORE.write_text(json.dumps(list(_load().values()), indent=2))


def list_missions() -> list[dict]:
    return sorted(_load().values(), key=lambda m: m["created"])


def register_mission(name: str, lat: float, lon: float, radius_km: float = 1.0, ceiling_m: float = 120.0, t_idx: int | None = None,
                     force: bool = False, operator: str = "operator") -> dict:
    brief = mission_risk(lat, lon, alt_m=ceiling_m, radius_km=max(radius_km, 5.0), t_idx=t_idx)
    status = "approved" if brief["verdict"] == "GO" else "approved-with-caution" if brief["verdict"] == "CAUTION" else "rejected"
    if status == "rejected" and force:
        status = "forced"
    m = dict(id=uuid.uuid4().hex[:8], name=name or f"mission-{len(_load()) + 1}", lat=float(lat), lon=float(lon), radius_km=float(radius_km),
             ceiling_m=float(ceiling_m), operator=operator, status=status, created=time.time(),
             brief=dict(verdict=brief["verdict"], score=brief["score"], zone=brief["zone"], reasons=brief["reasons"][:4],
                        airport=brief["airport"]))
    with _LOCK:
        _load()[m["id"]] = m
        _save()
    return m


def remove_mission(mission_id: str) -> bool:
    with _LOCK:
        ok = _load().pop(mission_id, None) is not None
        if ok:
            _save()
        return ok


def clear_missions() -> int:
    with _LOCK:
        n = len(_load())
        _load().clear()
        _save()
        return n


def _active(m: dict) -> bool:
    return m["status"] in ("approved", "approved-with-caution", "forced")


def check_intrusions(airspace: Airspace, t_idx: int, predictions: pd.DataFrame | None) -> list[dict]:
    """Alerts for manned aircraft inside, or predicted to enter, any active mission geofence."""
    missions = [m for m in list_missions() if _active(m)]
    if not missions:
        return []
    snap = airspace.snapshot(t_idx)
    air = snap[~snap["on_ground"]]
    alerts: list[dict] = []
    for m in missions:
        fence_km = m["radius_km"] + HORIZONTAL_BUFFER_KM
        top_m = m["ceiling_m"] + VERTICAL_BUFFER_M
        if len(air):
            d = haversine_m(air["latitude"].values, air["longitude"].values, np.full(len(air), m["lat"]), np.full(len(air), m["lon"])) / 1000.0
            inside = air[(d <= fence_km) & (air["altitude_m"].values <= top_m)]
            for r, dist in zip(inside.itertuples(), d[(d <= fence_km) & (air["altitude_m"].values <= top_m)]):
                alerts.append(dict(type="INTRUSION", severity="critical", mission_id=m["id"], mission=m["name"], icao24=r.icao24,
                                   callsign=r.callsign, t_entry_s=0, dist_km=round(float(dist), 2), altitude_ft=round(float(r.altitude_ft)),
                                   lat=float(r.latitude), lon=float(r.longitude),
                                   detail=f"{r.callsign} is inside the geofence of {m['name']} at {r.altitude_ft:.0f} ft ({dist:.1f} km from centre)"))
        if predictions is not None and len(predictions):
            already = {a["icao24"] for a in alerts if a["mission_id"] == m["id"]}
            p = predictions[~predictions["icao24"].isin(already)]
            if len(p):
                dp = haversine_m(p["latitude"].values, p["longitude"].values, np.full(len(p), m["lat"]), np.full(len(p), m["lon"])) / 1000.0
                hit = p[(dp <= fence_km) & (p["altitude_m"].values <= top_m)].assign(dist_km=dp[(dp <= fence_km) & (p["altitude_m"].values <= top_m)])
                for icao, g in hit.groupby("icao24"):
                    first = g.sort_values("horizon_s").iloc[0]
                    cs = airspace.callsign(icao)
                    level = "at landing / very low level" if first.altitude_m < 150 else f"at {first.altitude_m / config.FT_TO_M:.0f} ft"
                    alerts.append(dict(type="PREDICTED_INTRUSION", severity="alert" if first.horizon_s <= 120 else "warning", mission_id=m["id"],
                                       mission=m["name"], icao24=icao, callsign=cs, t_entry_s=int(first.horizon_s), dist_km=round(float(first.dist_km), 2),
                                       altitude_ft=round(float(first.altitude_m) / config.FT_TO_M), lat=float(first.latitude), lon=float(first.longitude),
                                       detail=f"{cs} predicted inside the geofence of {m['name']} in {int(first.horizon_s)} s {level}"))
    order = {"critical": 0, "alert": 1, "warning": 2}
    alerts.sort(key=lambda a: (order[a["severity"]], a["t_entry_s"]))
    return alerts
