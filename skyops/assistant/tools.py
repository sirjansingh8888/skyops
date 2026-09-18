"""Tools the SkyOps assistant can call. Each returns compact JSON text built from the live modules.

Docstrings matter: the SDK turns them into the tool descriptions Claude reads.
"""
from __future__ import annotations

import json

import numpy as np
from anthropic import beta_tool

from skyops.airspace.anomalies import anomaly_summary, detect_anomalies
from skyops.airspace.conflicts import conflict_summary, detect_conflicts
from skyops.airspace.loader import get_airspace, to_records
from skyops.airspace.predict import haversine_m, predict_at
from skyops.mission import mission_risk


def _j(obj) -> str:
    return json.dumps(obj, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o))


def _t(t_idx: int | None) -> int:
    A = get_airspace()
    return A.n_snapshots - 1 if t_idx is None else int(max(0, min(A.n_snapshots - 1, t_idx)))


@beta_tool
def airspace_overview(t_idx: int | None = None) -> str:
    """Summary of the monitored airspace at a replay snapshot: aircraft counts, conflict and anomaly totals.

    Args:
        t_idx: Snapshot index 0-19 in the replay (each about 18 s apart). Defaults to the latest snapshot.
    """
    A = get_airspace()
    t = _t(t_idx)
    snap = A.snapshot(t)
    pred = predict_at(A, t)
    return _j(dict(t_idx=t, snapshot_time=int(A.times[t]), aircraft=int(len(snap)), airborne=int((~snap.on_ground).sum()),
                   conflicts=conflict_summary(detect_conflicts(snap, pred)),
                   anomalies=anomaly_summary(detect_anomalies(snap, A.snapshot(t - 1) if t > 0 else None)),
                   summary=A.summary()))


@beta_tool
def list_conflicts(t_idx: int | None = None, max_items: int = 10) -> str:
    """Predicted losses of separation between manned aircraft (5 NM / 1000 ft, 3 NM near airports), worst first.

    Args:
        t_idx: Snapshot index 0-19. Defaults to the latest snapshot.
        max_items: Maximum number of conflicts to return.
    """
    A = get_airspace()
    t = _t(t_idx)
    cf = detect_conflicts(A.snapshot(t), predict_at(A, t))
    return _j(dict(t_idx=t, summary=conflict_summary(cf), conflicts=cf[:max_items]))


@beta_tool
def list_anomalies(t_idx: int | None = None, max_items: int = 15) -> str:
    """Abnormal aircraft behaviour flags (emergency squawks, extreme climb/descent, speed outliers, stale contact, jumps).

    Args:
        t_idx: Snapshot index 0-19. Defaults to the latest snapshot.
        max_items: Maximum number of flags to return.
    """
    A = get_airspace()
    t = _t(t_idx)
    an = detect_anomalies(A.snapshot(t), A.snapshot(t - 1) if t > 0 else None)
    return _j(dict(t_idx=t, summary=anomaly_summary(an), anomalies=an[:max_items]))


@beta_tool
def aircraft_info(callsign_or_icao: str, t_idx: int | None = None) -> str:
    """Current state and 5-minute predicted path of one aircraft, looked up by callsign (e.g. IGO1477) or ICAO24 hex.

    Args:
        callsign_or_icao: Callsign or ICAO24 address, case-insensitive.
        t_idx: Snapshot index 0-19. Defaults to the latest snapshot.
    """
    A = get_airspace()
    t = _t(t_idx)
    key = callsign_or_icao.strip().upper()
    snap = A.snapshot(t)
    row = snap[(snap.callsign.str.upper() == key) | (snap.icao24.str.upper() == key)]
    if row.empty:
        return _j(dict(error=f"no aircraft '{callsign_or_icao}' in snapshot {t}"))
    icao = row.iloc[0].icao24
    pred = predict_at(A, t)
    path = pred[pred.icao24 == icao][["horizon_s", "latitude", "longitude", "altitude_m"]].round(4).to_dict("records")
    track = A.track(icao)[["t_rel", "latitude", "longitude", "altitude_ft", "speed_kt", "vs_fpm"]].round(1).to_dict("records")
    return _j(dict(state=to_records(row)[0], predicted_path=path, history=track[-6:]))


@beta_tool
def traffic_near(lat: float, lon: float, radius_km: float = 25.0, t_idx: int | None = None, max_items: int = 15) -> str:
    """Manned aircraft within a radius of a point, nearest first, with altitude and distance.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        radius_km: Search radius in kilometres.
        t_idx: Snapshot index 0-19. Defaults to the latest snapshot.
        max_items: Maximum number of aircraft to return.
    """
    A = get_airspace()
    t = _t(t_idx)
    snap = A.snapshot(t)
    d = haversine_m(snap.latitude.values, snap.longitude.values, np.full(len(snap), lat), np.full(len(snap), lon)) / 1000
    near = snap.assign(dist_km=np.round(d, 2))
    near = near[near.dist_km <= radius_km].sort_values("dist_km").head(max_items)
    cols = ["callsign", "icao24", "origin_country", "altitude_ft", "speed_kt", "vs_fpm", "true_track", "on_ground", "dist_km"]
    return _j(dict(t_idx=t, count=int(len(near)), aircraft=near[cols].to_dict("records")))


@beta_tool
def mission_risk_brief(lat: float, lon: float, alt_m: float = 100.0, radius_km: float = 10.0, t_idx: int | None = None) -> str:
    """Go / no-go risk assessment for launching a drone at a site now: airport zoning (red < 5 km, yellow < 12 km),
    low-level manned traffic, predicted intrusions, nearby conflicts and anomalies, with reasons.

    Args:
        lat: Launch site latitude in decimal degrees.
        lon: Launch site longitude in decimal degrees.
        alt_m: Planned drone altitude above ground in metres (routine ceiling is 120 m).
        radius_km: Radius around the site to check for traffic.
        t_idx: Snapshot index 0-19. Defaults to the latest snapshot.
    """
    return _j(mission_risk(lat, lon, alt_m, radius_km, _t(t_idx)))


@beta_tool
def fleet_health(subset: str = "FD001", state: str | None = None, max_items: int = 10) -> str:
    """Remaining-useful-life estimates for the turbofan fleet (NASA C-MAPSS test engines), worst engines first.

    Args:
        subset: C-MAPSS subset: FD001, FD002, FD003 or FD004.
        state: Optional filter: ground, maintenance, watch or healthy.
        max_items: Maximum number of engines to return.
    """
    from skyops.fleet.predict import fleet_status

    fs = fleet_status(subset)
    eng = [e for e in fs["engines"] if state is None or e["state"] == state]
    return _j(dict(subset=subset, model=fs["model"], counts=fs["counts"], rmse_vs_truth=fs["rmse_vs_truth"], engines=eng[:max_items]))


@beta_tool
def engine_detail(unit: int, subset: str = "FD001") -> str:
    """Sensor trends and rolling RUL estimate for one engine of the fleet.

    Args:
        unit: Engine unit number within the subset (1-based).
        subset: C-MAPSS subset: FD001, FD002, FD003 or FD004.
    """
    from skyops.fleet.predict import engine_history, fleet_status

    fs = fleet_status(subset)
    rec = next((e for e in fs["engines"] if e["unit"] == unit), None)
    h = engine_history(subset, unit)
    return _j(dict(engine=rec, cycles=len(h["cycles"]), rul_last5=h["rul"]["p50"][-5:], sensors_last={k: v[-1] for k, v in h["sensors"].items()}))


@beta_tool
def drone_camera_assessment(frame: str | None = None, use_ground_truth: bool = False) -> str:
    """Analyse a drone camera frame (AU-AIR flight): detections of people/vehicles, their ground positions from the
    drone's GPS/altitude/heading, and a landing-zone GO/CAUTION/NO-GO score.

    Args:
        frame: Frame file name from the flight; defaults to the middle of the flight.
        use_ground_truth: Use the dataset's annotated boxes instead of the detector.
    """
    from skyops.perception.analyze import analyze_frame

    res = analyze_frame(frame, use_gt=use_ground_truth)
    res.pop("footprint", None)
    res["detections"] = res["detections"][:12]
    res["landing_zone"].pop("occupied", None)
    return _j(res)


@beta_tool
def landuse_assessment(tile: str | None = None) -> str:
    """Segment an aerial map tile into water / land / road / building / vegetation and score it for emergency landing.

    Args:
        tile: Tile file name (tile_000.png ... tile_071.png); defaults to tile_000.png.
    """
    from skyops.landuse.analyze import analyze_tile

    res = analyze_tile(tile)
    res["assessment"].pop("cell_scores", None)
    return _j(res)


@beta_tool
def geofence_status(t_idx: int | None = None) -> str:
    """Registered drone missions (geofences) and current intrusion alerts: manned aircraft inside a geofence or
    predicted to enter one, with time to entry. Also reports which airspace is active (recorded, scenario or live).

    Args:
        t_idx: Snapshot index 0-19. Defaults to the latest snapshot.
    """
    from skyops import utm
    from skyops.airspace import loader, scenarios

    A = get_airspace()
    t = _t(t_idx)
    missions = [{k: m[k] for k in ("id", "name", "lat", "lon", "radius_km", "ceiling_m", "status")} | {"brief": m["brief"]["verdict"]} for m in utm.list_missions()]
    return _j(dict(t_idx=t, airspace=loader.active_name(), scenario_notes=scenarios.notes(loader.active_name()), missions=missions,
                   alerts=utm.check_intrusions(A, t, predict_at(A, t))))


ALL_TOOLS = [airspace_overview, list_conflicts, list_anomalies, aircraft_info, traffic_near, mission_risk_brief, geofence_status,
             fleet_health, engine_detail, drone_camera_assessment, landuse_assessment]
