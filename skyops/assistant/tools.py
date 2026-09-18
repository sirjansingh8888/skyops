"""Tools the SkyOps assistant can call. Plain Python functions that return compact JSON text.

They are provider-neutral: `declarations()` turns the signatures and docstrings into JSON-schema function
declarations (used by the Gemini agent), and the Claude agent wraps the same functions with the Anthropic
SDK's `beta_tool`. Docstrings matter: the first paragraph is the tool description the model reads and the
`Args:` section supplies the parameter descriptions.
"""
from __future__ import annotations

import inspect
import json
import types
import typing

import numpy as np

from skyops.airspace.anomalies import anomaly_summary, detect_anomalies
from skyops.airspace.conflicts import conflict_summary, detect_conflicts
from skyops.airspace.loader import get_airspace, to_records
from skyops.airspace.predict import haversine_m, predict_at
from skyops.mission import mission_risk


def _j(obj) -> str:
    return json.dumps(obj, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o))


def _t(t_idx: int | None) -> int:
    return get_airspace().clip(t_idx)


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


def mission_risk_brief(lat: float, lon: float, alt_m: float = 100.0, radius_km: float = 10.0, t_idx: int | None = None) -> str:
    """Go / no-go risk assessment for launching a drone at a site now: airport zoning (red under 5 km, yellow under 12 km),
    low-level manned traffic, predicted intrusions, nearby conflicts and anomalies, with reasons.

    Args:
        lat: Launch site latitude in decimal degrees.
        lon: Launch site longitude in decimal degrees.
        alt_m: Planned drone altitude above ground in metres (routine ceiling is 120 m).
        radius_km: Radius around the site to check for traffic.
        t_idx: Snapshot index 0-19. Defaults to the latest snapshot.
    """
    return _j(mission_risk(lat, lon, alt_m, radius_km, _t(t_idx)))


def plan_drone_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float, alt_m: float = 100.0,
                     speed_ms: float = 15.0, endurance_min: float = 30.0, t_idx: int | None = None) -> str:
    """Plan a drone corridor between two points that avoids airport red zones and active geofences and penalises
    yellow zones and low-level manned traffic. Returns the verdict, direct vs planned distance, detour, flight time,
    battery use and the reasons.

    Args:
        start_lat: Start latitude in decimal degrees.
        start_lon: Start longitude in decimal degrees.
        end_lat: Destination latitude in decimal degrees.
        end_lon: Destination longitude in decimal degrees.
        alt_m: Cruise altitude above ground in metres (routine ceiling is 120 m).
        speed_ms: Cruise ground speed in metres per second.
        endurance_min: Battery endurance in minutes.
        t_idx: Snapshot index 0-19. Defaults to the latest snapshot.
    """
    from skyops.route import plan_route

    r = plan_route(start_lat, start_lon, end_lat, end_lon, alt_m, speed_ms, endurance_min, _t(t_idx))
    r["direct"].pop("path", None)
    if r["planned"]:
        wp = r["planned"].pop("path", [])
        r["planned"]["waypoints_latlon"] = [[round(p[1], 4), round(p[0], 4)] for p in wp]
    return _j(r)


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


def landuse_assessment(tile: str | None = None) -> str:
    """Segment an aerial map tile into water / land / road / building / vegetation and score it for emergency landing.

    Args:
        tile: Tile file name (tile_000.png ... tile_071.png); defaults to tile_000.png.
    """
    from skyops.landuse.analyze import analyze_tile

    res = analyze_tile(tile)
    res["assessment"].pop("cell_scores", None)
    return _j(res)


FUNCTIONS = [airspace_overview, list_conflicts, list_anomalies, aircraft_info, traffic_near, mission_risk_brief, plan_drone_route, geofence_status,
             fleet_health, engine_detail, drone_camera_assessment, landuse_assessment]
_BY_NAME = {f.__name__: f for f in FUNCTIONS}
_JSON_TYPES = {int: "integer", float: "number", str: "string", bool: "boolean"}


def _param_docs(fn) -> dict[str, str]:
    doc, out, in_args, last = inspect.getdoc(fn) or "", {}, False, None
    for line in doc.splitlines():
        if line.strip() == "Args:":
            in_args = True
            continue
        if in_args and ":" in line and line.startswith("    ") and not line.startswith("        "):
            name, desc = line.strip().split(":", 1)
            out[name.strip()] = desc.strip()
            last = name.strip()
        elif in_args and last and line.strip():
            out[last] += " " + line.strip()
    return out


def declarations() -> list[dict]:
    """JSON-schema function declarations generated from the tool signatures (Gemini Interactions API format)."""
    decls = []
    for fn in FUNCTIONS:
        hints = typing.get_type_hints(fn)
        docs = _param_docs(fn)
        props, required = {}, []
        for name, p in inspect.signature(fn).parameters.items():
            tp = hints.get(name, str)
            if typing.get_origin(tp) in (typing.Union, types.UnionType):  # Optional[X] -> X
                tp = next(a for a in typing.get_args(tp) if a is not type(None))
            props[name] = {"type": _JSON_TYPES.get(tp, "string"), "description": docs.get(name, name)}
            if p.default is inspect.Parameter.empty:
                required.append(name)
        description = " ".join((inspect.getdoc(fn) or fn.__name__).split("Args:")[0].split())
        decls.append({"type": "function", "name": fn.__name__, "description": description,
                      "parameters": {"type": "object", "properties": props, "required": required}})
    return decls


def run_tool(name: str, arguments: dict | None) -> str:
    """Execute a tool by name with model-supplied arguments; errors come back as JSON so the model can recover."""
    fn = _BY_NAME.get(name)
    if fn is None:
        return _j(dict(error=f"unknown tool {name}"))
    try:
        allowed = inspect.signature(fn).parameters
        args = {k: v for k, v in (arguments or {}).items() if k in allowed and v is not None}
        hints = typing.get_type_hints(fn)
        for k, v in list(args.items()):  # models sometimes send 5.0 for an integer parameter
            tp = hints.get(k)
            if tp in (int, int | None) and isinstance(v, float):
                args[k] = int(v)
        return fn(**args)
    except Exception as e:  # noqa: BLE001
        return _j(dict(error=f"{type(e).__name__}: {e}"))
