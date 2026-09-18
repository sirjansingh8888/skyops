"""Rule-based stand-in for the LLM assistant.

Hackathon WiFi and free-tier quotas fail at the worst moment. When no API key is configured, or the LLM call
fails, this router answers from the *same tools*: it picks a tool from keywords in the question, runs it, and
formats the result with a template. No language model is involved, so answers are plainer, but every number is
live and correct, and the preset questions in the console always work.
"""
from __future__ import annotations

import json
import re
import time

from skyops.assistant.tools import run_tool

_CALLSIGN = re.compile(r"\b([A-Z]{2,4}\d{1,4}[A-Z]{0,2}|DEMO0[12])\b")


def _load(name: str, **args) -> dict:
    return json.loads(run_tool(name, {k: v for k, v in args.items() if v is not None}))


def _reasons(items: list[dict], n: int = 4) -> str:
    return "\n".join(f"- {r['text']}" for r in items[:n])


def _mission(ctx: dict, t_idx) -> tuple[str, str]:
    site = ctx.get("mission_site")
    if not site:
        return "mission_risk_brief", "Set a mission site in the Mission tab first, then ask again."
    r = _load("mission_risk_brief", lat=site["lat"], lon=site["lon"], alt_m=site.get("alt_m", 100.0), radius_km=site.get("radius_km", 10.0), t_idx=t_idx)
    return "mission_risk_brief", (f"{r['verdict']} for a launch at {site['lat']:.4f}, {site['lon']:.4f} (risk {r['score']}/100, {r['zone']} zone, "
                                  f"{r['airport']['name']} {r['airport']['distance_km']} km away).\n{_reasons(r['reasons'])}")


def _route(ctx: dict, t_idx) -> tuple[str, str]:
    site, dest = ctx.get("mission_site"), ctx.get("route_destination")
    if not (site and dest):
        return "plan_drone_route", "Set a start (mission site) and a destination in the Mission tab's route planner, then ask again."
    r = _load("plan_drone_route", start_lat=site["lat"], start_lon=site["lon"], end_lat=dest["lat"], end_lon=dest["lon"], alt_m=site.get("alt_m", 100.0), t_idx=t_idx)
    p = r.get("planned")
    head = f"{r['verdict']}: direct {r['direct']['length_km']} km"
    if p:
        head += f", planned corridor {p['length_km']} km (+{p['detour_pct']} %), {p['eta_min']} min, battery {p['battery_pct']} %."
    return "plan_drone_route", f"{head}\n{_reasons(r['reasons'])}"


def _conflicts(ctx: dict, t_idx) -> tuple[str, str]:
    r = _load("list_conflicts", t_idx=t_idx, max_items=5)
    s = r["summary"]
    lines = [f"{s['critical']} critical, {s['alert']} alert, {s['warning']} warning and {s['marginal']} marginal (altitude-noise) pairs at snapshot {r['t_idx'] + 1}."]
    for c in r["conflicts"]:
        when = "separation lost now" if c["t_loss_s"] == 0 else f"loss in {c['t_loss_s']:.0f} s"
        lines.append(f"- {c['a']['callsign']} / {c['b']['callsign']}: {c['severity']}, {when}, CPA {c['cpa_nm']} NM, {c['v_sep_ft']} ft apart vertically, {'converging' if c['converging'] else 'diverging'}")
    if any(c["severity"] in ("critical", "alert") for c in r["conflicts"]):
        lines.append("First action: resolve the top pair with a level change or a heading change for the aircraft with more room.")
    return "list_conflicts", "\n".join(lines)


def _anomalies(ctx: dict, t_idx) -> tuple[str, str]:
    r = _load("list_anomalies", t_idx=t_idx, max_items=30)
    real = [a for a in r["anomalies"] if a["severity"] != "info"][:6]
    s = r["summary"]
    if not real:
        return "list_anomalies", f"No abnormal behaviour at snapshot {r['t_idx'] + 1} ({s['info']} coverage-gap notes only)."
    return "list_anomalies", f"{len(real)} abnormal track(s) at snapshot {r['t_idx'] + 1}:\n" + "\n".join(f"- {a['callsign']}: {a['type'].lower().replace('_', ' ')} ({a['severity']}), {a['detail']}" for a in real)


def _fleet(ctx: dict, t_idx) -> tuple[str, str]:
    r = _load("fleet_health", subset="FD001", max_items=6)
    c = r["counts"]
    lines = [f"Fleet FD001 ({r['model']}, RMSE {r['rmse_vs_truth']} cycles): {c['ground']} to ground, {c['maintenance']} need maintenance, {c['watch']} to watch, {c['healthy']} healthy."]
    lines += [f"- {e['engine_id']}: RUL {e['rul_p50']} cycles (80 % band {e['rul_p10']} to {e['rul_p90']}): {e['action']}" for e in r["engines"]]
    return "fleet_health", "\n".join(lines)


def _camera(ctx: dict, t_idx) -> tuple[str, str]:
    r = _load("drone_camera_assessment", frame=ctx.get("current_drone_frame"))
    lz, s, t = r["landing_zone"], r["summary"], r["telemetry"]
    seen = ", ".join(f"{v} {k}" for k, v in s["counts"].items()) or "nothing"
    return "drone_camera_assessment", (f"Landing zone {lz['verdict']} (score {lz['score']}/100, {lz['free_fraction'] * 100:.0f} % of the view clear). "
                                       f"The {r['source']} detector sees {seen} from {t['alt_m']} m, heading {t['yaw_deg']:.0f} deg. "
                                       f"Best touchdown cell: column {lz['target']['col'] + 1}, row {lz['target']['row'] + 1}.")


def _landuse(ctx: dict, t_idx) -> tuple[str, str]:
    r = _load("landuse_assessment", tile=ctx.get("current_landuse_tile"))
    a = r["assessment"]
    top = ", ".join(f"{k} {v * 100:.0f} %" for k, v in sorted(r["fractions"].items(), key=lambda kv: -kv[1])[:3])
    return "landuse_assessment", f"{a['verdict']} for an emergency landing on {r['tile']}: suitability {a['score']}/100, hazards {a['hazard_fraction'] * 100:.0f} %. Surface: {top} ({r['model']})."


def _geofence(ctx: dict, t_idx) -> tuple[str, str]:
    r = _load("geofence_status", t_idx=t_idx)
    if not r["missions"]:
        return "geofence_status", "No missions are registered. Register one in the Mission tab and the tower will watch its geofence."
    lines = [f"{len(r['missions'])} registered mission(s), {len(r['alerts'])} alert(s) on the {r['airspace']} airspace."]
    lines += [f"- {a['detail']}" for a in r["alerts"][:5]]
    return "geofence_status", "\n".join(lines)


def _aircraft(callsign: str, t_idx) -> tuple[str, str]:
    r = _load("aircraft_info", callsign_or_icao=callsign, t_idx=t_idx)
    if "error" in r:
        return "aircraft_info", r["error"]
    s = r["state"]
    return "aircraft_info", (f"{s['callsign']} ({s['origin_country']}): {s['altitude_ft']:.0f} ft, {s['speed_kt']:.0f} kt, {s['vs_fpm']:+.0f} ft/min, heading {s['true_track']:.0f} deg, "
                             f"squawk {s['squawk'] or 'none'}, {s['airport_km']} km from the nearest airport{' (simulated)' if s.get('simulated') else ''}.")


def _overview(ctx: dict, t_idx) -> tuple[str, str]:
    r = _load("airspace_overview", t_idx=t_idx)
    c, a = r["conflicts"], r["anomalies"]
    return "airspace_overview", (f"Snapshot {r['t_idx'] + 1}: {r['aircraft']} aircraft, {r['airborne']} airborne. Conflicts: {c['critical']} critical, {c['alert']} alert, "
                                 f"{c['warning']} warning. Anomalies: {a['critical'] + a['alert'] + a['warning']} flagged. Ask about conflicts, anomalies, a launch, "
                                 f"a route, the camera, land use, geofences or engines.")


_ROUTES = [
    (("route", "deliver", "corridor", "detour", "from a to b"), _route),
    (("launch", "fly", "safe", "mission site", "take off", "takeoff", "go/no-go", "no-go"), _mission),
    (("geofence", "intrusion", "registered"), _geofence),
    (("conflict", "separation", "pairs", "collision"), _conflicts),
    (("anomal", "abnormal", "emergency", "squawk", "weird", "strange"), _anomalies),
    (("engine", "fleet", "ground", "rul", "maintenance", "remaining useful"), _fleet),
    (("camera", "frame", "landing zone", "detect", "people below", "drone sees"), _camera),
    (("land use", "landuse", "tile", "surface", "segment", "emergency landing"), _landuse),
]


class OfflineAssistant:
    provider = "offline"
    model = "rule-based (no LLM)"

    def reset(self) -> None:
        pass

    def ask(self, text: str, t_idx: int | None = None, context: dict | None = None, note: str | None = None) -> dict:
        t0, ctx, q = time.time(), dict(context or {}), text.lower()
        handler = next((fn for keys, fn in _ROUTES if any(k in q for k in keys)), None)
        try:
            m = _CALLSIGN.search(text.upper())
            if handler is None and m:
                tool, answer = _aircraft(m.group(1), t_idx)
            else:
                tool, answer = (handler or _overview)(ctx, t_idx)
        except Exception as e:  # noqa: BLE001
            tool, answer = "error", f"The offline assistant could not answer that: {type(e).__name__}: {e}"
        prefix = note or "Offline mode (no language model): a templated answer from the live tools."
        return dict(question=text, answer=f"{prefix}\n\n{answer}", tool_calls=[dict(name=tool, input={})], provider=self.provider,
                    model=self.model, seconds=round(time.time() - t0, 1))
