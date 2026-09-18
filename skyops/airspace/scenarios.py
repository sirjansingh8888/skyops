"""Demo scenarios layered on the recorded capture.

Real traffic rarely misbehaves on cue, so for demonstrations SkyOps can inject clearly-labelled simulated
events into the replay. Everything injected is marked origin_country="Simulated" (-> `simulated` column)
or listed in the scenario's `notes`, so the console can show what is real and what is staged.

  baseline    the capture as recorded
  emergency   one real cruising flight squawks 7700 and makes an emergency descent to 10,000 ft
  converging  two simulated jets on crossing tracks at the same flight level (loss of separation on cue)
  combined    both of the above
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from skyops.airspace.loader import Airspace
from skyops.airspace.predict import step_great_circle

SCENARIOS = {
    "baseline": "Recorded traffic only",
    "emergency": "A cruising flight squawks 7700 and descends to 10,000 ft",
    "converging": "Two simulated jets converge at FL360 over central India",
    "combined": "Emergency descent + converging pair",
}
_NOTES: dict[str, list[str]] = {}


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def inject_emergency(raw: pd.DataFrame, near: tuple[float, float] = (19.5, 75.5), start_idx: int = 5,
                     descent_ms: float = -25.0, floor_m: float = 3048.0) -> tuple[pd.DataFrame, str]:
    """Pick the full-length cruising track nearest `near` and turn it into an emergency descent."""
    df = raw.copy()
    times = np.sort(df["snapshot_time"].unique())
    t_start = times[min(start_idx, len(times) - 1)]
    counts = df.groupby("icao24").size()
    at_start = df[(df["snapshot_time"] == t_start) & (df["baro_altitude"] > 9000) & (~df["on_ground"].astype(str).str.lower().eq("true"))]
    at_start = at_start[at_start["icao24"].map(counts) >= len(times) - 1]
    if at_start.empty:
        return df, "emergency: no suitable cruising flight found"
    d = _haversine_km(at_start["latitude"].values, at_start["longitude"].values, near[0], near[1])
    target = at_start.iloc[int(np.argmin(d))]
    icao, alt0 = target["icao24"], float(target["baro_altitude"])
    sel = (df["icao24"] == icao) & (df["snapshot_time"] >= t_start)
    dt = (df.loc[sel, "snapshot_time"] - t_start).astype(float)
    new_alt = np.maximum(alt0 + descent_ms * dt, floor_m)
    df.loc[sel, "baro_altitude"] = new_alt
    df.loc[sel, "geo_altitude"] = new_alt + 150.0
    df.loc[sel, "vertical_rate"] = np.where(new_alt > floor_m, descent_ms, 0.0)
    df.loc[sel, "squawk"] = "7700"
    cs = str(target["callsign"]).strip() or icao
    return df, f"emergency: {cs} ({icao}) squawks 7700 from snapshot {start_idx + 1} and descends at {abs(descent_ms) * 196.85:.0f} ft/min to 10,000 ft"


def inject_converging(raw: pd.DataFrame, cross: tuple[float, float] = (21.6, 78.2), cross_idx: int = 14, speed_ms: float = 235.0,
                      level_m: float = 10972.8) -> tuple[pd.DataFrame, str]:
    """Two simulated jets, eastbound and northbound, reaching `cross` at the same time and level."""
    df = raw.copy()
    times = np.sort(df["snapshot_time"].unique())
    t_cross = float(times[min(cross_idx, len(times) - 1)])
    rows = []
    for icao, callsign, track in (("sim001", "DEMO01", 90.0), ("sim002", "DEMO02", 0.0)):
        for t in times:
            dist_to_go = speed_ms * (t_cross - float(t))  # negative after the crossing point
            lat, lon = step_great_circle(np.array([cross[0]]), np.array([cross[1]]), np.array([(track + 180.0) % 360.0]), np.array([dist_to_go]))
            rows.append(dict(snapshot_time=int(t), icao24=icao, callsign=callsign, origin_country="Simulated", time_position=int(t),
                             last_contact=int(t), longitude=float(lon[0]), latitude=float(lat[0]), baro_altitude=level_m, on_ground=False,
                             velocity=speed_ms, true_track=track, vertical_rate=0.0, sensors=np.nan, geo_altitude=level_m + 150.0,
                             squawk="2000", spi=False, position_source=0))
    out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return out, f"converging: simulated DEMO01 (eastbound) and DEMO02 (northbound) meet at FL360 over {cross[0]:.1f}N {cross[1]:.1f}E at snapshot {cross_idx + 1}"


def build(name: str, raw: pd.DataFrame) -> Airspace:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}' (choose from {', '.join(SCENARIOS)})")
    notes: list[str] = []
    if name in ("emergency", "combined"):
        raw, n = inject_emergency(raw)
        notes.append(n)
    if name in ("converging", "combined"):
        raw, n = inject_converging(raw)
        notes.append(n)
    _NOTES[name] = notes
    return Airspace.from_raw(raw, name=name)


def notes(name: str) -> list[str]:
    return _NOTES.get(name, [])
