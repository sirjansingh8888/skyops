"""Load the OpenSky ADS-B capture into a tidy, replayable form.

The organisers' CSV is a stack of OpenSky *state vectors*: one row per aircraft per snapshot,
20 snapshots roughly 18 s apart over the Indian subcontinent (about 185 aircraft each).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from skyops import config
from skyops.airspace.airports import nearest_airport

STATE_COLUMNS = [
    "icao24", "callsign", "origin_country", "snapshot_time", "t_idx", "t_rel", "latitude", "longitude",
    "altitude_m", "altitude_ft", "on_ground", "velocity", "speed_kt", "true_track", "vertical_rate",
    "vs_fpm", "squawk", "stale_s", "near_airport", "airport_km",
]


def load_states(path: Path | str = config.OPENSKY_CSV) -> pd.DataFrame:
    """Read the raw CSV and add derived columns. Rows are sorted by aircraft then time."""
    df = pd.read_csv(path, dtype={"squawk": "string", "icao24": "string", "callsign": "string"})
    df["callsign"] = df["callsign"].fillna("").str.strip()
    df["callsign"] = df["callsign"].where(df["callsign"] != "", df["icao24"].str.upper())
    df["on_ground"] = df["on_ground"].astype(str).str.lower().eq("true")
    df["squawk"] = df["squawk"].fillna("")

    df["altitude_m"] = df["baro_altitude"].fillna(df["geo_altitude"])
    df.loc[df["on_ground"] & df["altitude_m"].isna(), "altitude_m"] = 0.0
    df["altitude_m"] = df["altitude_m"].fillna(0.0).clip(lower=0.0)
    df["vertical_rate"] = df["vertical_rate"].fillna(0.0)
    df["velocity"] = df["velocity"].fillna(0.0)
    df["true_track"] = df["true_track"].fillna(0.0) % 360.0

    df["altitude_ft"] = df["altitude_m"] / config.FT_TO_M
    df["speed_kt"] = df["velocity"] * config.MS_TO_KT
    df["vs_fpm"] = df["vertical_rate"] * config.MS_TO_FPM
    df["stale_s"] = (df["snapshot_time"] - df["last_contact"]).clip(lower=0)

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    times = np.sort(df["snapshot_time"].unique())
    t_index = {t: i for i, t in enumerate(times)}
    df["t_idx"] = df["snapshot_time"].map(t_index).astype(int)
    df["t_rel"] = (df["snapshot_time"] - times[0]).astype(int)

    # terminal-area flag: within TERMINAL_RADIUS_KM of a known airport
    ap = [nearest_airport(la, lo) for la, lo in zip(df["latitude"].values, df["longitude"].values)]
    df["airport_km"] = [round(d, 1) for _, d in ap]
    df["airport_icao"] = [a["icao"] for a, _ in ap]
    df["near_airport"] = df["airport_km"] <= config.TERMINAL_RADIUS_KM

    return df.sort_values(["icao24", "snapshot_time"]).reset_index(drop=True)


class Airspace:
    """In-memory replay of the capture with per-snapshot and per-aircraft access."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.times: list[int] = [int(t) for t in np.sort(df["snapshot_time"].unique())]
        self.t0 = self.times[0]
        self._by_t = {t: g for t, g in df.groupby("t_idx")}
        self._by_ac = {k: g for k, g in df.groupby("icao24")}

    @classmethod
    def from_csv(cls, path: Path | str = config.OPENSKY_CSV) -> "Airspace":
        return cls(load_states(path))

    # ---- snapshots ------------------------------------------------------------------
    @property
    def n_snapshots(self) -> int:
        return len(self.times)

    def snapshot(self, t_idx: int) -> pd.DataFrame:
        t_idx = int(np.clip(t_idx, 0, self.n_snapshots - 1))
        return self._by_t[t_idx]

    def index_for_seconds(self, t_rel: float) -> int:
        """Snapshot index whose time is nearest to t_rel seconds after the capture start."""
        rel = np.array(self.times) - self.t0
        return int(np.argmin(np.abs(rel - t_rel)))

    def history(self, t_idx: int, n: int = 3) -> pd.DataFrame:
        """The last n states of every aircraft, up to and including snapshot t_idx."""
        lo = max(0, t_idx - n + 1)
        return self.df[(self.df["t_idx"] >= lo) & (self.df["t_idx"] <= t_idx)]

    # ---- aircraft -------------------------------------------------------------------
    def track(self, icao24: str) -> pd.DataFrame:
        return self._by_ac.get(icao24, self.df.iloc[0:0])

    def aircraft(self) -> list[str]:
        return list(self._by_ac.keys())

    # ---- summaries ------------------------------------------------------------------
    def summary(self) -> dict:
        df = self.df
        return dict(
            n_aircraft=int(df["icao24"].nunique()),
            n_snapshots=self.n_snapshots,
            span_s=int(self.times[-1] - self.t0),
            cadence_s=float(np.median(np.diff(self.times))) if self.n_snapshots > 1 else 0.0,
            start_unix=int(self.t0),
            bbox=dict(lat_min=float(df.latitude.min()), lat_max=float(df.latitude.max()),
                      lon_min=float(df.longitude.min()), lon_max=float(df.longitude.max())),
            airborne_per_snapshot=float(df[~df.on_ground].groupby("t_idx").size().mean()),
            countries=df.drop_duplicates("icao24")["origin_country"].value_counts().head(8).to_dict(),
        )


def to_records(df: pd.DataFrame, columns: list[str] | None = None) -> list[dict]:
    """JSON-safe records (NaN -> None, numpy scalars -> python)."""
    cols = [c for c in (columns or STATE_COLUMNS) if c in df.columns]
    out = df[cols].astype(object).where(df[cols].notna(), None)
    recs = out.to_dict("records")
    for r in recs:
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                r[k] = int(v)
            elif isinstance(v, (np.floating,)):
                r[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                r[k] = bool(v)
    return recs


@lru_cache(maxsize=1)
def get_airspace() -> Airspace:
    """Process-wide cached capture (used by the API)."""
    return Airspace.from_csv()
