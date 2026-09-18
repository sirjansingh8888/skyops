"""Live ADS-B feed from the OpenSky Network REST API (https://openskynetwork.github.io/opensky-api/rest.html).

Anonymous access works but is rate-limited (about 400 credits/day; one call over the India box costs 4), so
the default poll interval is 30 s and the feed should be switched on for minutes, not hours. With an OpenSky
API client (client id + secret in .env) the limit is ten times higher. The feed keeps the latest 20 snapshots
in the same schema as the recorded capture, so prediction, conflicts, anomalies, the mission brief and the
assistant all work unchanged on live traffic.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import pandas as pd
import requests

from skyops.airspace import loader
from skyops.airspace.loader import RAW_COLUMNS, Airspace
from skyops.config import settings

STATES_URL = "https://opensky-network.org/api/states/all"
TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
INDIA_BBOX = dict(lamin=6.0, lomin=68.0, lamax=36.0, lomax=98.0)
_token_cache: dict = {"value": None, "expires": 0.0}


def _bearer() -> str | None:
    """OAuth2 client-credentials token when SKYOPS_OPENSKY_CLIENT_ID / _SECRET are configured."""
    if not (settings.opensky_client_id and settings.opensky_client_secret):
        return None
    if _token_cache["value"] and time.time() < _token_cache["expires"] - 30:
        return _token_cache["value"]
    r = requests.post(TOKEN_URL, data=dict(grant_type="client_credentials", client_id=settings.opensky_client_id,
                                           client_secret=settings.opensky_client_secret), timeout=20)
    r.raise_for_status()
    tok = r.json()
    _token_cache.update(value=tok["access_token"], expires=time.time() + float(tok.get("expires_in", 1500)))
    return _token_cache["value"]


def fetch_states(bbox: dict | None = None, timeout: float = 25.0) -> pd.DataFrame:
    """One snapshot of state vectors inside bbox, as a DataFrame with the capture's raw columns."""
    headers = {}
    tok = _bearer()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    r = requests.get(STATES_URL, params=bbox or INDIA_BBOX, headers=headers, timeout=timeout)
    if r.status_code == 429:
        raise RuntimeError(f"OpenSky rate limit reached (retry after {r.headers.get('X-Rate-Limit-Retry-After-Seconds', '?')} s)")
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("states") or []
    cols = RAW_COLUMNS[1:]  # everything except snapshot_time, in API order
    df = pd.DataFrame([s[:17] for s in rows], columns=cols)
    df.insert(0, "snapshot_time", int(payload.get("time") or time.time()))
    df["sensors"] = np.nan
    return df[RAW_COLUMNS]


class LiveFeed:
    """Background poller that keeps a rolling window of snapshots registered as the 'live' airspace."""

    def __init__(self, interval_s: float | None = None, window: int = 20, bbox: dict | None = None):
        self.interval_s = float(interval_s or settings.live_interval_s)
        self.bbox = bbox or INDIA_BBOX
        self.buffer: deque[pd.DataFrame] = deque(maxlen=window)
        self.running = False
        self.polls = 0
        self.last_error: str | None = None
        self.last_poll: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def poll_once(self) -> int:
        snap = fetch_states(self.bbox)
        if len(snap) and (not self.buffer or int(snap.snapshot_time.iloc[0]) != int(self.buffer[-1].snapshot_time.iloc[0])):
            self.buffer.append(snap)
        if self.buffer:
            loader.register("live", Airspace.from_raw(pd.concat(list(self.buffer), ignore_index=True), name="live"))
        self.polls += 1
        self.last_poll = time.time()
        self.last_error = None
        return len(snap)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as e:  # noqa: BLE001 - keep the feed alive, surface the error in status()
                self.last_error = str(e)[:300]
            self._stop.wait(self.interval_s)
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self.poll_once()  # fail fast (raises) so the caller can report why live mode is unavailable
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run, name="skyops-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.running = False

    def status(self) -> dict:
        return dict(running=self.running, polls=self.polls, snapshots=len(self.buffer), interval_s=self.interval_s,
                    aircraft=int(len(self.buffer[-1])) if self.buffer else 0, last_poll=self.last_poll, last_error=self.last_error,
                    authenticated=bool(settings.opensky_client_id and settings.opensky_client_secret))


_feed: LiveFeed | None = None


def get_feed() -> LiveFeed:
    global _feed
    if _feed is None:
        _feed = LiveFeed()
    return _feed
