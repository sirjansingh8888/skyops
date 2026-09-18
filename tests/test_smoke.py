"""Smoke tests: every layer loads and produces sane output; the API answers. Run: pytest -q"""
from __future__ import annotations

import pytest

from skyops import config

HAS_OPENSKY = config.OPENSKY_CSV.exists()
HAS_AUAIR = (config.AUAIR_DIR / "annotations.json").exists()
HAS_DUBAI = (config.DUBAI_DIR / "images").exists()
HAS_CMAPSS = config.CMAPSS_DIR.exists()

pytestmark = pytest.mark.skipif(not HAS_OPENSKY, reason="run scripts/download_data.py --only opensky first")


@pytest.fixture(scope="session")
def airspace():
    from skyops.airspace.loader import Airspace

    return Airspace.from_csv()


def test_airspace_loads(airspace):
    s = airspace.summary()
    assert s["n_aircraft"] > 100 and s["n_snapshots"] == 20
    snap = airspace.snapshot(5)
    assert {"icao24", "callsign", "latitude", "longitude", "altitude_m", "near_airport"} <= set(snap.columns)


def test_prediction_and_evaluation(airspace):
    from skyops.airspace.predict import evaluate, predict_at

    pred = predict_at(airspace, 5)
    assert set(pred.horizon_s.unique()) == set(config.PREDICTION_HORIZONS_S)
    assert pred.latitude.between(-90, 90).all() and pred.longitude.between(-180, 180).all()
    ev = evaluate(airspace, horizons=(60,))
    assert ev["horizons"][60]["n"] > 500 and ev["horizons"][60]["median_m"] < 600


def test_conflicts_and_anomalies(airspace):
    from skyops.airspace.anomalies import detect_anomalies
    from skyops.airspace.conflicts import conflict_summary, detect_conflicts
    from skyops.airspace.predict import predict_at

    snap = airspace.snapshot(5)
    cf = detect_conflicts(snap, predict_at(airspace, 5))
    s = conflict_summary(cf)
    assert s["total"] == len(cf) and all(c["severity"] in ("critical", "alert", "warning", "marginal") for c in cf)
    an = detect_anomalies(snap, airspace.snapshot(4))
    assert all({"type", "severity", "icao24", "detail"} <= set(a) for a in an)


def test_mission_brief(airspace):
    from skyops.mission import mission_risk

    r = mission_risk(28.61, 77.05, 100, 10, 5, airspace=airspace)
    assert r["verdict"] in ("GO", "CAUTION", "NO-GO") and 0 <= r["score"] <= 100 and r["reasons"]
    g = mission_risk(12.97, 79.16, 100, 15, 5, airspace=airspace)
    assert g["zone"] == "green"


@pytest.mark.skipif(not HAS_CMAPSS, reason="C-MAPSS not downloaded")
def test_fleet_status():
    from skyops.fleet.predict import engine_history, fleet_status

    fs = fleet_status("FD001")
    assert fs["n_engines"] == 100 and sum(fs["counts"].values()) == 100
    e = fs["engines"][0]
    assert e["rul_p10"] <= e["rul_p50"] <= e["rul_p90"]
    h = engine_history("FD001", e["unit"])
    assert len(h["cycles"]) == e["cycles_observed"]


@pytest.mark.skipif(not HAS_AUAIR, reason="AU-AIR not downloaded")
def test_perception_ground_truth_path():
    from skyops.perception.analyze import analyze_frame

    r = analyze_frame(use_gt=True)
    assert r["source"] == "ground-truth" and r["landing_zone"]["verdict"] in ("GO", "CAUTION", "NO-GO")
    assert len(r["footprint"]) > 8 and r["telemetry"]["alt_m"] > 0


@pytest.mark.skipif(not HAS_DUBAI, reason="Dubai tiles not downloaded")
def test_landuse():
    from skyops.landuse.analyze import analyze_tile

    r = analyze_tile("tile_000.png")
    assert abs(sum(r["fractions"].values()) - 1) < 1e-3 and r["assessment"]["verdict"] in ("GO", "CAUTION", "NO-GO")


def test_api_roundtrip():
    from fastapi.testclient import TestClient

    from skyops.api.main import app

    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "ok"
    snap = c.get("/api/airspace/snapshot/3").json()
    assert snap["t_idx"] == 3 and len(snap["states"]) > 100 and "conflicts" in snap
    brief = c.post("/api/mission/brief", json=dict(lat=19.08, lon=72.88, alt_m=100, radius_km=10, t_idx=3)).json()
    assert brief["verdict"] in ("GO", "CAUTION", "NO-GO")
    assert c.get("/api/assistant/status").status_code == 200
