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


def test_scenarios_inject_on_cue():
    from skyops.airspace import scenarios
    from skyops.airspace.anomalies import detect_anomalies
    from skyops.airspace.conflicts import detect_conflicts
    from skyops.airspace.loader import load_raw
    from skyops.airspace.predict import predict_at

    A = scenarios.build("combined", load_raw())
    assert int(A.df.simulated.sum()) == 2 * A.n_snapshots
    cf = detect_conflicts(A.snapshot(13), predict_at(A, 13))
    assert any(c["severity"] == "critical" and "DEMO" in c["a"]["callsign"] + c["b"]["callsign"] for c in cf)
    early = detect_conflicts(A.snapshot(4), predict_at(A, 4))
    assert any("DEMO" in c["a"]["callsign"] and c["severity"] in ("warning", "alert") and c["converging"] for c in early)
    an = detect_anomalies(A.snapshot(8), A.snapshot(7))
    assert any(a["type"] == "EMERGENCY_SQUAWK" and a["severity"] == "critical" for a in an)


def test_utm_geofence_alerts(tmp_path, monkeypatch, airspace):
    from skyops import utm
    from skyops.airspace.predict import predict_at

    monkeypatch.setattr(utm, "_STORE", tmp_path / "missions.json")
    monkeypatch.setattr(utm, "_MISSIONS", None)
    m = utm.register_mission("approach test", 28.545, 77.0, radius_km=4, ceiling_m=120, t_idx=5, force=True)
    assert m["status"] != "rejected" and (tmp_path / "missions.json").exists()
    alerts = utm.check_intrusions(airspace, 5, predict_at(airspace, 5))
    assert alerts and all(a["type"] in ("INTRUSION", "PREDICTED_INTRUSION") for a in alerts)
    far = utm.register_mission("vellore", 12.9692, 79.1559, radius_km=1, ceiling_m=100, t_idx=5)
    assert far["status"] == "approved"
    assert not [a for a in utm.check_intrusions(airspace, 5, predict_at(airspace, 5)) if a["mission_id"] == far["id"]]
    assert utm.remove_mission(m["id"]) and utm.remove_mission(far["id"])


def test_gru_pipeline_shapes(airspace):
    """The learned-predictor code path works end to end with untrained weights (no training here)."""
    torch = pytest.importorskip("torch")
    from skyops.airspace import train_gru as tg

    small = type(airspace)(airspace.df[airspace.df.icao24.isin(airspace.aircraft()[:12])].copy(), name="small")
    X, Y, M, G = tg.build_samples(small)
    assert X.shape[1:] == (tg.K, tg.N_FEAT) and Y.shape[1:] == (len(tg.HORIZONS), 2) and M.shape == Y.shape[:2]
    out = tg.make_model()(torch.tensor(X[:5]))
    assert tuple(out.shape) == (5, len(tg.HORIZONS), 2)


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


def test_assistant_tool_declarations_and_dispatch():
    import json

    from skyops.assistant.tools import FUNCTIONS, declarations, run_tool

    decls = declarations()
    assert len(decls) == len(FUNCTIONS) and all(d["type"] == "function" and d["description"] for d in decls)
    brief = next(d for d in decls if d["name"] == "mission_risk_brief")
    assert brief["parameters"]["required"] == ["lat", "lon"]
    assert brief["parameters"]["properties"]["alt_m"]["type"] == "number" and brief["parameters"]["properties"]["t_idx"]["type"] == "integer"
    out = json.loads(run_tool("mission_risk_brief", {"lat": 12.97, "lon": 79.16, "t_idx": 5.0, "bogus": 1}))
    assert out["verdict"] == "GO"
    assert "error" in json.loads(run_tool("no_such_tool", {}))


def test_gemini_function_calling_loop(monkeypatch):
    """The stateless Interactions loop: a function_call step is executed and answered, then the text is returned."""
    pytest.importorskip("google.genai")
    from types import SimpleNamespace

    from skyops.assistant import gemini_agent

    calls = []

    class FakeStep(SimpleNamespace):
        def model_dump(self):
            return {k: v for k, v in vars(self).items()}

    class FakeInteractions:
        def create(self, **kw):
            calls.append(kw)
            if len(calls) == 1:
                return SimpleNamespace(steps=[FakeStep(type="function_call", id="c1", name="airspace_overview", arguments={"t_idx": 3})], output_text=None)
            assert kw["input"][-1]["type"] == "function_result" and kw["input"][-1]["call_id"] == "c1"
            return SimpleNamespace(steps=[FakeStep(type="model_output", content=[{"type": "text", "text": "All quiet."}])], output_text="All quiet.")

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    a = gemini_agent.GeminiAssistant()
    a.client = SimpleNamespace(interactions=FakeInteractions())
    r = a.ask("How busy is it?", t_idx=3)
    assert r["answer"] == "All quiet." and r["tool_calls"][0]["name"] == "airspace_overview" and len(calls) == 2
    assert calls[0]["store"] is False and calls[0]["tools"] and "SkyOps" in calls[0]["system_instruction"]
    assert [s["type"] for s in a.history] == ["user_input", "function_call", "function_result", "model_output"]


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
    assert c.get("/api/airspace/snapshot/-1").json()["t_idx"] == 19
    try:
        assert c.post("/api/scenario", json={"name": "converging"}).json()["active"] == "converging"
        assert sum(s["simulated"] for s in c.get("/api/airspace/snapshot/10").json()["states"]) == 2
        assert c.post("/api/scenario", json={"name": "nope"}).status_code == 404
    finally:
        c.post("/api/scenario", json={"name": "baseline"})
    m = c.get("/api/metrics").json()
    assert "trajectory" in m and "fleet" in m and m["replay"]["snapshots"] == 20
