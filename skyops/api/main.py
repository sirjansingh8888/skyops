"""SkyOps HTTP API and static UI host.

Run:  uvicorn skyops.api.main:app --reload --port 8000     (or: python -m skyops.api.main)
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import io
import json
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from skyops import config, utm
from skyops.airspace import loader, scenarios
from skyops.airspace.airports import AIRPORTS
from skyops.airspace.anomalies import anomaly_summary, detect_anomalies
from skyops.airspace.conflicts import conflict_summary, detect_conflicts
from skyops.airspace.loader import get_airspace, to_records
from skyops.airspace.predict import evaluate, predict_at
from skyops.config import settings
from skyops.mission import mission_risk

app = FastAPI(title="SkyOps", version="0.2.0", description="AI control tower for the drone era: airspace, perception, land use, fleet health.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------------------- health
@app.get("/api/health")
def health() -> dict:
    A = get_airspace()
    return dict(status="ok", airspace=A.name, snapshots=A.n_snapshots, aircraft=A.summary()["n_aircraft"], device=config.resolve_device(),
                data=dict(visdrone=(config.VISDRONE_DIR / "VisDrone2019-DET-val").exists(), auair=(config.AUAIR_DIR / "annotations.json").exists(),
                          dubai=(config.DUBAI_DIR / "images").exists(), cmapss=config.CMAPSS_DIR.exists()),
                models=dict(visdrone_yolo=(config.MODELS_DIR / "visdrone_yolo.pt").exists(), landuse_unet=(config.ROOT / settings.landuse_weights).exists(),
                            rul_lgbm=(config.MODELS_DIR / "rul_lgbm" / "manifest.json").exists(), traj_gru=(config.MODELS_DIR / "traj_gru.pt").exists()))


# ----------------------------------------------------------------------------- airspace
@app.get("/api/airspace/summary")
def airspace_summary() -> dict:
    A = get_airspace()
    return dict(**A.summary(), times=[int(t - A.t0) for t in A.times], airports=AIRPORTS, scenario=loader.active_name(),
                scenario_notes=scenarios.notes(loader.active_name()))


@app.get("/api/airspace/snapshot/{t_idx}")
def airspace_snapshot(t_idx: int, predict: bool = True, horizons: str = "60,120,180,300") -> dict:
    """Everything the map needs for one tick: states, predicted paths, conflicts, anomalies, UTM alerts. t_idx=-1 means latest."""
    A = get_airspace()
    t = A.clip(t_idx)
    snap = A.snapshot(t)
    hz = tuple(int(h) for h in horizons.split(",") if h.strip())
    pred = predict_at(A, t, horizons=hz) if predict else None
    paths: dict[str, list] = {}
    if pred is not None:
        for icao, g in pred.groupby("icao24"):
            paths[icao] = [[round(float(r.longitude), 5), round(float(r.latitude), 5), round(float(r.altitude_m)), int(r.horizon_s)] for r in g.itertuples()]
    conflicts = detect_conflicts(snap, pred)
    anomalies = detect_anomalies(snap, A.snapshot(t - 1) if t > 0 else None)
    return dict(t_idx=t, t_rel=int(A.times[t] - A.t0), snapshot_time=int(A.times[t]), n_snapshots=A.n_snapshots, airspace=A.name,
                states=to_records(snap), predicted=paths, conflicts=conflicts, conflict_summary=conflict_summary(conflicts),
                anomalies=anomalies, anomaly_summary=anomaly_summary(anomalies), utm_alerts=utm.check_intrusions(A, t, pred),
                missions=utm.list_missions())


@app.get("/api/airspace/tracks")
def airspace_tracks() -> dict:
    """Full recorded trajectory per aircraft: [lon, lat, alt_m, t_rel] lists (for trails)."""
    A = get_airspace()
    out = {}
    for icao, g in A.df.groupby("icao24"):
        out[icao] = [[round(float(r.longitude), 5), round(float(r.latitude), 5), round(float(r.altitude_m)), int(r.t_rel)] for r in g.itertuples()]
    return dict(tracks=out)


@app.get("/api/airspace/aircraft/{icao24}")
def airspace_aircraft(icao24: str, t_idx: int | None = None) -> dict:
    A = get_airspace()
    tr = A.track(icao24)
    if tr.empty:
        raise HTTPException(404, f"unknown aircraft {icao24}")
    pred = predict_at(A, A.clip(t_idx))
    return dict(track=to_records(tr), predicted=pred[pred.icao24 == icao24].round(5).to_dict("records"))


@lru_cache(maxsize=1)
def _evaluation() -> dict:
    return evaluate(get_airspace("baseline"))


@app.get("/api/airspace/evaluate")
def airspace_evaluate() -> dict:
    return _evaluation()


# ----------------------------------------------------------------------------- scenarios and live feed
class ScenarioRequest(BaseModel):
    name: str


@app.get("/api/scenarios")
def scenario_list() -> dict:
    return dict(active=loader.active_name(), scenarios=[dict(name=k, description=v) for k, v in scenarios.SCENARIOS.items()],
                notes=scenarios.notes(loader.active_name()))


@app.post("/api/scenario")
def scenario_set(req: ScenarioRequest) -> dict:
    from skyops.airspace.live import get_feed

    if req.name not in scenarios.SCENARIOS:
        raise HTTPException(404, f"unknown scenario {req.name}")
    get_feed().stop()
    A = loader.set_active(req.name)
    return dict(active=A.name, notes=scenarios.notes(req.name), n_snapshots=A.n_snapshots)


class LiveRequest(BaseModel):
    interval_s: float | None = Field(None, ge=10, le=300)


@app.post("/api/live/start")
def live_start(req: LiveRequest) -> dict:
    from skyops.airspace.live import get_feed

    feed = get_feed()
    if req.interval_s:
        feed.interval_s = req.interval_s
    try:
        feed.start()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"live feed unavailable: {e}")
    loader.set_active("live")
    return feed.status()


@app.post("/api/live/stop")
def live_stop() -> dict:
    from skyops.airspace.live import get_feed

    get_feed().stop()
    if loader.active_name() == "live":
        loader.set_active("baseline")
    return get_feed().status()


@app.get("/api/live/status")
def live_status() -> dict:
    from skyops.airspace.live import get_feed

    return dict(**get_feed().status(), active=loader.active_name() == "live")


# ----------------------------------------------------------------------------- mission brief and UTM desk
class MissionRequest(BaseModel):
    lat: float
    lon: float
    alt_m: float = 100.0
    radius_km: float = Field(10.0, gt=0, le=100)
    t_idx: int | None = None
    frame: str | None = None      # attach a drone-camera assessment
    tile: str | None = None       # attach a land-use assessment
    use_gt: bool = False


@app.post("/api/mission/brief")
def mission_brief(req: MissionRequest) -> dict:
    perception = landuse = None
    if req.frame:
        from skyops.perception.analyze import analyze_frame

        perception = analyze_frame(req.frame, use_gt=req.use_gt, origin=(req.lat, req.lon))
        perception.pop("footprint", None)
    if req.tile:
        from skyops.landuse.analyze import analyze_tile

        landuse = analyze_tile(req.tile)["assessment"]
    out = mission_risk(req.lat, req.lon, req.alt_m, req.radius_km, req.t_idx, perception=perception, landuse=landuse)
    if perception:
        perception["landing_zone"].pop("occupied", None)
        out["perception"] = dict(frame=perception["frame"], source=perception["source"], summary=perception["summary"], landing_zone=perception["landing_zone"])
    if landuse:
        landuse.pop("cell_scores", None)
        out["landuse"] = landuse
    return out


class RegisterRequest(BaseModel):
    name: str = ""
    lat: float
    lon: float
    radius_km: float = Field(1.0, gt=0, le=50)
    ceiling_m: float = Field(120.0, gt=0, le=1000)
    t_idx: int | None = None
    force: bool = False


@app.get("/api/utm/missions")
def utm_missions() -> dict:
    return dict(missions=utm.list_missions())


@app.post("/api/utm/missions")
def utm_register(req: RegisterRequest) -> dict:
    return utm.register_mission(req.name, req.lat, req.lon, req.radius_km, req.ceiling_m, req.t_idx, req.force)


@app.delete("/api/utm/missions/{mission_id}")
def utm_remove(mission_id: str) -> dict:
    if not utm.remove_mission(mission_id):
        raise HTTPException(404, f"unknown mission {mission_id}")
    return dict(removed=mission_id)


@app.delete("/api/utm/missions")
def utm_clear() -> dict:
    return dict(removed=utm.clear_missions())


# ----------------------------------------------------------------------------- fleet
@app.get("/api/fleet/{subset}")
def fleet(subset: str) -> dict:
    from skyops.fleet.features import SUBSETS
    from skyops.fleet.predict import fleet_status

    if subset not in SUBSETS:
        raise HTTPException(404, f"unknown subset {subset}")
    return fleet_status(subset)


@app.get("/api/fleet/{subset}/{unit}")
def fleet_engine(subset: str, unit: int) -> dict:
    from skyops.fleet.predict import engine_history

    h = engine_history(subset, unit)
    if not h["cycles"]:
        raise HTTPException(404, f"unknown engine {subset}-{unit}")
    return h


# ----------------------------------------------------------------------------- drone perception
@app.get("/api/drone/frames")
def drone_frames() -> dict:
    from skyops.perception import auair

    track = auair.mission_track()
    return dict(n=len(track), frames=track)


@app.get("/api/drone/frame/{name}")
def drone_frame(name: str, gt: bool = False, tilt: float = Query(45.0, ge=5, le=90), hfov: float = Query(69.0, ge=20, le=120),
                conf: float | None = Query(None, ge=0.01, le=0.9), origin_lat: float | None = None, origin_lon: float | None = None) -> dict:
    from skyops.perception import auair
    from skyops.perception.analyze import analyze_frame

    if name not in auair.load_index():
        raise HTTPException(404, f"unknown frame {name}")
    origin = (origin_lat, origin_lon) if origin_lat is not None and origin_lon is not None else None
    return analyze_frame(name, use_gt=gt, tilt_deg=tilt, hfov_deg=hfov, conf=conf, origin=origin)


@app.get("/api/drone/frame/{name}/image")
def drone_frame_image(name: str, gt: bool = False, annotated: bool = True, conf: float | None = Query(None, ge=0.01, le=0.9)):
    from skyops.perception import auair
    from skyops.perception.analyze import annotated_frame_jpeg

    if name not in auair.load_index():
        raise HTTPException(404, f"unknown frame {name}")
    if not annotated:
        return FileResponse(auair.frame_path(name), media_type="image/jpeg")
    return Response(annotated_frame_jpeg(name, use_gt=gt, conf=conf), media_type="image/jpeg")


@app.post("/api/drone/detect")
async def drone_detect(file: UploadFile = File(...), conf: float | None = Query(None, ge=0.01, le=0.9)) -> dict:
    import base64

    import cv2

    from skyops.perception.analyze import analyze_upload

    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not an image")
    res, jpg = analyze_upload(img, conf=conf)
    res["image_jpeg_base64"] = base64.b64encode(jpg).decode()
    return res


# ----------------------------------------------------------------------------- land use
@app.get("/api/landuse/tiles")
def landuse_tiles() -> dict:
    from skyops.landuse.analyze import tiles

    return dict(tiles=tiles())


@app.get("/api/landuse/tile/{name}")
def landuse_tile(name: str) -> dict:
    from skyops.landuse.analyze import analyze_tile

    try:
        return analyze_tile(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/api/landuse/tile/{name}/image")
def landuse_tile_image(name: str, mode: str = Query("overlay", pattern="^(image|overlay|mask|gt)$")):
    from skyops.landuse.analyze import tile_png

    try:
        return Response(tile_png(name, mode), media_type="image/png")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/landuse/segment")
async def landuse_segment(file: UploadFile = File(...)) -> dict:
    import base64

    from PIL import Image

    from skyops.landuse.analyze import analyze_upload

    img = np.array(Image.open(io.BytesIO(await file.read())).convert("RGB"))
    res, png = analyze_upload(img)
    res["image_png_base64"] = base64.b64encode(png).decode()
    return res


# ----------------------------------------------------------------------------- metrics (model card)
@lru_cache(maxsize=1)
def _replay_stats() -> dict:
    A = get_airspace("baseline")
    tot = dict(critical=0, alert=0, warning=0, marginal=0)
    pairs: set[str] = set()
    n_anom = 0
    for t in range(A.n_snapshots):
        for c in detect_conflicts(A.snapshot(t), predict_at(A, t)):
            tot[c["severity"]] += 1
            if c["severity"] != "marginal":
                pairs.add(c["id"])
        n_anom += sum(a["severity"] != "info" for a in detect_anomalies(A.snapshot(t), A.snapshot(t - 1) if t else None))
    return dict(snapshots=A.n_snapshots, conflict_flags=tot, distinct_conflict_pairs=len(pairs), anomaly_flags=n_anom)


@lru_cache(maxsize=1)
def _landuse_baseline_accuracy() -> float | None:
    try:
        from skyops.landuse.analyze import analyze_tile, tiles

        acc = [analyze_tile(t).get("pixel_accuracy_vs_gt") for t in tiles()[:12]]
        acc = [a for a in acc if a is not None]
        return round(float(np.mean(acc)), 3) if acc else None
    except Exception:  # noqa: BLE001
        return None


@app.get("/api/metrics")
def metrics() -> dict:
    """Numbers for the pitch: what each model is, how it was evaluated, and how well it does."""
    out: dict = dict(trajectory=dict(**_evaluation(), note="error vs the aircraft's real later positions in the capture"), replay=_replay_stats())
    gru_json = config.MODELS_DIR / "traj_gru.json"
    if gru_json.exists():
        out["trajectory"]["gru"] = json.loads(gru_json.read_text())
    man = config.MODELS_DIR / "rul_lgbm" / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        out["fleet"] = dict(model="LightGBM point + conformalised quantile offsets", test=m["test"], validation=m["val"],
                            top_features=m["top_features"][:8], trained_seconds=m["trained_seconds"], baseline_knn_rmse_fd001=22.2)
    else:
        out["fleet"] = dict(model="k-NN fallback (no training)", note="run python -m skyops.fleet.train")
    perc: dict = dict(model="visdrone-finetuned YOLO" if (config.MODELS_DIR / "visdrone_yolo.pt").exists() else "COCO-pretrained YOLOv8n (placeholder)")
    yolo_json = config.MODELS_DIR / "visdrone_yolo.json"
    if yolo_json.exists() and (config.MODELS_DIR / "visdrone_yolo.pt").exists():
        y = json.loads(yolo_json.read_text())
        perc.update(model=f"{y.get('model', 'YOLO')} fine-tuned on VisDrone ({y.get('train_images')} train / {y.get('val_images')} val images, {y.get('imgsz')} px)",
                    epochs=y["epochs_run"], map50=y["map50"], map50_95=y["map50_95"], precision=y["precision"], recall=y["recall"])
    transfer = config.MODELS_DIR / "auair_transfer_eval.json"
    if transfer.exists():
        perc["auair_transfer"] = json.loads(transfer.read_text())
    out["perception"] = perc
    lu_json = (config.ROOT / settings.landuse_weights).with_suffix(".json")
    if lu_json.exists():
        lm = json.loads(lu_json.read_text())
        out["landuse"] = dict(model="U-Net ResNet-18", best_miou=lm["best_miou"], epochs=lm["epochs"], per_class_iou=lm["history"][-1]["iou"])
    else:
        out["landuse"] = dict(model="colour rules (placeholder)", pixel_accuracy=_landuse_baseline_accuracy())
    return out


# ----------------------------------------------------------------------------- assistant
class ChatRequest(BaseModel):
    message: str
    t_idx: int | None = None
    reset: bool = False
    context: dict | None = None


@app.post("/api/assistant/chat")
def assistant_chat(req: ChatRequest) -> dict:
    from skyops import assistant

    st = assistant.status()
    if not st["available"]:
        return dict(error=f"Assistant offline: put {st['key_env']}=... in skyops/.env and restart the server (see HANDOFF.md).",
                    answer=None, tool_calls=[])
    a = assistant.get_assistant()
    if req.reset:
        a.reset()
    return a.ask(req.message, t_idx=req.t_idx, context=req.context)


@app.get("/api/assistant/status")
def assistant_status() -> dict:
    from skyops import assistant

    return assistant.status()


# ----------------------------------------------------------------------------- static UI (must be last)
if config.WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")


def main() -> None:
    import uvicorn

    uvicorn.run("skyops.api.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
