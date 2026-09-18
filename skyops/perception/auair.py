"""AU-AIR multimodal UAV dataset: frames with synced flight telemetry and object boxes.

annotations.json holds one record per frame: GPS (latitude / 'longtitude'), altitude (millimetres),
body velocities linear_x/y/z (m/s), attitude angle_phi/theta/psi (roll/pitch/yaw, radians) and boxes
{top,left,height,width,class}. Only frames that exist on disk (the downloaded subset) are indexed.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from skyops import config

CATEGORIES = ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]
AUAIR_TO_VISDRONE = {"Human": "pedestrian", "Car": "car", "Truck": "truck", "Van": "van", "Motorbike": "motor",
                     "Bicycle": "bicycle", "Bus": "bus", "Trailer": "truck"}


def _alt_m(v: float) -> float:
    return v / 1000.0 if v > 500 else v  # the file stores millimetres


@lru_cache(maxsize=1)
def load_index(auair_dir: Path = config.AUAIR_DIR) -> dict[str, dict]:
    path = Path(auair_dir) / "annotations.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    on_disk = {p.name for p in (Path(auair_dir) / "images").glob("*.jpg")}
    idx = {r["image_name"]: r for r in data["annotations"] if r["image_name"] in on_disk}
    return dict(sorted(idx.items()))


def frames() -> list[str]:
    return list(load_index().keys())


def frame_path(name: str) -> Path:
    return config.AUAIR_DIR / "images" / name


def telemetry(rec: dict) -> dict:
    t = rec.get("time", {})
    try:
        ts = datetime(int(t["year"]), int(t["month"]), int(t["day"]), int(t["hour"]), int(t["min"]), int(t["sec"])).isoformat()
    except Exception:  # noqa: BLE001
        ts = None
    vx, vy, vz = float(rec.get("linear_x", 0)), float(rec.get("linear_y", 0)), float(rec.get("linear_z", 0))
    return dict(
        lat=float(rec["latitude"]), lon=float(rec["longtitude"]), alt_m=round(_alt_m(float(rec["altitude"])), 2),
        vx=round(vx, 3), vy=round(vy, 3), vz=round(vz, 3), speed_ms=round(math.hypot(vx, vy), 3),
        roll_deg=round(math.degrees(float(rec.get("angle_phi", 0))), 2),
        pitch_deg=round(math.degrees(float(rec.get("angle_theta", 0))), 2),
        yaw_deg=round(math.degrees(float(rec.get("angle_psi", 0))) % 360.0, 2),
        platform=rec.get("platform", "UAV"), time=ts,
    )


def gt_boxes(rec: dict) -> list[dict]:
    out = []
    for b in rec.get("bbox", []):
        cls = CATEGORIES[int(b["class"])] if 0 <= int(b["class"]) < len(CATEGORIES) else "Unknown"
        out.append(dict(x1=float(b["left"]), y1=float(b["top"]), x2=float(b["left"] + b["width"]),
                        y2=float(b["top"] + b["height"]), label=AUAIR_TO_VISDRONE.get(cls, "car"), raw=cls, conf=1.0))
    return out


def frame_record(name: str) -> dict:
    rec = load_index()[name]
    return dict(name=name, width=int(rec.get("image_width:", rec.get("image_width", 1920))),
                height=int(rec.get("image_height", 1080)), telemetry=telemetry(rec), gt=gt_boxes(rec))


def mission_track() -> list[dict]:
    """GPS/altitude trace of the downloaded subset, in frame order (for the mission map)."""
    return [dict(name=n, **{k: v for k, v in telemetry(r).items() if k in ("lat", "lon", "alt_m", "yaw_deg", "speed_ms")})
            for n, r in load_index().items()]
