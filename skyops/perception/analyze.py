"""One-call analysis of a drone frame: detections, ground positions, camera footprint, landing zone."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from skyops.perception import auair
from skyops.perception.detect import Detector, encode_jpeg, get_detector, read_image, summarize
from skyops.perception.geoproject import DEFAULT_HFOV_DEG, DEFAULT_TILT_DEG, GroundProjector, footprint_latlon, project_detections
from skyops.perception.landing import landing_zone


def _default_frame() -> str:
    fr = auair.frames()
    if not fr:
        raise FileNotFoundError("no AU-AIR frames on disk: run scripts/download_data.py --only auair")
    return fr[len(fr) // 2]


@lru_cache(maxsize=256)
def _detect_cached(name: str, conf: float, imgsz: int) -> tuple:
    det = get_detector()
    return tuple(det.detect(read_image(auair.frame_path(name)), conf=conf, imgsz=imgsz))


def analyze_frame(name: str | None = None, use_gt: bool = False, tilt_deg: float = DEFAULT_TILT_DEG,
                  hfov_deg: float = DEFAULT_HFOV_DEG, conf: float | None = None, imgsz: int = 1280,
                  origin: tuple[float, float] | None = None) -> dict:
    """Detections (model or dataset ground truth) with geo-projection and landing-zone score.

    origin: optional (lat, lon) to relocate the mission (the AU-AIR flight was recorded in Aarhus, Denmark);
    detections are placed relative to this point instead of the real GPS fix.
    """
    name = name or _default_frame()
    rec = auair.frame_record(name)
    tel = rec["telemetry"]
    if use_gt:
        dets, source = list(rec["gt"]), "ground-truth"
    else:
        dets, source = [dict(d) for d in _detect_cached(name, conf or get_detector().conf, imgsz)], get_detector().kind
    lat0, lon0 = origin if origin else (tel["lat"], tel["lon"])
    proj = GroundProjector(rec["width"], rec["height"], tel["alt_m"], tel["yaw_deg"], tilt_deg=tilt_deg, hfov_deg=hfov_deg,
                           pitch_deg=tel["pitch_deg"], roll_deg=tel["roll_deg"])
    dets = project_detections(dets, proj, lat0, lon0)
    lz = landing_zone(dets, rec["width"], rec["height"])
    return dict(frame=name, index=auair.frames().index(name), n_frames=len(auair.frames()), width=rec["width"], height=rec["height"],
                source=source, telemetry=tel, origin=dict(lat=lat0, lon=lon0, relocated=origin is not None),
                camera=dict(tilt_deg=tilt_deg, hfov_deg=hfov_deg), detections=dets, summary=summarize(dets),
                footprint=footprint_latlon(proj, lat0, lon0), landing_zone=lz, gt_count=len(rec["gt"]))


def annotated_frame_jpeg(name: str | None = None, use_gt: bool = False, conf: float | None = None, imgsz: int = 1280,
                         draw_landing: bool = True, max_width: int = 1280) -> bytes:
    res = analyze_frame(name, use_gt=use_gt, conf=conf, imgsz=imgsz)
    img = read_image(auair.frame_path(res["frame"]))
    img = Detector.annotate(img, res["detections"])
    if draw_landing:
        lz = res["landing_zone"]
        gx, gy = lz["grid"]["cols"], lz["grid"]["rows"]
        h, w = img.shape[:2]
        overlay = img.copy()
        for r, row in enumerate(lz["occupied"]):
            for c, occ in enumerate(row):
                if occ:
                    cv2.rectangle(overlay, (c * w // gx, r * h // gy), ((c + 1) * w // gx, (r + 1) * h // gy), (0, 0, 255), -1)
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        t = lz["target"]
        color = (80, 220, 80) if lz["verdict"] == "GO" else (0, 200, 255) if lz["verdict"] == "CAUTION" else (0, 0, 255)
        cv2.rectangle(img, (t["col"] * w // gx, t["row"] * h // gy), ((t["col"] + 1) * w // gx, (t["row"] + 1) * h // gy), color, 3)
        cv2.putText(img, f"LANDING {lz['verdict']} {lz['score']:.0f}", (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    if img.shape[1] > max_width:
        s = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return encode_jpeg(img)


def analyze_upload(image_bgr: np.ndarray, conf: float | None = None, imgsz: int = 1280) -> tuple[dict, bytes]:
    det = get_detector()
    dets = det.detect(image_bgr, conf=conf or det.conf, imgsz=imgsz)
    h, w = image_bgr.shape[:2]
    lz = landing_zone(dets, w, h)
    out = Detector.annotate(image_bgr, dets)
    return dict(width=w, height=h, source=det.kind, detections=dets, summary=summarize(dets), landing_zone=lz), encode_jpeg(out)
