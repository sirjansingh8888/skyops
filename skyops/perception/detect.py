"""Object detection on drone imagery with Ultralytics YOLO.

Until the VisDrone fine-tune exists (models/visdrone_yolo.pt, produced by skyops.perception.train_visdrone),
a COCO-pretrained YOLO is used and its labels are mapped onto the VisDrone vocabulary so the rest of
the system (landing-zone scoring, geo-projection, UI) is identical before and after fine-tuning.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from skyops import config
from skyops.config import settings

VISDRONE_CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
COCO_TO_VISDRONE = {"person": "pedestrian", "bicycle": "bicycle", "car": "car", "motorcycle": "motor", "bus": "bus", "truck": "truck"}
PEOPLE = {"pedestrian", "people"}
VEHICLES = {"bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"}
COLORS = {"pedestrian": (80, 220, 100), "people": (80, 220, 100), "bicycle": (250, 200, 60), "car": (60, 160, 255),
          "van": (60, 200, 255), "truck": (255, 120, 60), "tricycle": (200, 120, 255), "awning-tricycle": (200, 120, 255),
          "bus": (255, 80, 160), "motor": (250, 230, 80)}


def resolve_weights() -> tuple[str, str]:
    """(weights path, kind). Prefers the fine-tuned VisDrone model when it exists."""
    fine = config.MODELS_DIR / "visdrone_yolo.pt"
    if fine.exists():
        return str(fine), "visdrone-finetuned"
    return settings.yolo_weights, "coco-pretrained"


class Detector:
    def __init__(self, weights: str | None = None, device: str | None = None, conf: float | None = None, imgsz: int = 1280):
        from ultralytics import YOLO  # imported lazily: torch import is slow

        if weights:
            self.weights, self.kind = weights, "custom"
        else:
            self.weights, self.kind = resolve_weights()
        self.model = YOLO(self.weights)
        self.names = self.model.names
        self.device = device or config.resolve_device()
        # the COCO placeholder needs a low threshold to see anything from the air; the fine-tune is confident enough for 0.25
        self.conf = conf if conf is not None else (0.15 if self.kind == "coco-pretrained" else 0.25)
        self.imgsz = imgsz

    def detect(self, image, conf: float | None = None, imgsz: int | None = None) -> list[dict]:
        """image: path, PIL image or BGR numpy array. Returns boxes in pixel coordinates."""
        res = self.model.predict(image, conf=conf or self.conf, imgsz=imgsz or self.imgsz, device=self.device, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            raw = self.names[int(b.cls[0])]
            label = raw if self.kind != "coco-pretrained" else COCO_TO_VISDRONE.get(raw)
            if label is None:
                continue
            out.append(dict(x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1),
                            conf=round(float(b.conf[0]), 3), label=label, raw=raw))
        return out

    @staticmethod
    def annotate(image_bgr: np.ndarray, dets: list[dict], thickness: int = 2) -> np.ndarray:
        img = image_bgr.copy()
        for d in dets:
            c = COLORS.get(d["label"], (200, 200, 200))[::-1]  # RGB -> BGR
            p1, p2 = (int(d["x1"]), int(d["y1"])), (int(d["x2"]), int(d["y2"]))
            cv2.rectangle(img, p1, p2, c, thickness)
            txt = f"{d['label']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (p1[0], max(0, p1[1] - th - 6)), (p1[0] + tw + 4, p1[1]), c, -1)
            cv2.putText(img, txt, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img


def summarize(dets: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    return dict(total=len(dets), people=sum(counts.get(k, 0) for k in PEOPLE),
                vehicles=sum(counts.get(k, 0) for k in VEHICLES), counts=counts)


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    return Detector()


def encode_jpeg(image_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img
