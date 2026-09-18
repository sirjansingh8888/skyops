#!/usr/bin/env python
"""How well does a detector find the objects annotated in the AU-AIR drone frames? (inference only)

AU-AIR is *not* used for training, so this is an honest transfer test of the VisDrone fine-tune on the
footage the demo actually shows. Reports class-agnostic recall and precision at IoU >= 0.3 over evenly
spaced frames, for the COCO-pretrained placeholder and for models/visdrone_yolo.pt when present.

    python scripts/eval_auair.py --frames 120
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skyops import config  # noqa: E402
from skyops.perception import auair  # noqa: E402
from skyops.perception.detect import Detector, read_image  # noqa: E402


def iou(a: dict, b: dict) -> float:
    x1, y1, x2, y2 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"]), min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(det: Detector, names: list[str], conf: float, imgsz: int, thr: float = 0.3) -> dict:
    tp = n_gt = n_pred = 0
    for n in names:
        rec = auair.frame_record(n)
        preds = det.detect(read_image(auair.frame_path(n)), conf=conf, imgsz=imgsz)
        used: set[int] = set()
        for g in rec["gt"]:
            best, bi = 0.0, -1
            for i, p in enumerate(preds):
                if i in used:
                    continue
                v = iou(g, p)
                if v > best:
                    best, bi = v, i
            if best >= thr:
                tp += 1
                used.add(bi)
        n_gt += len(rec["gt"])
        n_pred += len(preds)
    return dict(frames=len(names), gt_boxes=n_gt, predictions=n_pred, recall=round(tp / max(n_gt, 1), 3), precision=round(tp / max(n_pred, 1), 3))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()
    allf = auair.frames()
    names = [allf[i] for i in np.linspace(0, len(allf) - 1, min(a.frames, len(allf))).astype(int)]
    out = {"coco-pretrained yolov8n": evaluate(Detector(weights="yolov8n.pt"), names, a.conf, a.imgsz)}
    out["coco-pretrained yolov8n"]["note"] = "COCO labels, all classes counted"
    fine = config.MODELS_DIR / "visdrone_yolo.pt"
    if fine.exists():
        out["visdrone fine-tuned"] = evaluate(Detector(weights=str(fine)), names, a.conf, a.imgsz)
    print(json.dumps(out, indent=2))
    (config.MODELS_DIR / "auair_transfer_eval.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
