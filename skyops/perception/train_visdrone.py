"""Convert the VisDrone-DET subset to YOLO format and fine-tune a YOLO model on it.

Not run automatically. Example on the GPU laptop:
    python -m skyops.perception.train_visdrone --model yolov8s.pt --epochs 30 --imgsz 1024 --device 0
Produces models/visdrone_yolo.pt (best checkpoint) which skyops.perception.detect picks up automatically.

VisDrone annotation line: x,y,w,h,score,category,truncation,occlusion
categories: 0 ignored, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle,
8 awning-tricycle, 9 bus, 10 motor, 11 others. We keep 1..10 (score 0 = ignored regions are dropped).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from skyops import config
from skyops.perception.detect import VISDRONE_CLASSES

YOLO_DIR = config.DATA_PROCESSED / "visdrone_yolo"


def convert(split: str, src: Path = config.VISDRONE_DIR, dst: Path = YOLO_DIR) -> int:
    img_dir, ann_dir = src / f"VisDrone2019-DET-{split}" / "images", src / f"VisDrone2019-DET-{split}" / "annotations"
    out_img, out_lbl = dst / "images" / split, dst / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(img_dir.glob("*.jpg")):
        ann = ann_dir / (img.stem + ".txt")
        if not ann.exists():
            continue
        w, h = Image.open(img).size
        lines = []
        for raw in ann.read_text().splitlines():
            p = raw.strip().split(",")
            if len(p) < 6:
                continue
            x, y, bw, bh, score, cat = (float(v) for v in p[:6])
            if score == 0 or not (1 <= cat <= 10) or bw <= 0 or bh <= 0:
                continue
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{int(cat) - 1} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (out_lbl / (img.stem + ".txt")).write_text("\n".join(lines))
        link = out_img / img.name
        if not link.exists():
            shutil.copy2(img, link)
        n += 1
    return n


def write_yaml(dst: Path = YOLO_DIR) -> Path:
    y = dst / "visdrone.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(VISDRONE_CLASSES))
    y.write_text(f"path: {dst.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8s.pt", help="starting weights (yolov8n/s/m.pt or yolo11n/s.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="'0' for the first GPU, 'cpu' otherwise")
    ap.add_argument("--convert-only", action="store_true")
    a = ap.parse_args()

    n_tr, n_va = convert("train"), convert("val")
    yaml = write_yaml()
    print(f"converted train={n_tr} val={n_va}; dataset yaml at {yaml}")
    if a.convert_only:
        return

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(data=str(yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, project=str(config.ROOT / "runs"),
                name="visdrone", exist_ok=True, patience=10, plots=True)
    best = config.ROOT / "runs" / "visdrone" / "weights" / "best.pt"
    config.MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, config.MODELS_DIR / "visdrone_yolo.pt")
    print("saved", config.MODELS_DIR / "visdrone_yolo.pt")


if __name__ == "__main__":
    main()
