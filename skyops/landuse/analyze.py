"""One-call analysis of an aerial tile: segmentation, class fractions, landing suitability, overlay image."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from skyops import config
from skyops.landuse.data import CLASSES, COLORS_RGB, class_fractions, list_tiles, load_tile, rgb_mask_to_index
from skyops.landuse.infer import get_segmenter, landing_assessment, overlay


def tiles() -> list[str]:
    return [p.name for p, _ in list_tiles()]


def _tile_paths(name: str) -> tuple[Path, Path | None]:
    img = config.DUBAI_DIR / "images" / name
    if not img.exists():
        raise FileNotFoundError(f"unknown tile {name}")
    mask = config.DUBAI_DIR / "masks" / name
    return img, (mask if mask.exists() else None)


@lru_cache(maxsize=128)
def _segment_cached(name: str) -> np.ndarray:
    img, _ = _tile_paths(name)
    rgb, _ = load_tile(img)
    return get_segmenter().segment(rgb)


def analyze_tile(name: str | None = None) -> dict:
    name = name or (tiles()[0] if tiles() else None)
    if name is None:
        raise FileNotFoundError("no Dubai tiles on disk: run scripts/download_data.py --only dubai")
    img_path, mask_path = _tile_paths(name)
    idx = _segment_cached(name)
    out = dict(tile=name, width=int(idx.shape[1]), height=int(idx.shape[0]), model=get_segmenter().name,
               fractions=class_fractions(idx), assessment=landing_assessment(idx),
               legend={c: "#%02x%02x%02x" % COLORS_RGB[c] for c in CLASSES})
    if mask_path is not None:
        gt = rgb_mask_to_index(np.array(cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2RGB)))
        out["ground_truth_fractions"] = class_fractions(gt)
        out["pixel_accuracy_vs_gt"] = round(float((gt == idx).mean()), 3)
    return out


def tile_png(name: str, mode: str = "overlay") -> bytes:
    """mode: image | overlay | mask | gt"""
    img_path, mask_path = _tile_paths(name)
    rgb, _ = load_tile(img_path)
    if mode == "image":
        out = rgb
    elif mode == "gt" and mask_path is not None:
        out = np.array(cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2RGB))
    else:
        idx = _segment_cached(name)
        from skyops.landuse.data import index_to_rgb

        out = index_to_rgb(idx) if mode == "mask" else overlay(rgb, idx)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def analyze_upload(image_rgb: np.ndarray) -> tuple[dict, bytes]:
    idx = get_segmenter().segment(image_rgb)
    res = dict(width=int(idx.shape[1]), height=int(idx.shape[0]), model=get_segmenter().name, fractions=class_fractions(idx),
               assessment=landing_assessment(idx), legend={c: "#%02x%02x%02x" % COLORS_RGB[c] for c in CLASSES})
    ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay(image_rgb, idx), cv2.COLOR_RGB2BGR))
    return res, buf.tobytes()
