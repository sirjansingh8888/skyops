"""Land-use segmentation inference plus a landing-suitability assessment.

Uses models/landuse_unet.pt (from skyops.landuse.train) when present. Before training, a transparent
colour-rule segmenter keeps the endpoint and UI working; the API reports which one produced the mask.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from skyops import config
from skyops.config import settings
from skyops.landuse.data import CLASSES, LANDING_SUITABILITY, class_fractions, index_to_rgb


class HeuristicSegmenter:
    """Brightness/saturation rules for the Dubai tiles. Crude, but honest and instant.

    The tiles are nearly grey: per-class median value (V) is water 0.10, vegetation 0.33, road 0.42,
    unlabeled 0.59, building 0.66, land (sand) 0.74, and only water has real saturation (0.33).
    So the placeholder separates surfaces by brightness bands, with a saturation test for water and
    a local-texture test (buildings have sharper edges than sand) to split the two brightest classes.
    """

    name = "colour-rules"

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        s, v = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
        grey = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        texture = cv2.blur(np.abs(cv2.Laplacian(grey, cv2.CV_32F)), (15, 15)) / 255.0
        idx = np.full(v.shape, CLASSES.index("unlabeled"), dtype=np.uint8)
        idx[(v < 0.22) | ((s > 0.28) & (v < 0.4))] = CLASSES.index("water")
        idx[(v >= 0.22) & (v < 0.37)] = CLASSES.index("vegetation")
        idx[(v >= 0.37) & (v < 0.5)] = CLASSES.index("road")
        bright = v >= 0.5
        # per-class median texture: building 0.21, unlabeled 0.15, road/vegetation/land ~0.10-0.11, water 0.05
        idx[bright & (texture >= 0.15)] = CLASSES.index("building")
        idx[bright & (texture < 0.15)] = CLASSES.index("land")
        return cv2.medianBlur(idx, 9)


class UNetSegmenter:
    name = "unet-resnet18"

    def __init__(self, weights: Path, device: str):
        import segmentation_models_pytorch as smp
        import torch

        ck = torch.load(weights, map_location=device)
        self.model = smp.Unet(ck.get("encoder", "resnet18"), encoder_weights=None, classes=len(ck.get("classes", CLASSES)))
        self.model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()})
        self.model.eval().to(device)
        self.device = device
        self.torch = torch

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        t = self.torch
        h, w = image_rgb.shape[:2]
        H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32  # U-Net needs multiples of 32
        pad = np.zeros((H, W, 3), dtype=np.uint8)
        pad[:h, :w] = image_rgb
        x = t.from_numpy(pad).permute(2, 0, 1).float() / 255.0
        x = (x - t.tensor([0.485, 0.456, 0.406])[:, None, None]) / t.tensor([0.229, 0.224, 0.225])[:, None, None]
        with t.no_grad():
            out = self.model(x[None].to(self.device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
        return out[:h, :w]


@lru_cache(maxsize=1)
def get_segmenter():
    w = config.ROOT / settings.landuse_weights
    if w.exists():
        try:
            return UNetSegmenter(w, settings.device)
        except Exception as e:  # noqa: BLE001
            print(f"[landuse] could not load {w}: {e}; using colour rules")
    return HeuristicSegmenter()


def landing_assessment(idx: np.ndarray, grid: tuple[int, int] = (8, 8)) -> dict:
    """Suitability score 0-100 from surface classes, plus the best grid cell for a touchdown."""
    h, w = idx.shape
    gy, gx = grid
    suit = np.vectorize(lambda c: LANDING_SUITABILITY[CLASSES[c]])(idx.astype(int)) if idx.size else np.zeros_like(idx, float)
    cells = np.zeros((gy, gx))
    for r in range(gy):
        for c in range(gx):
            cell = suit[r * h // gy:(r + 1) * h // gy, c * w // gx:(c + 1) * w // gx]
            cells[r, c] = float(cell.mean()) if cell.size else 0.0
    r, c = np.unravel_index(int(np.argmax(cells)), cells.shape)
    frac = class_fractions(idx)
    hazards = round(frac["building"] + frac["water"], 3)
    score = round(100.0 * float(suit.mean()), 1)
    verdict = "GO" if cells[r, c] >= 0.8 and hazards < 0.5 else "CAUTION" if cells[r, c] >= 0.5 else "NO-GO"
    return dict(score=score, verdict=verdict, hazard_fraction=hazards, fractions=frac,
                best_cell=dict(row=int(r), col=int(c), suitability=round(float(cells[r, c]), 3),
                               cx=float((c + 0.5) * w / gx), cy=float((r + 0.5) * h / gy)),
                grid=dict(rows=gy, cols=gx), cell_scores=np.round(cells, 3).tolist())


def overlay(image_rgb: np.ndarray, idx: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = index_to_rgb(idx)
    return (image_rgb.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
