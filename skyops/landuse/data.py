"""Dubai aerial semantic-segmentation tiles (MBRSC / Humans in the Loop, CC0).

72 RGB tiles (~800x650) with RGB masks. The mask colours differ from classes.json in the archive; the
values below are the ones actually present in the PNGs (they match the public Kaggle release).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from skyops import config

CLASSES = ["building", "land", "road", "vegetation", "water", "unlabeled"]
COLORS_RGB = {
    "building": (60, 16, 152), "land": (132, 41, 246), "road": (110, 193, 228),
    "vegetation": (254, 221, 58), "water": (226, 169, 41), "unlabeled": (155, 155, 155),
}
PALETTE = np.array([COLORS_RGB[c] for c in CLASSES], dtype=np.uint8)
# how suitable each surface is for an emergency landing (0 = never, 1 = ideal)
LANDING_SUITABILITY = {"building": 0.0, "land": 1.0, "road": 0.35, "vegetation": 0.6, "water": 0.0, "unlabeled": 0.3}


def list_tiles(dubai_dir: Path = config.DUBAI_DIR) -> list[tuple[Path, Path]]:
    imgs = sorted((Path(dubai_dir) / "images").glob("*.png"))
    return [(p, Path(dubai_dir) / "masks" / p.name) for p in imgs if (Path(dubai_dir) / "masks" / p.name).exists()]


def rgb_mask_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    """Nearest-palette-colour class index per pixel (robust to JPEG-like colour noise)."""
    flat = mask_rgb.reshape(-1, 3).astype(np.int32)
    d = ((flat[:, None, :] - PALETTE[None, :, :].astype(np.int32)) ** 2).sum(-1)
    return d.argmin(1).astype(np.uint8).reshape(mask_rgb.shape[:2])


def index_to_rgb(idx: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(idx, 0, len(CLASSES) - 1)]


def load_tile(img_path: Path, mask_path: Path | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    img = np.array(Image.open(img_path).convert("RGB"))
    mask = rgb_mask_to_index(np.array(Image.open(mask_path).convert("RGB"))) if mask_path else None
    return img, mask


def class_fractions(idx: np.ndarray) -> dict[str, float]:
    counts = np.bincount(idx.ravel(), minlength=len(CLASSES))
    total = max(int(counts.sum()), 1)
    return {c: round(float(counts[i]) / total, 4) for i, c in enumerate(CLASSES)}


class TileDataset:
    """Random 256x256 crops with flips. Torch is imported lazily so the API can run without it."""

    def __init__(self, pairs: list[tuple[Path, Path]], size: int = 256, augment: bool = True, repeats: int = 8):
        self.items = [load_tile(i, m) for i, m in pairs]
        self.size, self.augment, self.repeats = size, augment, repeats

    def __len__(self) -> int:
        return len(self.items) * self.repeats

    def __getitem__(self, i: int):
        import torch

        img, mask = self.items[i % len(self.items)]
        h, w = mask.shape
        s = self.size
        rng = np.random.default_rng()
        y0, x0 = (rng.integers(0, h - s + 1), rng.integers(0, w - s + 1)) if self.augment else ((h - s) // 2, (w - s) // 2)
        im, mk = img[y0:y0 + s, x0:x0 + s], mask[y0:y0 + s, x0:x0 + s]
        if self.augment:
            if rng.random() < 0.5:
                im, mk = im[:, ::-1], mk[:, ::-1]
            if rng.random() < 0.5:
                im, mk = im[::-1, :], mk[::-1, :]
        x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return x, torch.from_numpy(np.ascontiguousarray(mk)).long()
