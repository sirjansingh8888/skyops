"""Landing-zone safety scoring from detections (people / vehicles below the drone).

The frame is divided into a grid. A cell is *occupied* when a detection box (grown by a safety margin)
overlaps it. The recommended touchdown cell is the free cell with the largest clearance from any
occupied cell, preferring cells near the frame centre. Score is 0-100.
"""
from __future__ import annotations

import numpy as np

from skyops.perception.detect import PEOPLE


def landing_zone(dets: list[dict], width: int, height: int, grid: tuple[int, int] = (12, 8),
                 margin_frac: float = 0.15, min_clearance_cells: int = 2) -> dict:
    gx, gy = grid
    occ = np.zeros((gy, gx), dtype=bool)
    people_cells = 0
    for d in dets:
        w, h = d["x2"] - d["x1"], d["y2"] - d["y1"]
        mx, my = w * margin_frac + width * 0.02, h * margin_frac + height * 0.02
        c0 = int(np.clip((d["x1"] - mx) / width * gx, 0, gx - 1))
        c1 = int(np.clip((d["x2"] + mx) / width * gx, 0, gx - 1))
        r0 = int(np.clip((d["y1"] - my) / height * gy, 0, gy - 1))
        r1 = int(np.clip((d["y2"] + my) / height * gy, 0, gy - 1))
        occ[r0:r1 + 1, c0:c1 + 1] = True
        if d["label"] in PEOPLE:
            people_cells += (r1 - r0 + 1) * (c1 - c0 + 1)

    free_frac = float(1.0 - occ.mean())
    # Chebyshev distance from every cell to the nearest occupied cell (grid is tiny, brute force is fine)
    rr, cc = np.indices(occ.shape)
    if occ.any():
        occ_r, occ_c = np.where(occ)
        dist = np.min(np.maximum(np.abs(rr[..., None] - occ_r), np.abs(cc[..., None] - occ_c)), axis=-1)
    else:
        dist = np.full(occ.shape, max(gx, gy))
    # prefer the centre: subtract a small centre-distance penalty
    centre_pen = (np.abs(rr - (gy - 1) / 2) / gy + np.abs(cc - (gx - 1) / 2) / gx) * 0.5
    score_map = dist - centre_pen
    score_map[occ] = -1
    r, c = np.unravel_index(int(np.argmax(score_map)), occ.shape)
    clearance = int(dist[r, c]) if not occ[r, c] else 0
    target = dict(col=int(c), row=int(r), cx=float((c + 0.5) / gx * width), cy=float((r + 0.5) / gy * height),
                  clearance_cells=clearance)

    people_present = any(d["label"] in PEOPLE for d in dets)
    score = 100.0 * free_frac * min(1.0, clearance / max(min_clearance_cells, 1))
    if people_present:
        score *= 0.6
    verdict = "GO" if score >= 60 and clearance >= min_clearance_cells and not people_present else \
              "CAUTION" if score >= 30 else "NO-GO"
    return dict(score=round(score, 1), verdict=verdict, free_fraction=round(free_frac, 3), people_present=people_present,
                grid=dict(cols=gx, rows=gy), occupied=occ.astype(int).tolist(), target=target)
