"""Fine-tune a U-Net (ResNet-18 encoder, ImageNet weights) on the 72 Dubai tiles.

Not run automatically. Example (GPU laptop, ~5-10 minutes):
    python -m skyops.landuse.train --epochs 40 --device cuda --out models/landuse_unet.pt
CPU is possible with --epochs 8 --size 192 but slow. Holds out 12 tiles for validation and reports
per-class IoU. The first run downloads the ImageNet encoder weights (~45 MB).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from skyops import config
from skyops.landuse.data import CLASSES, TileDataset, list_tiles


def iou_per_class(pred: np.ndarray, target: np.ndarray, n: int = len(CLASSES)) -> list[float]:
    out = []
    for c in range(n):
        p, t = pred == c, target == c
        union = np.logical_or(p, t).sum()
        out.append(float(np.logical_and(p, t).sum() / union) if union else float("nan"))
    return out


def train(epochs: int, size: int, batch: int, lr: float, device: str, out: Path, holdout: int = 12, seed: int = 0) -> dict:
    import segmentation_models_pytorch as smp
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    pairs = list_tiles()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    val_pairs = [pairs[i] for i in order[:holdout]]
    tr_pairs = [pairs[i] for i in order[holdout:]]
    dl_tr = DataLoader(TileDataset(tr_pairs, size=size, augment=True), batch_size=batch, shuffle=True, num_workers=0)
    dl_va = DataLoader(TileDataset(val_pairs, size=size, augment=False, repeats=1), batch_size=batch, shuffle=False, num_workers=0)

    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl_tr))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=CLASSES.index("unlabeled"))
    best, history, t0 = -1.0, [], time.time()
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss)
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for x, y in dl_va:
                preds.append(model(x.to(device)).argmax(1).cpu().numpy())
                tgts.append(y.numpy())
        ious = iou_per_class(np.concatenate(preds), np.concatenate(tgts))
        miou = float(np.nanmean(ious[:-1]))  # ignore 'unlabeled'
        history.append(dict(epoch=ep + 1, train_loss=tl / max(len(dl_tr), 1), miou=miou, iou={c: ious[i] for i, c in enumerate(CLASSES)}))
        print(f"epoch {ep+1}/{epochs} loss {tl/max(len(dl_tr),1):.3f} mIoU {miou:.3f}")
        if miou > best:
            best = miou
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(), encoder="resnet18", classes=CLASSES, size=size), out)
    manifest = dict(kind="unet-resnet18", best_miou=best, epochs=epochs, size=size, seconds=round(time.time() - t0, 1),
                    history=history, val_tiles=[p[0].name for p in val_pairs])
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(config.MODELS_DIR / "landuse_unet.pt"))
    a = ap.parse_args()
    m = train(a.epochs, a.size, a.batch, a.lr, a.device, Path(a.out))
    print("best mIoU", round(m["best_miou"], 3))


if __name__ == "__main__":
    main()
