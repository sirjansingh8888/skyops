"""Learned trajectory predictor: a small GRU that corrects dead reckoning.

Dead reckoning is already strong on straight cruise; most of its error comes from turns and speed changes.
Instead of predicting positions from scratch, the GRU reads the last K states of an aircraft and predicts the
*residual* (east, north, km) of the dead-reckoning forecast at 60 / 120 / 180 s. With the organisers' capture
alone (214 tracks x 20 points) the gain is modest; record more traffic first for a real improvement:

    python scripts/record_opensky.py --minutes 30 --interval 20
    python -m skyops.airspace.train_gru --csv data/raw/opensky/opensky_trajectories.csv data/raw/opensky/recorded_*.csv

Not run automatically. CPU is enough (about a minute per 10k samples). Produces models/traj_gru.pt, which
`skyops.airspace.predict.get_predictor()` picks up, and prints a held-out comparison against dead reckoning.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from skyops import config
from skyops.airspace.loader import Airspace, derive, load_raw
from skyops.airspace.predict import DeadReckoningPredictor, _interp_track

K = 4
HORIZONS = (60, 120, 180)
N_FEAT = 8


def enu_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    east = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * config.R_EARTH_M / 1000.0
    north = np.radians(lat - lat0) * config.R_EARTH_M / 1000.0
    return east, north


def history_features(hist: pd.DataFrame) -> np.ndarray:
    """(K, N_FEAT) features of the last K states, relative to the most recent one."""
    h = hist.tail(K)
    last = h.iloc[-1]
    e, n = enu_km(h.latitude.values.astype(float), h.longitude.values.astype(float), float(last.latitude), float(last.longitude))
    trk = np.radians(h.true_track.values.astype(float))
    f = np.stack([(h.time_position.values.astype(float) - float(last.time_position)) / 60.0, e / 10.0, n / 10.0,
                  (h.altitude_m.values.astype(float) - float(last.altitude_m)) / 1000.0, h.velocity.values.astype(float) / 250.0,
                  np.sin(trk), np.cos(trk), h.vertical_rate.values.astype(float) / 10.0], axis=1)
    return f.astype(np.float32)


def build_samples(airspace: Airspace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X (N,K,F), Y (N,H,2) residual km, M (N,H) mask, G (N,) group id per aircraft (for a leak-free split)."""
    dr = DeadReckoningPredictor()
    X, Y, M, G = [], [], [], []
    for gi, (icao, tr) in enumerate(airspace.df.groupby("icao24")):
        tr = tr[~tr.on_ground].drop_duplicates("time_position").sort_values("time_position")
        if len(tr) < K + 2:
            continue
        for i in range(K - 1, len(tr) - 1):
            hist = tr.iloc[i - K + 1:i + 1]
            last = hist.iloc[-1]
            pred = dr.predict(hist, HORIZONS).set_index("horizon_s")
            y = np.zeros((len(HORIZONS), 2), np.float32)
            m = np.zeros(len(HORIZONS), np.float32)
            for hi, h in enumerate(HORIZONS):
                actual = _interp_track(tr, float(last.time_position) + h)
                if actual is None:
                    continue
                ae, an = enu_km(np.array([actual[0]]), np.array([actual[1]]), float(last.latitude), float(last.longitude))
                pe, pn = enu_km(np.array([pred.loc[h, "latitude"]]), np.array([pred.loc[h, "longitude"]]), float(last.latitude), float(last.longitude))
                y[hi] = (ae[0] - pe[0], an[0] - pn[0])
                m[hi] = 1.0
            if m.sum() == 0:
                continue
            X.append(history_features(hist))
            Y.append(y)
            M.append(m)
            G.append(gi)
    return np.stack(X), np.stack(Y), np.stack(M), np.array(G)


def make_model(hidden: int = 64):
    import torch.nn as nn

    class ResidualGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(N_FEAT, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, len(HORIZONS) * 2))

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1]).view(-1, len(HORIZONS), 2)

    return ResidualGRU()


def train(csvs: list[str], epochs: int, out: Path, seed: int = 0) -> dict:
    import torch

    torch.manual_seed(seed)
    parts = [build_samples(Airspace(derive(load_raw(p)), name=Path(p).stem)) for p in csvs]
    offset, Xs, Ys, Ms, Gs = 0, [], [], [], []
    for X, Y, M, G in parts:
        Xs.append(X); Ys.append(Y); Ms.append(M); Gs.append(G + offset)
        offset += int(G.max()) + 1
    X, Y, M, G = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ms), np.concatenate(Gs)
    rng = np.random.default_rng(seed)
    groups = rng.permutation(np.unique(G))
    val_groups = set(groups[:max(1, len(groups) // 5)].tolist())
    va = np.array([g in val_groups for g in G])
    Xt, Yt, Mt = (torch.tensor(a[~va]) for a in (X, Y, M))
    Xv, Yv, Mv = (torch.tensor(a[va]) for a in (X, Y, M))
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    huber = torch.nn.HuberLoss(reduction="none", delta=0.5)

    def loss_fn(p, y, m):
        return (huber(p, y).sum(-1) * m).sum() / m.sum().clamp(min=1)

    def err_m(p, y, m):  # mean residual error in metres per horizon
        d = torch.linalg.norm(p - y, dim=-1) * 1000.0
        return [float((d[:, i] * m[:, i]).sum() / m[:, i].sum().clamp(min=1)) for i in range(len(HORIZONS))]

    best, best_state, t0 = float("inf"), None, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(model(Xt[b]), Yt[b], Mt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(Xv), Yv, Mv))
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        gru_err = err_m(model(Xv), Yv, Mv)
        dr_err = err_m(torch.zeros_like(Yv), Yv, Mv)
    report = dict(samples=int(len(X)), val_samples=int(va.sum()), horizons=list(HORIZONS), dead_reckoning_mean_m=dr_err, gru_mean_m=gru_err,
                  improvement_pct=[round(100 * (1 - g / d), 1) if d else 0.0 for g, d in zip(gru_err, dr_err)], seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), k=K, horizons=list(HORIZONS), report=report), out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", default=[str(config.OPENSKY_CSV)], help="one or more state-vector CSVs (globs allowed)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default=str(config.MODELS_DIR / "traj_gru.pt"))
    a = ap.parse_args()
    files = sorted({f for pat in a.csv for f in (glob.glob(pat) or [pat])})
    print(json.dumps(train(files, a.epochs, Path(a.out)), indent=2))


if __name__ == "__main__":
    main()
