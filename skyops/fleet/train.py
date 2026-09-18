"""Train the C-MAPSS Remaining-Useful-Life models (LightGBM point + quantile models).

Not run automatically: launch it explicitly, e.g.
    python -m skyops.fleet.train --subsets FD001 FD002 FD003 FD004 --out models/rul_lgbm
CPU only, takes about a minute for all four subsets. Produces:
    models/rul_lgbm/{p50,p10,p90}.txt   LightGBM boosters
    models/rul_lgbm/manifest.json       feature list, normalisation stats, validation metrics
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from skyops import config
from skyops.fleet.features import SUBSETS, build_test_table, build_training_table, feature_columns


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric PHM08 score: late predictions are penalised more than early ones."""
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def train(subsets: tuple[str, ...], out_dir: Path, seed: int = 42, rounds: int = 2000) -> dict:
    t0 = time.time()
    F, stats = build_training_table(subsets)
    cols = feature_columns(F)
    X, y, groups = F[cols].values, F["rul"].values, F["engine_id"].values
    tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))
    dtr, dva = lgb.Dataset(X[tr_idx], y[tr_idx]), lgb.Dataset(X[va_idx], y[va_idx])
    base = dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    boosters, metrics = {}, {}
    for name, params in (("p50", dict(objective="regression")),
                         ("p10", dict(objective="quantile", alpha=0.10)),
                         ("p90", dict(objective="quantile", alpha=0.90))):
        b = lgb.train({**base, **params}, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        b.save_model(str(out_dir / f"{name}.txt"))
        boosters[name] = b
        pv = b.predict(X[va_idx], num_iteration=b.best_iteration)
        metrics[name] = dict(best_iteration=int(b.best_iteration), val_rmse=float(np.sqrt(np.mean((pv - y[va_idx]) ** 2))))

    # held-out test sets (last cycle of every test engine, true RUL from the RUL files)
    test_metrics = {}
    for sub in subsets:
        _, last = build_test_table(sub, stats)
        p = boosters["p50"].predict(last[cols].values, num_iteration=boosters["p50"].best_iteration)
        yt = np.minimum(last["rul_true"].values, 125.0)  # same clipping as the training target
        test_metrics[sub] = dict(n=int(len(last)), rmse=float(np.sqrt(np.mean((p - yt) ** 2))),
                                 mae=float(np.mean(np.abs(p - yt))), nasa_score=nasa_score(yt, p))
    imp = sorted(zip(cols, boosters["p50"].feature_importance("gain")), key=lambda t: -t[1])[:15]
    manifest = dict(kind="lightgbm-rul", subsets=list(subsets), features=cols, stats=stats, val=metrics, test=test_metrics,
                    top_features=[dict(feature=f, gain=float(g)) for f, g in imp], trained_seconds=round(time.time() - t0, 1))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    ap.add_argument("--out", default=str(config.MODELS_DIR / "rul_lgbm"))
    ap.add_argument("--rounds", type=int, default=2000)
    args = ap.parse_args()
    m = train(tuple(args.subsets), Path(args.out), rounds=args.rounds)
    print(json.dumps({k: m[k] for k in ("val", "test", "trained_seconds")}, indent=2))
    print("top features:", [t["feature"] for t in m["top_features"][:8]])


if __name__ == "__main__":
    main()
