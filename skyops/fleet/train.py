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

    # 1) point model (L2)
    b50 = lgb.train({**base, "objective": "regression"}, dtr, num_boost_round=rounds, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    b50.save_model(str(out_dir / "p50.txt"))
    boosters["p50"] = b50
    p50_tr, p50_va = b50.predict(X[tr_idx], num_iteration=b50.best_iteration), b50.predict(X[va_idx], num_iteration=b50.best_iteration)
    metrics["p50"] = dict(best_iteration=int(b50.best_iteration), val_rmse=float(np.sqrt(np.mean((p50_va - y[va_idx]) ** 2))))

    # 2) quantile models learn an *offset* around the point prediction (init_score = p50). Training them from
    #    the label quantile stalls for p90: the 125-cycle cap makes every gradient identical, so no split has gain.
    for name, alpha in (("p10", 0.10), ("p90", 0.90)):
        dq_tr = lgb.Dataset(X[tr_idx], y[tr_idx], init_score=p50_tr)
        dq_va = lgb.Dataset(X[va_idx], y[va_idx], init_score=p50_va)
        b = lgb.train({**base, "objective": "quantile", "alpha": alpha, "boost_from_average": False}, dq_tr, num_boost_round=rounds,
                      valid_sets=[dq_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        b.save_model(str(out_dir / f"{name}.txt"))
        boosters[name] = b
        off = b.predict(X[va_idx], num_iteration=b.best_iteration)
        metrics[name] = dict(best_iteration=int(b.best_iteration), mean_offset=float(off.mean()),
                             val_fraction_below=float(np.mean(y[va_idx] <= p50_va + off)))

    # 3) conformalised quantile regression: the offsets were fit on training residuals, which are optimistic,
    #    so widen the band by the validation conformity score that restores the nominal 80 % coverage.
    lo_va = p50_va + boosters["p10"].predict(X[va_idx], num_iteration=boosters["p10"].best_iteration)
    hi_va = p50_va + boosters["p90"].predict(X[va_idx], num_iteration=boosters["p90"].best_iteration)
    conformity = np.maximum(lo_va - y[va_idx], y[va_idx] - hi_va)
    conformal_q = float(max(0.0, np.quantile(conformity, 0.80)))
    metrics["conformal"] = dict(q=conformal_q, val_coverage_before=float(np.mean((y[va_idx] >= lo_va) & (y[va_idx] <= hi_va))),
                                val_coverage_after=float(np.mean((y[va_idx] >= lo_va - conformal_q) & (y[va_idx] <= hi_va + conformal_q))))

    # held-out test sets (last cycle of every test engine, true RUL from the RUL files)
    test_metrics = {}
    for sub in subsets:
        _, last = build_test_table(sub, stats)
        Xt = last[cols].values
        p = b50.predict(Xt, num_iteration=b50.best_iteration)
        lo = np.minimum(p + boosters["p10"].predict(Xt, num_iteration=boosters["p10"].best_iteration) - conformal_q, p)
        hi = np.maximum(p + boosters["p90"].predict(Xt, num_iteration=boosters["p90"].best_iteration) + conformal_q, p)
        yt = np.minimum(last["rul_true"].values, 125.0)  # same clipping as the training target
        test_metrics[sub] = dict(n=int(len(last)), rmse=float(np.sqrt(np.mean((p - yt) ** 2))), mae=float(np.mean(np.abs(p - yt))),
                                 nasa_score=nasa_score(yt, p), interval_coverage=float(np.mean((yt >= lo) & (yt <= hi))),
                                 interval_width=float(np.mean(hi - lo)))
    imp = sorted(zip(cols, b50.feature_importance("gain")), key=lambda t: -t[1])[:15]
    manifest = dict(kind="lightgbm-rul", quantile_mode="offset", conformal_q=conformal_q, subsets=list(subsets), features=cols, stats=stats, val=metrics,
                    test=test_metrics, top_features=[dict(feature=f, gain=float(g)) for f, g in imp],
                    trained_seconds=round(time.time() - t0, 1))
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
