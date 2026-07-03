"""
Round 2 — OOF train, modelo SOLO-IMAGEN.

Features: pools ie/sat, flags listado escaso (patron error 1-foto/satelite).
LGBM Huber + multi-seed KFold OOF.
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

import features_img as FI

TRAIN_TAB = Path("../participant/data/tabular/train_processed.csv")
OUT_DIR = Path("submissions_train")
TARGET = "log_price"
PRICE = "lastSoldPrice_hpi_adjusted"

SEEDS = [42, 99, 123]
N_FOLDS = 5
LGBM_PARAMS = dict(
    n_estimators=800, learning_rate=0.025, num_leaves=63,
    min_child_samples=25, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.55, reg_alpha=1.5, reg_lambda=2.5,
    objective="huber", alpha=0.9,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wmape(actual, pred):
    return float(np.abs(pred - actual).sum() / np.abs(actual).sum() * 100)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    log("Agregando features de imagen (train)...")
    pooled = FI.pooled_matrices("train")
    feats, _ = FI.fit_features(pooled)
    flags = pooled[-1]

    tab = pd.read_csv(TRAIN_TAB, usecols=["zpid", TARGET, PRICE])
    df = feats.join(tab.set_index("zpid"), how="inner").dropna(subset=[TARGET])
    flags = flags.reindex(df.index).fillna(0)
    risk = flags["single_photo"].values.astype(bool)
    log(f"Propiedades: {len(df):,} | 1-foto: {risk.sum():,}")

    X = df[feats.columns]
    y = df[TARGET].values
    actual = df[PRICE].values
    n = len(df)

    oof_sum = np.zeros(n)
    for seed in SEEDS:
        t0 = time.time()
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(n)
        for tr, va in kf.split(X):
            model = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=seed,
                                      n_jobs=-1, verbosity=-1)
            model.fit(X.iloc[tr], y[tr])
            oof[va] = model.predict(X.iloc[va])
        oof_sum += oof
        log(f"  seed {seed}: MAE ${mean_absolute_error(actual, np.expm1(oof)):,.0f} "
            f"({time.time() - t0:.0f}s)")

    oof_log = oof_sum / len(SEEDS)
    oof_price = np.expm1(oof_log)
    log(f"OOF -> R²(log) {r2_score(y, oof_log):.4f} | "
        f"MAE ${mean_absolute_error(actual, oof_price):,.0f} | "
        f"wMAPE {wmape(actual, oof_price):.2f}% | "
        f"medAPE {np.median(np.abs(oof_price - actual) / actual * 100):.2f}%")
    if risk.any():
        a, p = actual[risk], oof_price[risk]
        log(f"  [1-foto n={risk.sum():,}] medAPE {np.median(np.abs(p-a)/a*100):.1f}% | "
            f"cheap<150k medAPE {np.median(np.abs(p[a<150000]-a[a<150000])/a[a<150000]*100):.1f}%")

    comparison = pd.DataFrame({
        "zpid": df.index.values,
        "valor_real": actual,
        "predicted_price": oof_price,
        "error_pct": np.abs(oof_price - actual) / actual * 100,
        "single_photo": flags["single_photo"].values,
        "satellite_only": flags["satellite_only"].values,
    })
    comparison.to_csv(OUT_DIR / "comparison_oof.csv", index=False)

    all_zpids = pd.read_csv(TRAIN_TAB, usecols=["zpid"])
    preds = pd.DataFrame({"zpid": df.index.values, "predicted_price": oof_price})
    sub = all_zpids.merge(preds, on="zpid", how="left")
    sub["predicted_price"] = sub["predicted_price"].fillna(float(np.median(oof_price)))
    sub.to_csv(OUT_DIR / "practice_submission.csv", index=False)
    log(f"Guardado practice_submission.csv + comparison_oof.csv")


if __name__ == "__main__":
    main()
