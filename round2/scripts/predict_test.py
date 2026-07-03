"""Round 2 — Prediccion test, mismo modelo que train_oof."""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import features_img as FI
from train_oof import LGBM_PARAMS, SEEDS, TARGET, PRICE

TRAIN_TAB = Path("../participant/data/tabular/train_processed.csv")
TEMPLATE = Path("../participant/submissions/template.csv")
OUT_PATH = Path("submissions/submission.csv")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    OUT_PATH.parent.mkdir(exist_ok=True)
    train_pooled = FI.pooled_matrices("train")
    train_feats, store = FI.fit_features(train_pooled)
    test_feats = FI.transform_features(FI.pooled_matrices("test"), store)

    tab = pd.read_csv(TRAIN_TAB, usecols=["zpid", TARGET, PRICE])
    df = train_feats.join(tab.set_index("zpid"), how="inner").dropna(subset=[TARGET])
    cols = train_feats.columns
    Xtr, y = df[cols], df[TARGET].values
    Xte = test_feats[cols]

    test_log = np.zeros(len(test_feats))
    for seed in SEEDS:
        t0 = time.time()
        m = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=seed, n_jobs=-1, verbosity=-1)
        m.fit(Xtr, y)
        test_log += m.predict(Xte)
        log(f"  seed {seed} ({time.time() - t0:.0f}s)")
    test_log /= len(SEEDS)
    test_price = np.expm1(test_log)

    sub = pd.read_csv(TEMPLATE, usecols=["zpid"]).merge(
        pd.DataFrame({"zpid": test_feats.index, "predicted_price": test_price}),
        on="zpid", how="left")
    sub["predicted_price"] = sub["predicted_price"].fillna(float(np.median(test_price)))
    sub.to_csv(OUT_PATH, index=False)
    log(f"Guardado {OUT_PATH} ({len(sub):,} filas)")


if __name__ == "__main__":
    main()
