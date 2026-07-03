#!/usr/bin/env python3
"""Round 4 — Prediccion test para el modelo final texto+LLM."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import features_llm as FL
import features_text as FT
import features_text_llm as FTL
from train_oof import LGBM_PARAMS, SEEDS

TRAIN_TAB = Path("../participant/data/tabular/train_processed.csv")
TEST_TAB = Path("../participant/data/tabular/test_processed.csv")
TEMPLATE = Path("../participant/submissions/template.csv")
OUT_PATH = Path("submissions/submission.csv")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["openai", "ollama"], default=FL.DEFAULT_PROVIDER)
    p.add_argument("--embedding-model", default=FL.DEFAULT_EMBED_MODEL)
    p.add_argument("--llm-model", default=FL.DEFAULT_LLM_MODEL)
    p.add_argument("--cache-tag", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_PATH.parent.mkdir(exist_ok=True)

    train = pd.read_csv(TRAIN_TAB, usecols=["zpid", FT.TEXT_COL, FT.TARGET, FT.PRICE])
    test = pd.read_csv(TEST_TAB, usecols=["zpid", FT.TEXT_COL])
    template = pd.read_csv(TEMPLATE, usecols=["zpid"])
    train = train.dropna(subset=[FT.TARGET]).reset_index(drop=True)

    y = train[FT.TARGET].values
    log(
        f"Round4 texto+LLM test | Train {len(train):,} | Test {len(test):,} | "
        f"Template {len(template):,} | provider={args.provider} | ensemble {len(SEEDS)} seeds"
    )

    test_log = np.zeros(len(test))
    train_log = np.zeros(len(train))
    n_features = []
    for seed in SEEDS:
        x_train, store = FTL.fit_features(
            "train",
            train,
            seed=seed,
            provider=args.provider,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            cache_tag=args.cache_tag,
        )
        x_test = FTL.transform_features("test", test, store)
        model = lgb.LGBMRegressor(
            **LGBM_PARAMS,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_train, y)
        test_log += model.predict(x_test)
        train_log += model.predict(x_train)
        n_features.append(x_train.shape[1])
        log(
            f"  seed {seed}: {x_train.shape[1]} features | "
            f"tfidf var {store.tfidf_store.explained_variance:.1%} | "
            f"llm emb var {store.emb_explained_variance:.1%}"
        )

    test_log /= len(SEEDS)
    train_log /= len(SEEDS)
    test_price = np.expm1(test_log)

    train_real = train[FT.PRICE].values
    train_pred = np.expm1(train_log)
    log(
        f"Diagnostico train in-sample: R2(log) {r2_score(y, train_log):.4f} | "
        f"MAE ${mean_absolute_error(train_real, train_pred):,.0f} | "
        f"features media {np.mean(n_features):.0f}"
    )

    pred = pd.DataFrame({"zpid": test["zpid"], "predicted_price": test_price})
    sub = template.merge(pred, on="zpid", how="left")
    missing = sub["predicted_price"].isna().sum()
    if missing:
        raise ValueError(f"Faltan predicciones para {missing} zpids del template")

    sub.to_csv(OUT_PATH, index=False)
    log(f"Guardado {OUT_PATH} ({len(sub):,} filas)")
    print(sub["predicted_price"].describe())


if __name__ == "__main__":
    main()
