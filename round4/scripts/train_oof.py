#!/usr/bin/env python3
"""
Round 4 — OOF train, modelo SOLO-TEXTO con LLM.

Features finales:
  - TF-IDF + SVD sobre description
  - embeddings OpenAI/Ollama + SVD
  - flags estructurados extraidos por LLM

Antes de correr por primera vez:
    /home/matias/miniconda3/envs/labo2/bin/python scripts/build_llm_cache.py --splits train test

Run from round4/:
    /home/matias/miniconda3/envs/labo2/bin/python scripts/train_oof.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

import features_llm as FL
import features_text as FT
import features_text_llm as FTL

TRAIN_TAB = Path("../participant/data/tabular/train_processed.csv")
OUT_DIR = Path("submissions_train")

SEEDS = [42, 99, 123]
N_FOLDS = 5
LGBM_PARAMS = dict(
    n_estimators=900,
    learning_rate=0.025,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.70,
    reg_alpha=1.8,
    reg_lambda=3.5,
    objective="huber",
    alpha=0.9,
)


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(pred - actual).sum() / np.abs(actual).sum() * 100)


def report_deciles(actual: np.ndarray, pred: np.ndarray) -> None:
    d = pd.DataFrame({"real": actual, "pred": pred})
    d["dec"] = pd.qcut(d["real"], 10, labels=False)
    log("Sesgo% por decil de precio real:")
    for dec, sub in d.groupby("dec"):
        real = sub["real"].values
        bias = (sub["pred"].values - real).mean() / real.mean() * 100
        print(
            f"    D{dec} ${real.min():>9,.0f}-${real.max():>9,.0f} {bias:+6.1f}%",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["openai", "ollama"], default=FL.DEFAULT_PROVIDER)
    p.add_argument("--embedding-model", default=FL.DEFAULT_EMBED_MODEL)
    p.add_argument("--llm-model", default=FL.DEFAULT_LLM_MODEL)
    p.add_argument("--cache-tag", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(TRAIN_TAB, usecols=["zpid", FT.TEXT_COL, FT.TARGET, FT.PRICE])
    df = df.dropna(subset=[FT.TARGET]).reset_index(drop=True)
    y = df[FT.TARGET].values
    actual = df[FT.PRICE].values

    log(
        f"Round4 texto+LLM | {len(SEEDS)} seeds x {N_FOLDS} folds | "
        f"{len(df):,} propiedades | provider={args.provider} | "
        f"embed={args.embedding_model} | llm={args.llm_model}"
    )

    oof_sum = np.zeros(len(df))
    tfidf_ev, emb_ev, n_features = [], [], []
    for seed in SEEDS:
        t0 = time.time()
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(df))
        for fold, (tr_idx, va_idx) in enumerate(kf.split(df), start=1):
            tr = df.iloc[tr_idx]
            va = df.iloc[va_idx]
            x_tr, store = FTL.fit_features(
                "train",
                tr,
                seed=seed + fold,
                provider=args.provider,
                embedding_model=args.embedding_model,
                llm_model=args.llm_model,
                cache_tag=args.cache_tag,
            )
            x_va = FTL.transform_features("train", va, store)
            model = lgb.LGBMRegressor(
                **LGBM_PARAMS,
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(x_tr, y[tr_idx])
            oof[va_idx] = model.predict(x_va)
            tfidf_ev.append(store.tfidf_store.explained_variance)
            emb_ev.append(store.emb_explained_variance)
            n_features.append(x_tr.shape[1])
        oof_sum += oof
        log(
            f"  seed {seed}: MAE ${mean_absolute_error(actual, np.expm1(oof)):,.0f} "
            f"({time.time() - t0:.0f}s)"
        )

    oof_log = oof_sum / len(SEEDS)
    oof_price = np.expm1(oof_log)
    metrics = {
        "mae": float(mean_absolute_error(actual, oof_price)),
        "wmape": wmape(actual, oof_price),
        "medape": float(np.median(np.abs(oof_price - actual) / actual * 100)),
        "r2_log": float(r2_score(y, oof_log)),
        "tfidf_svd_explained_variance_mean": float(np.mean(tfidf_ev)),
        "llm_emb_svd_explained_variance_mean": float(np.mean(emb_ev)),
        "n_features_mean": float(np.mean(n_features)),
    }
    log(
        f"OOF texto+LLM -> MAE ${metrics['mae']:,.0f} | "
        f"wMAPE {metrics['wmape']:.2f}% | medAPE {metrics['medape']:.2f}% | "
        f"R2(log) {metrics['r2_log']:.4f}"
    )
    report_deciles(actual, oof_price)

    flags = FL.load_flag_features(
        "train", df, model=args.llm_model, cache_tag=args.cache_tag, provider=args.provider
    ).reset_index(drop=True)
    comparison = pd.concat(
        [
            pd.DataFrame(
                {
                    "zpid": df["zpid"].values,
                    "valor_real": actual,
                    "predicted_price": oof_price,
                    "error": oof_price - actual,
                    "error_pct": np.abs(oof_price - actual) / actual * 100,
                    "description_length": FT.clean_text(df).str.len().values,
                }
            ),
            flags,
        ],
        axis=1,
    )
    comparison.to_csv(OUT_DIR / "comparison_oof.csv", index=False)

    practice = pd.read_csv(TRAIN_TAB, usecols=["zpid"]).merge(
        comparison[["zpid", "predicted_price"]],
        on="zpid",
        how="left",
    )
    practice["predicted_price"] = practice["predicted_price"].fillna(float(np.median(oof_price)))
    practice.to_csv(OUT_DIR / "practice_submission.csv", index=False)

    with open(OUT_DIR / "text_model_config.json", "w") as f:
        json.dump(
            {
                "seeds": SEEDS,
                "n_folds": N_FOLDS,
                "lgbm_params": LGBM_PARAMS,
                "provider": args.provider,
                "embedding_model": args.embedding_model,
                "llm_model": args.llm_model,
                "cache_tag": args.cache_tag,
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    log("Guardado practice_submission.csv + comparison_oof.csv + text_model_config.json")


if __name__ == "__main__":
    main()
