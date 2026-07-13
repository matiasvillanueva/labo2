"""
Round 5 — Submission de test. Usa la estrategia elegida por train_oof.py
(round5_config.json), combinando tabular+imagen (round3) con texto/LLM (round4):

  - feature_fusion : fit full de un unico LightGBM con TODAS las columnas
    (tabular + imagen + TF-IDF/SVD + embeddings LLM/SVD + flags) y predice test.
  - text_residual  : toma la submission de round3 como base tabular+imagen y le suma
    el residual de texto/LLM acotado (alpha, cap del OOF), sin recomputar round3.

Corre train_oof.py primero (genera round5_config.json y, para el residual, usa el OOF
de round3). Salida: submissions/submission.csv (+ diff_vs_round3_test.csv).

Run from round5/:
    /home/matias/miniconda3/envs/labo2/bin/python scripts/predict_test.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import features as F
import features_fusion as FF
import features_text_llm as FTL
from features_fusion import LLM_KWARGS
from train_oof import SEEDS, TABULAR_PARAMS, TAU, TEXT_PARAMS

TRAIN_PATH = "data/tabular/train_processed.csv"
TEST_PATH = "data/tabular/test_processed.csv"
TEMPLATE_PATH = "../participant/submissions/template.csv"
ROUND3_COMPARISON = "../round3/submissions_train/comparison_oof.csv"
ROUND3_SUBMISSION = "../round3/submissions/submission.csv"
OOF_DIR = Path("submissions_train")
OUT_DIR = Path("submissions")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def predict_fusion(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """feature_fusion: fit full de LightGBM cuantil con todas las columnas."""
    tab_q = dict(TABULAR_PARAMS, objective="quantile", alpha=TAU)
    img_train = FF.image_frame("train")
    img_test = FF.image_frame("test")
    test_log = np.zeros(len(test))
    for seed in SEEDS:
        x_train, x_test = FF.assemble_full(train, test, img_train, img_test, seed=seed)
        m = lgb.LGBMRegressor(**tab_q, random_state=seed, n_jobs=-1, verbosity=-1)
        m.fit(x_train, train[F.TARGET].values, categorical_feature=F.CATEGORICAL_FEATURES)
        test_log += m.predict(x_test)
        log(f"  [fusion] seed {seed}: {x_train.shape[1]} cols")
    return test_log / len(SEEDS)


def predict_text_residual(train: pd.DataFrame, test: pd.DataFrame,
                          alpha: float, cap: float) -> tuple[np.ndarray, np.ndarray]:
    """text_residual: base round3 (test) + residual texto/LLM acotado.

    El residual full se entrena sobre real - round3_oof (sin leakage) y se predice en
    test; la base de test es la submission de round3 (no se recomputa round3)."""
    r3_oof = pd.read_csv(ROUND3_COMPARISON).set_index("zpid")["predicted_price"]
    base_train_log = np.log1p(r3_oof.reindex(train["zpid"].values).values)
    res_target = train[F.TARGET].values - base_train_log

    r3_sub = pd.read_csv(ROUND3_SUBMISSION).set_index("zpid")["predicted_price"]
    base_test_price = r3_sub.reindex(test["zpid"].values).values
    base_test_log = np.log1p(base_test_price)

    # alpha=0 => el residual no aporta (OOF eligio el piso de seguridad = round3).
    # Se evita entrenar el modelo de texto porque el delta seria cero de todos modos.
    if alpha == 0.0:
        log("  [residual] alpha=0: round5 == round3 (texto/LLM no mejora en OOF)")
        return base_test_price.copy(), base_test_price

    res_test_sum = np.zeros(len(test))
    for seed in SEEDS:
        x_train, store = FTL.fit_features("train", train, seed=seed, **LLM_KWARGS)
        x_test = FTL.transform_features("test", test, store)
        m = lgb.LGBMRegressor(**TEXT_PARAMS, random_state=seed, n_jobs=-1, verbosity=-1)
        m.fit(x_train, res_target)
        res_test_sum += m.predict(x_test)
        log(f"  [residual] seed {seed}: {x_train.shape[1]} cols")
    res_test = res_test_sum / len(SEEDS)

    delta = np.clip(alpha * res_test, -cap, cap)
    return np.expm1(base_test_log + delta), base_test_price


def main() -> None:
    cfg_path = OOF_DIR / "round5_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Falta {cfg_path}. Corre primero scripts/train_oof.py")
    with open(cfg_path) as f:
        cfg = json.load(f)
    strategy = cfg["strategy"]

    train = F.build_features(pd.read_csv(TRAIN_PATH))
    test = F.build_features(pd.read_csv(TEST_PATH))
    template = pd.read_csv(TEMPLATE_PATH)
    log(f"Round5 test | estrategia={strategy} | Train {len(train):,} | "
        f"Test {len(test):,} | Template {len(template):,}")

    r3_base_test = None
    if strategy == "feature_fusion":
        test_log = predict_fusion(train, test)
        test_price = np.expm1(test_log)
    elif strategy == "text_residual":
        alpha = float(cfg["residual_alpha"])
        cap = float(cfg["residual_cap"])
        log(f"Residual texto/LLM sobre round3 | alpha={alpha} cap={cap}")
        test_price, r3_base_test = predict_text_residual(train, test, alpha, cap)
    else:
        raise ValueError(f"Estrategia desconocida en config: {strategy}")

    pred = pd.DataFrame({"zpid": test["zpid"].values, "predicted_price": test_price})
    sub = template[["zpid"]].merge(pred, on="zpid", how="left")
    if sub["predicted_price"].isna().any():
        raise ValueError("Faltan predicciones para zpids del template")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "submission.csv"
    sub.to_csv(out, index=False)

    # ── Diff vs submission de round3 (mismos zpids del template) ──
    r3_sub = pd.read_csv(ROUND3_SUBMISSION).rename(
        columns={"predicted_price": "round3_price"})
    diff = sub.merge(r3_sub, on="zpid", how="left").rename(
        columns={"predicted_price": "round5_price"})
    diff["diff"] = diff["round5_price"] - diff["round3_price"]
    diff["diff_pct"] = diff["diff"] / diff["round3_price"] * 100
    diff.to_csv(OUT_DIR.parent / "submissions_train" / "diff_vs_round3_test.csv", index=False)

    log(f"Guardado {out} ({len(sub):,} filas)")
    log(f"vs round3 (test): diff medio ${diff['diff'].mean():,.0f} | "
        f"|diff| medio ${diff['diff'].abs().mean():,.0f} | "
        f"|diff%| medio {diff['diff_pct'].abs().mean():.2f}%")
    print(sub["predicted_price"].describe())


if __name__ == "__main__":
    main()
