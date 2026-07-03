"""
Predice sobre test_processed.csv usando el mejor modelo guardado y genera la
submission lista para subir.

Ensemble multi-seed (como OOF) + predicción raw sin calibración post-modelo
(calibración empeoraba wMAPE ~2.5pp en practice).

Run from round1/:
    python scripts/predict_lgbm.py
"""

import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import features as F


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

TRAIN_PATH = "data/tabular/train_processed.csv"
TEST_PATH = "data/tabular/test_processed.csv"
TEMPLATE_PATH = "../participant/submissions/template.csv"
BEST_MODEL_PATH = Path("submissions_train/best_model.json")
DEBIAS_PATH = Path("submissions_train/debias.json")
OUTPUT_DIR = Path("submissions")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


def main():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe {BEST_MODEL_PATH}. Corré primero: python scripts/train_lgbm.py"
        )

    with open(BEST_MODEL_PATH) as f:
        best = json.load(f)

    debias_params = None
    if DEBIAS_PATH.exists():
        with open(DEBIAS_PATH) as f:
            debias_params = json.load(f)
        log(f"De-sesgo cargado ({debias_params.get('method')}) desde {DEBIAS_PATH}")
    else:
        log(f"AVISO: no existe {DEBIAS_PATH}; submission SIN de-sesgo. "
            f"Corré primero: python scripts/predict_train_oof.py")

    feature_cols = F.feature_columns()
    seeds = best.get("seeds", [42])
    params = best["params"]
    cal = "calibrado" if F.USE_CALIBRATION else "raw"
    log(f"Modelo ganador '{best['config_name']}' | Val MAE ${best['val_mae']:,.0f}")
    obj = f"quantile τ={F.PREDICTION_QUANTILE}" if F.PREDICTION_QUANTILE is not None else "media"
    log(f"Features: {len(feature_cols)} | objetivo {obj} | salida {cal} | k={F.CONSERVATIVE_SCALE} | "
        f"ensemble {len(seeds)} seeds")

    train = F.build_features(pd.read_csv(TRAIN_PATH))
    test = F.build_features(pd.read_csv(TEST_PATH))
    template = pd.read_csv(TEMPLATE_PATH)
    log(f"Train: {len(train):,} filas | Test: {len(test):,} filas | Template: {len(template):,} zpids")

    train, test = F.apply_full_encodings(train, test)
    train = F.cast_categoricals(train)
    test = F.cast_categoricals(test)

    test_log_sum = np.zeros(len(test))
    train_log_sum = np.zeros(len(train))
    for seed in seeds:
        model = lgb.LGBMRegressor(
            **params, **F.model_objective_params(),
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
        model.fit(train[feature_cols], train[F.TARGET], categorical_feature=F.CATEGORICAL_FEATURES)
        test_log_sum += model.predict(test[feature_cols])
        train_log_sum += model.predict(train[feature_cols])

    test_log = test_log_sum / len(seeds)
    train_log = train_log_sum / len(seeds)

    test_pred_log = F.finalize_predictions(test_log, test, debias=debias_params)
    test_pred_price = np.expm1(test_pred_log)

    train_pred = np.expm1(F.finalize_predictions(train_log, train, debias=debias_params))
    train_real = np.expm1(train[F.TARGET].values)
    log(f"Diagnóstico train: R² {r2_score(train_real, train_pred):.4f} | "
        f"MAE ${mean_absolute_error(train_real, train_pred):,.0f}")

    predictions = pd.DataFrame({"zpid": test["zpid"], "predicted_price": test_pred_price})
    submission = template[["zpid"]].merge(predictions, on="zpid", how="left")
    missing = submission["predicted_price"].isna().sum()
    if missing:
        raise ValueError(f"Faltan predicciones para {missing} zpids del template")

    OUTPUT_DIR.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    log(f"Guardado {OUTPUT_PATH} ({len(submission)} filas)")
    print(submission["predicted_price"].describe())


if __name__ == "__main__":
    main()
