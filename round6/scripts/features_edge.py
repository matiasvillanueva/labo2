"""
Round 6 — Edge selectivo contra sobrevaluaciones peligrosas.

El CSV de una ronda real mostro que otro competidor podia empatar nuestra oferta.
En ese caso se termina pagando aproximadamente:

    cost ~= predicted_price * BID_FRACTION * (1 + TRANSACTION_COST)

Con los defaults (0.85 y 2%), una compra deja de ser rentable cuando
predicted_price / true_value > 1 / (0.85 * 1.02) ~= 1.153.

Este modulo aprende OOF:
  1. probabilidad de cruzar ese limite ("trap risk");
  2. severidad esperada de la sobrevaluacion en log-precio.

El edge solo recorta propiedades de riesgo alto. No baja globalmente todas las
valuaciones, para preservar el wMAPE de la base precisa.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import features as F

BID_FRACTION = 0.85
TRANSACTION_COST = 0.02
SAFE_PRED_TO_TRUE = 1.0 / (BID_FRACTION * (1.0 + TRANSACTION_COST))
SAFE_OVER_LOG = float(np.log(SAFE_PRED_TO_TRUE))
MAX_EDGE_LOG_CUT = 0.35

CLASSIFIER_PARAMS = dict(
    n_estimators=500, learning_rate=0.025, num_leaves=31,
    min_child_samples=40, colsample_bytree=0.65,
    reg_alpha=2.0, reg_lambda=5.0,
)
SEVERITY_PARAMS = dict(
    n_estimators=500, learning_rate=0.025, num_leaves=31,
    min_child_samples=40, colsample_bytree=0.65,
    reg_alpha=2.0, reg_lambda=5.0,
    objective="huber", alpha=0.8,
)

BASE_FEATURES = (
    F.RAW_FEATURES
    + F.ENGINEERED_FEATURES
    + F.PHOTO_FEATURES
    + F.DESC_PRICE_FEATURES
)


def build_features(df: pd.DataFrame, base_price: np.ndarray) -> pd.DataFrame:
    """Features independientes del target + diagnosticos de edge de la prediccion."""
    price = np.asarray(base_price, dtype=np.float64)
    out = df[BASE_FEATURES].copy()
    out["edge_base_log"] = np.log1p(np.maximum(price, 0.0))

    tax = df["taxAssessedValue"].replace(0, np.nan)
    latest_tax = df["latest_tax_value"].replace(0, np.nan)
    listing = df["last_listing_price"].where(df["last_listing_price"] >= 50_000)
    area = df["livingArea"].replace(0, np.nan)

    # Log-ratios: robustos ante anchors chicos y faciles de modelar.
    out["edge_log_pred_to_tax"] = np.log(np.maximum(price, 1.0) / tax)
    out["edge_log_pred_to_latest_tax"] = np.log(np.maximum(price, 1.0) / latest_tax)
    out["edge_log_pred_to_listing"] = np.log(np.maximum(price, 1.0) / listing)
    out["edge_log_pred_per_sqft"] = np.log(np.maximum(price, 1.0) / area)
    out = out.replace([np.inf, -np.inf], np.nan)
    return F.cast_categoricals(out)


def targets(base_price: np.ndarray, real_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_log = np.log1p(np.maximum(np.asarray(base_price, dtype=np.float64), 0.0))
    over_log = pred_log - np.asarray(real_log, dtype=np.float64)
    trap = (over_log > SAFE_OVER_LOG).astype(np.int8)
    severity = np.maximum(over_log, 0.0)
    return trap, severity


def fit_models(
    x: pd.DataFrame,
    trap: np.ndarray,
    severity: np.ndarray,
    seed: int = 42,
) -> tuple[lgb.LGBMClassifier, lgb.LGBMRegressor]:
    classifier = lgb.LGBMClassifier(
        **CLASSIFIER_PARAMS, random_state=seed, n_jobs=-1, verbosity=-1
    )
    classifier.fit(x, trap, categorical_feature=F.CATEGORICAL_FEATURES)

    regressor = lgb.LGBMRegressor(
        **SEVERITY_PARAMS, random_state=seed, n_jobs=-1, verbosity=-1
    )
    regressor.fit(x, severity, categorical_feature=F.CATEGORICAL_FEATURES)
    return classifier, regressor


def predict_models(
    classifier: lgb.LGBMClassifier,
    regressor: lgb.LGBMRegressor,
    x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    risk = classifier.predict_proba(x)[:, 1]
    severity = np.maximum(regressor.predict(x), 0.0)
    return risk, severity


def crossfit(
    df: pd.DataFrame,
    base_price: np.ndarray,
    real_log: np.ndarray,
    n_folds: int = 5,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicciones OOF del edge; cada fila se estima sin ver su error real."""
    x = build_features(df, base_price)
    trap, severity_target = targets(base_price, real_log)
    risk_oof = np.zeros(len(df))
    severity_oof = np.zeros(len(df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x)):
        classifier, regressor = fit_models(
            x.iloc[tr_idx], trap[tr_idx], severity_target[tr_idx], seed + fold
        )
        risk_oof[va_idx], severity_oof[va_idx] = predict_models(
            classifier, regressor, x.iloc[va_idx]
        )
    return risk_oof, severity_oof, trap


def apply_edge(
    base_price: np.ndarray,
    risk: np.ndarray,
    severity: np.ndarray,
    threshold: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica recorte solo si risk >= threshold; devuelve precio y cut log."""
    cut_log = np.where(
        np.asarray(risk) >= threshold,
        alpha * np.asarray(severity),
        0.0,
    )
    cut_log = np.clip(cut_log, 0.0, MAX_EDGE_LOG_CUT)
    pred_log = np.log1p(np.maximum(np.asarray(base_price), 0.0)) - cut_log
    return np.expm1(pred_log), cut_log
