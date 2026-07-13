"""
Round 3 — Features de imagen INTERPRETABLES (sin PCA) para el residual con cap.

A diferencia de features_img.py (351 cols con PCA de embeddings), aca solo exponemos
escalares con sentido fisico (materiales, composicion de tipos, flags de listado escaso,
confianza visual) para que un modelo chico corrija el residual del tabular sin diluir la
senal ni moverse demasiado.

Reusa pooled_matrices() de features_img.py (scal + flags) — no recalcula embeddings.

API:
  scalar_features(split) -> DataFrame index=zpid (~35 cols interpretables)
  gate_mask(feats)        -> Series 0/1 (1 = se permite ajuste de imagen)
  fit_impute(train_feats) -> medians (Series) + persiste embeddings/img_scalar_impute.json
  apply_impute(feats, medians) -> rellena props sin foto con medianas train
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import features_img as FI

IMPUTE_PATH = Path("embeddings/img_scalar_impute.json")

# Columnas interpretables que expone el residual (subconjunto de scal + flags).
MAT_COLS = [f"mat_{k}_{stat}" for k in FI.MAT_KEYS for stat in ("mean", "max")]
COMP_COLS = ["frac_interior", "frac_exterior", "frac_satellite", "frac_ie",
             "type_entropy", "n_photos_total", "has_satellite"]
FLAG_COLS = ["single_photo", "satellite_only", "sparse_listing", "log_n_images",
             "few_photos"]
CONF_COLS = ["emb_std"]
SCALAR_COLS = MAT_COLS + COMP_COLS + FLAG_COLS + CONF_COLS


def scalar_features(split: str) -> pd.DataFrame:
    """Escalares interpretables por zpid (index=zpid). Solo props con al menos 1 foto."""
    pooled = FI.pooled_matrices(split)
    scal, flags = pooled[6], pooled[7]
    feats = scal.join(flags, how="outer")
    cols = [c for c in SCALAR_COLS if c in feats.columns]
    return feats[cols]


def gate_mask(feats: pd.DataFrame) -> pd.Series:
    """1 = ajuste de imagen permitido. Se apaga donde la imagen empeoraba en round3:
    1 sola foto, solo satelite, o listados con menos de 3 fotos."""
    sparse = feats.get("sparse_listing", pd.Series(0, index=feats.index)).fillna(1)
    single = feats.get("single_photo", pd.Series(0, index=feats.index)).fillna(1)
    sat = feats.get("satellite_only", pd.Series(0, index=feats.index)).fillna(0)
    ok = (sparse == 0) & (single == 0) & (sat == 0)
    return ok.astype(int)


def fit_impute(train_feats: pd.DataFrame) -> pd.Series:
    medians = train_feats.median()
    IMPUTE_PATH.parent.mkdir(exist_ok=True)
    with open(IMPUTE_PATH, "w") as f:
        json.dump({c: float(medians[c]) for c in train_feats.columns}, f, indent=2)
    return medians


def apply_impute(feats: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return feats.reindex(columns=medians.index).fillna(medians)
