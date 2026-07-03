"""
Round 4 — Features solo-texto desde la columna description.

Usa TF-IDF sobre la descripcion cruda y reduce dimensionalidad con SVD. No usa
features tabulares, imagenes ni columnas derivadas existentes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_COL = "description"
TARGET = "log_price"
PRICE = "lastSoldPrice_hpi_adjusted"

MAX_FEATURES = 5000
N_COMPONENTS = 256
MIN_DF = 3
MAX_DF = 0.98


@dataclass
class TextFeatureStore:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    columns: list[str]
    explained_variance: float


def clean_text(df: pd.DataFrame) -> pd.Series:
    """Devuelve la descripcion como string, con NaN reemplazados por vacio."""
    if TEXT_COL not in df.columns:
        raise KeyError(f"Falta la columna requerida: {TEXT_COL}")
    return df[TEXT_COL].fillna("").astype(str)


def fit_features(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, TextFeatureStore]:
    texts = clean_text(df)
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    tfidf = vectorizer.fit_transform(texts)
    n_components = min(N_COMPONENTS, max(1, tfidf.shape[1] - 1), tfidf.shape[0])
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    values = svd.fit_transform(tfidf).astype(np.float32)
    columns = [f"text_svd_{i}" for i in range(n_components)]
    features = pd.DataFrame(values, index=df.index, columns=columns)
    store = TextFeatureStore(
        vectorizer=vectorizer,
        svd=svd,
        columns=columns,
        explained_variance=float(svd.explained_variance_ratio_.sum()),
    )
    return features, store


def transform_features(df: pd.DataFrame, store: TextFeatureStore) -> pd.DataFrame:
    tfidf = store.vectorizer.transform(clean_text(df))
    values = store.svd.transform(tfidf).astype(np.float32)
    return pd.DataFrame(values, index=df.index, columns=store.columns)
