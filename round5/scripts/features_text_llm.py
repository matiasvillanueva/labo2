"""
Round 4 — Features combinadas de texto:
  - TF-IDF + SVD
  - embeddings semanticos OpenAI/Ollama + SVD
  - flags estructurados extraidos por LLM
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

import features_llm as FL
import features_text as FT

LLM_EMB_COMPONENTS = 256


@dataclass
class ComboFeatureStore:
    tfidf_store: FT.TextFeatureStore
    emb_svd: TruncatedSVD
    emb_columns: list[str]
    emb_explained_variance: float
    provider: str
    embedding_model: str
    llm_model: str
    cache_tag: str


def _fit_embedding_features(
    split: str,
    df: pd.DataFrame,
    seed: int,
    model: str,
    cache_tag: str,
    provider: str,
) -> tuple[pd.DataFrame, TruncatedSVD, float]:
    raw = FL.load_embedding_matrix(split, df, model=model, cache_tag=cache_tag, provider=provider)
    n_components = min(LLM_EMB_COMPONENTS, max(1, raw.shape[1] - 1), raw.shape[0])
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    values = svd.fit_transform(raw).astype(np.float32)
    columns = [f"llm_emb_svd_{i}" for i in range(n_components)]
    features = pd.DataFrame(values, index=df.index, columns=columns)
    return features, svd, float(svd.explained_variance_ratio_.sum())


def _transform_embedding_features(
    split: str,
    df: pd.DataFrame,
    svd: TruncatedSVD,
    columns: list[str],
    model: str,
    cache_tag: str,
    provider: str,
) -> pd.DataFrame:
    raw = FL.load_embedding_matrix(split, df, model=model, cache_tag=cache_tag, provider=provider)
    values = svd.transform(raw).astype(np.float32)
    return pd.DataFrame(values, index=df.index, columns=columns)


def fit_features(
    split: str,
    df: pd.DataFrame,
    seed: int = 42,
    provider: str = FL.DEFAULT_PROVIDER,
    embedding_model: str = FL.DEFAULT_EMBED_MODEL,
    llm_model: str = FL.DEFAULT_LLM_MODEL,
    cache_tag: str = "",
) -> tuple[pd.DataFrame, ComboFeatureStore]:
    tfidf_features, tfidf_store = FT.fit_features(df, seed=seed)
    emb_features, emb_svd, emb_ev = _fit_embedding_features(
        split, df, seed=seed, model=embedding_model, cache_tag=cache_tag, provider=provider
    )
    flags = FL.load_flag_features(split, df, model=llm_model, cache_tag=cache_tag, provider=provider)
    features = pd.concat([tfidf_features, emb_features, flags], axis=1)
    store = ComboFeatureStore(
        tfidf_store=tfidf_store,
        emb_svd=emb_svd,
        emb_columns=list(emb_features.columns),
        emb_explained_variance=emb_ev,
        provider=provider,
        embedding_model=embedding_model,
        llm_model=llm_model,
        cache_tag=cache_tag,
    )
    return features, store


def transform_features(split: str, df: pd.DataFrame, store: ComboFeatureStore) -> pd.DataFrame:
    tfidf_features = FT.transform_features(df, store.tfidf_store)
    emb_features = _transform_embedding_features(
        split,
        df,
        store.emb_svd,
        store.emb_columns,
        model=store.embedding_model,
        cache_tag=store.cache_tag,
        provider=store.provider,
    )
    flags = FL.load_flag_features(
        split, df, model=store.llm_model, cache_tag=store.cache_tag, provider=store.provider
    )
    return pd.concat([tfidf_features, emb_features, flags], axis=1)
