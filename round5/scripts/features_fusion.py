"""
Round 5 — Ensamblado de features para la FUSION por columnas.

Combina, por zpid, las tres fuentes de senal de los rounds previos:
  - tabular  (round1/round3): F.feature_columns()  -> 52 cols
  - imagen   (round3):        IR.scalar_features()  -> ~33 escalares interpretables
  - texto/LLM(round4):        FTL.fit_features()    -> TF-IDF+SVD + emb LLM+SVD + flags

La parte tabular usa encodings OOF-safe por fold (cuantiles ZIP + KNN). La parte de
texto ajusta TF-IDF/SVD dentro de cada fold. La imagen se imputa con la mediana del
train-fold. Asi la fusion no filtra informacion del validation fold.

Reutilizado por train_oof.py (por fold) y predict_test.py (full train -> test).
"""

from __future__ import annotations

import pandas as pd

import features as F
import features_img_residual as IR
import features_llm as FL
import features_text_llm as FTL

TAB_COLS = F.feature_columns()

# kwargs por defecto del cache LLM ya materializado (OpenAI barato + flags cheap_v2)
LLM_KWARGS = dict(
    provider=FL.DEFAULT_PROVIDER,
    embedding_model=FL.DEFAULT_EMBED_MODEL,
    llm_model=FL.DEFAULT_LLM_MODEL,
    cache_tag="",
)


def image_frame(split: str) -> pd.DataFrame:
    """Escalares interpretables de imagen por zpid (index=zpid)."""
    return IR.scalar_features(split)


def _image_block(zpids, img_all: pd.DataFrame, medians: pd.Series, index) -> pd.DataFrame:
    block = img_all.reindex(zpids).fillna(medians)
    block.index = index
    return block


def assemble_fold(tr: pd.DataFrame, va: pd.DataFrame, img_all: pd.DataFrame, seed: int):
    """Devuelve (X_tr, X_va) fusionadas para un fold OOF (sin leakage)."""
    tr2, va2 = F.apply_fold_encodings(tr, va)
    tr2 = F.cast_categoricals(tr2)
    va2 = F.cast_categoricals(va2)

    img_cols = list(img_all.columns)
    medians = img_all.reindex(tr["zpid"].values).median()
    img_tr = _image_block(tr["zpid"].values, img_all, medians, tr2.index)
    img_va = _image_block(va["zpid"].values, img_all, medians, va2.index)

    txt_tr, store = FTL.fit_features("train", tr, seed=seed, **LLM_KWARGS)
    txt_va = FTL.transform_features("train", va, store)
    txt_tr.index = tr2.index
    txt_va.index = va2.index

    x_tr = pd.concat([tr2[TAB_COLS], img_tr[img_cols], txt_tr], axis=1)
    x_va = pd.concat([va2[TAB_COLS], img_va[img_cols], txt_va], axis=1)
    return x_tr, x_va, store


def assemble_full(
    train: pd.DataFrame,
    test: pd.DataFrame,
    img_train: pd.DataFrame,
    img_test: pd.DataFrame,
    seed: int,
):
    """Devuelve (X_train, X_test) fusionadas para el fit full -> submission."""
    tr2, te2 = F.apply_full_encodings(train, test)
    tr2 = F.cast_categoricals(tr2)
    te2 = F.cast_categoricals(te2)

    img_cols = list(img_train.columns)
    medians = img_train.reindex(train["zpid"].values).median()
    img_tr = _image_block(train["zpid"].values, img_train, medians, tr2.index)
    img_te = _image_block(test["zpid"].values, img_test, medians, te2.index)

    txt_tr, store = FTL.fit_features("train", train, seed=seed, **LLM_KWARGS)
    txt_te = FTL.transform_features("test", test, store)
    txt_tr.index = tr2.index
    txt_te.index = te2.index

    x_tr = pd.concat([tr2[TAB_COLS], img_tr[img_cols], txt_tr], axis=1)
    x_te = pd.concat([te2[TAB_COLS], img_te[img_cols], txt_te], axis=1)
    return x_tr, x_te
