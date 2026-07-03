"""
Round 2 — Agregacion de features por propiedad (zpid) a partir del cache de
embeddings de imagen. Importado por train_oof.py / predict_test.py / analyze_signal.py.

Entorno: el que tiene lightgbm + sklearn (p.ej. envs/labo2). NO importa torch.

Construye por zpid (solo desde imagenes):
  - pooled embedding global + pools por tipo (exterior / interior) + foto principal
    reducidos con PCA (fit en train, transform en test)
  - composicion de tipo (fracciones, n_images, has_satellite, entropia)
  - agregados de materiales (mean/max)
  - estadisticas del sufijo del filename (el "otro id")
  - diversidad visual (std de embeddings) y cantidad total de fotos (sin capear)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

EMB_DIR = Path("embeddings")
PARTICIPANT = Path("../participant")
EMB_DIM = 768  # DINOv2-base

# Deben coincidir con extract_embeddings.py
TYPE_KEYS = ["interior", "exterior", "satellite", "floorplan"]
MAT_KEYS = ["granite", "stainless", "hardwood", "tile", "modern_kitchen",
            "luxury", "outdated", "pool", "waterfront", "spacious"]

PCA_GLOBAL = 128
PCA_EXT = 32
PCA_INT = 32
PCA_MAIN = 64
PCA_IE = 32      # pool interior+exterior (sin satelite/plano)
PCA_SAT = 16     # pool solo satelite
SEED = 42


def _emb_path(split):
    return EMB_DIR / f"img_emb_{split}.npy", EMB_DIR / f"img_meta_{split}.csv"


def _full_meta_path(split):
    return PARTICIPANT / "data" / f"{split}_photo_metadata.csv"


def _group_mean(emb, zpids):
    df = pd.DataFrame(emb, index=zpids)
    return df.groupby(level=0).mean()


def _masked_pool(emb, meta, type_name, fallback):
    m = (meta["type_argmax"] == type_name).values
    if m.sum() == 0:
        return fallback.copy()
    pooled = _group_mean(emb[m], meta.loc[m, "zpid"].values)
    pooled = pooled.reindex(fallback.index).fillna(fallback)
    return pooled


def _main_pool(emb, meta, fallback):
    """Embedding de la foto principal (menor image_index) por propiedad."""
    pos = meta.sort_values("image_index").drop_duplicates("zpid", keep="first")
    main = pd.DataFrame(emb[pos.index.values], index=pos["zpid"].values)
    main = main.reindex(fallback.index).fillna(fallback)
    return main


def _ie_pool(emb, meta, fallback):
    """Promedio solo fotos interior/exterior (excluye satelite y floorplan)."""
    m = meta["type_argmax"].isin(["interior", "exterior"]).values
    if m.sum() == 0:
        return fallback.copy()
    pooled = _group_mean(emb[m], meta.loc[m, "zpid"].values)
    return pooled.reindex(fallback.index).fillna(fallback)


def _sat_pool(emb, meta, fallback):
    m = (meta["type_argmax"] == "satellite").values
    if m.sum() == 0:
        return pd.DataFrame(0.0, index=fallback.index, columns=range(emb.shape[1]))
    pooled = _group_mean(emb[m], meta.loc[m, "zpid"].values)
    return pooled.reindex(fallback.index).fillna(0)


def _listing_flags(meta, index):
    """Flags de listados escasos / satelite — patron principal de sobreestimacion."""
    g = meta.groupby("zpid")
    n = g.size().reindex(index).fillna(0)
    types = meta.groupby("zpid")["type_argmax"]
    all_sat = types.apply(lambda s: (s == "satellite").all()).reindex(index).fillna(0).astype(int)
    all_fp = types.apply(lambda s: (s == "floorplan").all()).reindex(index).fillna(0).astype(int)
    main_type = (meta.sort_values("image_index")
                   .drop_duplicates("zpid", keep="first")
                   .set_index("zpid")["type_argmax"])
    single_ext = ((n == 1) & (main_type.reindex(index) == "exterior")).fillna(0).astype(int)
    single_sat = ((n == 1) & (main_type.reindex(index) == "satellite")).fillna(0).astype(int)

    out = pd.DataFrame(index=index)
    out["single_photo"] = (n == 1).astype(int)
    out["sparse_listing"] = (n <= 3).astype(int)
    out["few_photos"] = (n <= 5).astype(int)
    out["log_n_images"] = np.log1p(n)
    out["satellite_only"] = all_sat
    out["floorplan_only"] = all_fp
    out["single_exterior"] = single_ext
    out["single_satellite"] = single_sat
    return out.fillna(0)


def _diversity(emb, zpids):
    """std de los embeddings entre fotos (promedio por dimension) -> 1 escalar/zpid."""
    df = pd.DataFrame(emb, index=zpids)
    std = df.groupby(level=0).std()
    return std.mean(axis=1)


def _scalars(meta, index):
    g = meta.copy()
    for k in TYPE_KEYS:
        g[f"is_{k}"] = (g["type_argmax"] == k).astype(float)
    grp = g.groupby("zpid")

    out = pd.DataFrame(index=index)
    out["n_images"] = grp.size()
    for k in TYPE_KEYS:
        out[f"frac_{k}"] = grp[f"is_{k}"].mean()
    out["has_satellite"] = (grp["is_satellite"].max() > 0).astype(int)
    for k in TYPE_KEYS:
        out[f"prob_{k}_mean"] = grp[f"prob_{k}"].mean()
    for k in MAT_KEYS:
        out[f"mat_{k}_mean"] = grp[f"mat_{k}"].mean()
        out[f"mat_{k}_max"] = grp[f"mat_{k}"].max()

    suf = grp["suffix"].agg(["mean", "min", "max", "std"])
    suf.columns = [f"suffix_{c}" for c in suf.columns]
    out = out.join(suf)
    main = (g.sort_values("image_index")
              .drop_duplicates("zpid", keep="first")
              .set_index("zpid")["suffix"].rename("suffix_main"))
    out = out.join(main)

    # Entropia de la distribucion de tipos (0 = todas iguales, alto = variado)
    fr = out[[f"frac_{k}" for k in TYPE_KEYS]].values
    out["type_entropy"] = -(np.where(fr > 0, fr * np.log(fr + 1e-9), 0)).sum(axis=1)
    return out.reindex(index).fillna(0)


def pooled_matrices(split):
    """Devuelve (g, ext, int_, ie, sat, main, scal, flags) por zpid."""
    emb_path, meta_path = _emb_path(split)
    if not emb_path.exists():
        raise FileNotFoundError(
            f"No existe {emb_path}. Corre primero scripts/extract_embeddings.py")
    emb = np.load(emb_path).astype(np.float32)
    meta = pd.read_csv(meta_path)

    ok = meta["ok"] == 1
    emb = emb[ok.values]
    meta = meta[ok].reset_index(drop=True)

    g = _group_mean(emb, meta["zpid"].values)
    g.index.name = "zpid"
    ext = _masked_pool(emb, meta, "exterior", g)
    int_ = _masked_pool(emb, meta, "interior", g)
    ie = _ie_pool(emb, meta, g)
    sat = _sat_pool(emb, meta, g)
    main = _main_pool(emb, meta, g)

    scal = _scalars(meta, g.index)
    scal["emb_std"] = _diversity(emb, meta["zpid"].values).reindex(g.index).fillna(0)
    full = pd.read_csv(_full_meta_path(split), usecols=["zpid"])
    n_total = full.groupby("zpid").size().reindex(g.index)
    scal["n_photos_total"] = n_total.fillna(scal["n_images"])
    scal["frac_ie"] = 1.0 - scal["frac_satellite"] - scal["frac_floorplan"]

    flags = _listing_flags(meta, g.index)
    return g, ext, int_, ie, sat, main, scal, flags


def _pca_fit(name, df, n_comp, store):
    n_comp = min(n_comp, df.shape[0], df.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED).fit(df.values)
    store[name] = pca
    cols = [f"{name}_pc{i}" for i in range(n_comp)]
    return pd.DataFrame(pca.transform(df.values), index=df.index, columns=cols)


def _pca_apply(name, df, store):
    pca = store[name]
    cols = [f"{name}_pc{i}" for i in range(pca.n_components_)]
    return pd.DataFrame(pca.transform(df.values), index=df.index, columns=cols)


def fit_features(pooled):
    g, ext, int_, ie, sat, main, scal, flags = pooled
    store = {}
    parts = [
        _pca_fit("g", g, PCA_GLOBAL, store),
        _pca_fit("ext", ext, PCA_EXT, store),
        _pca_fit("int", int_, PCA_INT, store),
        _pca_fit("ie", ie, PCA_IE, store),
        _pca_fit("sat", sat, PCA_SAT, store),
        _pca_fit("main", main, PCA_MAIN, store),
        scal,
        flags,
    ]
    feats = pd.concat(parts, axis=1)
    ev = sum(store[k].explained_variance_ratio_.sum() for k in store) / len(store)
    print(f"[features] {feats.shape[1]} cols | PCA var media {ev:.1%} | "
          f"{feats.shape[0]:,} propiedades")
    return feats, store


def transform_features(pooled, store):
    g, ext, int_, ie, sat, main, scal, flags = pooled
    parts = [
        _pca_apply("g", g, store),
        _pca_apply("ext", ext, store),
        _pca_apply("int", int_, store),
        _pca_apply("ie", ie, store),
        _pca_apply("sat", sat, store),
        _pca_apply("main", main, store),
        scal,
        flags,
    ]
    return pd.concat(parts, axis=1)
