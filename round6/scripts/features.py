"""
Feature engineering compartido entre train y test (round 1).

Encoding por zona: solo cuantiles ZIP (p10/p50/p90) + KNN espacial + ratios fiscal/barrio.
Sin medias zip/zip3 redundantes (zipcode categórica ya captura la zona).
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import BallTree

import features_desc_price as FDP

TARGET = "log_price"
PRICE_COL = "lastSoldPrice_hpi_adjusted"

KNN_K = 20
KNN_FEATURE = "knn_mean_log_price"

# Calibración post-modelo (opcional). False = raw LGBM → mejor wMAPE en torneo.
USE_CALIBRATION = False

# ── Conservadurismo en la puja ───────────────────────────────────────────────
# Dos formas (usar UNA, no ambas):
#
# 1) PREDICTION_QUANTILE (data-driven): entrena LightGBM para predecir un percentil
#    bajo del precio en vez de la media (objective='quantile'). τ=0.35 ≈ "ofrezco el
#    precio que la propiedad supera con ~65% de confianza". Conservador per-propiedad,
#    aprendido del dato. None = media (L2, comportamiento clásico).
#    Para usarlo: PREDICTION_QUANTILE = 0.35 y CONSERVATIVE_SCALE = 1.0. Requiere retrain.
#
# 2) CONSERVATIVE_SCALE (manual global): predicted_price *= k (k<1). Baja todas las
#    pujas por igual. Practice sim (OOF): ROI 7% (k=1) → 73% (k=0.90), Sharpe 0.6 → 3.4.
#    No requiere retrain (post-proceso). 1.0 = desactivado.
PREDICTION_QUANTILE = None
CONSERVATIVE_SCALE = 1.0

SHRINK_ALPHA = 0.88
ANCHOR_KNN_WEIGHT = 0.35
CLIP_SOFT_RATE_BELOW = 0.30   # conservador bajo q10 (distressed / barato)
CLIP_SOFT_RATE_ABOVE = 0.55   # suave arriba q90 (luxury / costero)

ZIP_Q10 = "zip_q10_log_price"
ZIP_Q50 = "zip_q50_log_price"
ZIP_Q90 = "zip_q90_log_price"
ZONE_QUANTILE_COL = "zipcode"

ATLANTIC_COAST_LON = -80.08
COAST_KM_THRESHOLD = 3.0

EXCLUDE = {
    "zpid",
    "lastSoldPrice_hpi_adjusted",
    "log_price",
    "description",
    "tag_price_cut",
}

RAW_FEATURES = [
    "bedrooms", "bathrooms", "livingArea", "yearBuilt", "lotAreaValue", "photoCount",
    "latitude", "longitude", "zipcode", "homeType",
    "taxAssessedValue", "propertyTaxRate", "latest_tax_value", "latest_tax_paid",
    "num_tax_records",
    "num_sales", "num_price_changes", "last_listing_price",
    "avg_school_rating", "max_school_rating", "min_school_distance",
    "has_hoa", "hoa_fee_monthly", "has_pool", "has_garage", "has_waterfront",
    "property_age", "bath_to_bed_ratio", "log_living_area", "log_lot_area", "zip_3digit",
    "desc_length", "desc_word_count", "desc_is_boilerplate",
    "desc_mentions_renovated", "desc_mentions_pool", "desc_mentions_view",
]

ENGINEERED_FEATURES = [
    "dist_to_coast_km", "living_per_bedroom", "tax_per_sqft",
    "log_last_listing", "log_tax_assessed", "listing_to_tax", "has_last_listing",
]

# Features de fotos derivadas de tabular (independientes del fold).
PHOTO_FEATURES = [
    "log_photo_count", "photos_per_100sqft", "photo_sparse_x_large_home",
]
# Feature de fotos relativa al ZIP (fiteada solo en train fold, OOF-safe).
PHOTO_ZONE_FEATURES = ["photo_vs_zip_median"]

# Features lexicas de banda de precio desde la descripcion (regex, sin LLM).
DESC_PRICE_FEATURES = list(FDP.FEATURE_COLUMNS)

ZONE_QUANTILES = [0.10, 0.50, 0.90]
ZONE_QUANTILE_SMOOTHING = 20.0

ZONE_RATIO_FEATURES = [
    "tax_to_zip_q50", "listing_to_zip_q50", "zip_log_spread", "coastal_living_area",
]


def _quantile_feature_names(prefix: str) -> list[str]:
    return [f"{prefix}_q{int(q * 100)}_log_price" for q in ZONE_QUANTILES]


def zone_encoding_features() -> list[str]:
    """Cuantiles ZIP + KNN + ratios post-encoding."""
    return _quantile_feature_names("zip") + [KNN_FEATURE] + ZONE_RATIO_FEATURES


TARGET_ENCODING_FEATURES = zone_encoding_features()
CATEGORICAL_FEATURES = ["homeType", "zipcode", "zip_3digit"]


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["log_living_area"] = np.log1p(out["livingArea"].fillna(0).clip(lower=0))
    out["log_lot_area"] = np.log1p(out["lotAreaValue"].fillna(0).clip(lower=0))

    if "property_age" not in out.columns:
        out["property_age"] = 2024 - out["yearBuilt"]
    out["property_age"] = out["property_age"].fillna(out["property_age"].median())

    out["bath_to_bed_ratio"] = out["bathrooms"] / out["bedrooms"].replace(0, np.nan)
    out["bath_to_bed_ratio"] = out["bath_to_bed_ratio"].fillna(1.0)
    out["zip_3digit"] = (out["zipcode"] // 100).astype(int)

    out["dist_to_coast_km"] = _haversine_km(
        out["latitude"], out["longitude"],
        out["latitude"], np.full(len(out), ATLANTIC_COAST_LON),
    )

    out["living_per_bedroom"] = out["livingArea"] / out["bedrooms"].replace(0, np.nan)
    out["living_per_bedroom"] = out["living_per_bedroom"].fillna(out["livingArea"])

    out["tax_per_sqft"] = out["taxAssessedValue"] / out["livingArea"].replace(0, np.nan)
    out["tax_per_sqft"] = out["tax_per_sqft"].replace([np.inf, -np.inf], np.nan)

    listing = out["last_listing_price"]
    tax = out["taxAssessedValue"]
    out["log_last_listing"] = np.log1p(listing.fillna(0).clip(lower=0))
    out["log_tax_assessed"] = np.log1p(tax.fillna(0).clip(lower=0))
    out["has_last_listing"] = (listing.notna() & (listing > 0)).astype(int)
    out["listing_to_tax"] = listing / tax.replace(0, np.nan)
    out["listing_to_tax"] = out["listing_to_tax"].replace([np.inf, -np.inf], np.nan)

    # ── Fotos <-> precio (parte no-zonal, sin leakage) ────────────────────────
    photos = out["photoCount"].fillna(0).clip(lower=0)
    out["log_photo_count"] = np.log1p(photos)
    out["photos_per_100sqft"] = photos / (out["livingArea"].replace(0, np.nan) / 100.0)
    out["photos_per_100sqft"] = out["photos_per_100sqft"].replace([np.inf, -np.inf], np.nan)
    # Casas grandes con pocas fotos suelen estar mal listadas -> sesgo.
    few_photos = (photos <= 5).astype(float)
    out["photo_sparse_x_large_home"] = few_photos * out["log_living_area"]

    # ── Banda de precio desde la descripcion (regex, sin LLM, sin leakage) ────
    desc = FDP.build(out)
    for col in FDP.FEATURE_COLUMNS:
        out[col] = desc[col].values

    return out


def fit_zone_quantiles(
    train_df: pd.DataFrame,
    col: str,
    smoothing: float = ZONE_QUANTILE_SMOOTHING,
) -> tuple[dict[float, pd.Series], dict[float, float]]:
    global_qs = {q: float(train_df[TARGET].quantile(q)) for q in ZONE_QUANTILES}
    counts = train_df.groupby(col)[TARGET].count()
    raw = train_df.groupby(col)[TARGET].quantile(ZONE_QUANTILES).unstack()

    mappings: dict[float, pd.Series] = {}
    for q in ZONE_QUANTILES:
        zone_q = raw[q].reindex(counts.index)
        smooth = (counts * zone_q + smoothing * global_qs[q]) / (counts + smoothing)
        mappings[q] = smooth.fillna(global_qs[q])
    return mappings, global_qs


def apply_zone_quantile(
    df: pd.DataFrame, col: str, mapping: pd.Series, global_q: float
) -> pd.Series:
    return df[col].map(mapping).fillna(global_q)


def fit_photo_zone_median(
    train_df: pd.DataFrame, col: str = ZONE_QUANTILE_COL,
    smoothing: float = ZONE_QUANTILE_SMOOTHING,
) -> tuple[pd.Series, float]:
    """Mediana de photoCount por ZIP, suavizada hacia la global. OOF-safe (solo train)."""
    photos = train_df["photoCount"].fillna(0).clip(lower=0)
    global_med = float(photos.median())
    tmp = pd.DataFrame({col: train_df[col].values, "p": photos.values})
    counts = tmp.groupby(col)["p"].count()
    zone_med = tmp.groupby(col)["p"].median()
    smooth = (counts * zone_med + smoothing * global_med) / (counts + smoothing)
    return smooth.fillna(global_med), global_med


def apply_photo_zone_median(
    df: pd.DataFrame, mapping: pd.Series, global_med: float,
    col: str = ZONE_QUANTILE_COL,
) -> pd.Series:
    zip_med = df[col].map(mapping).fillna(global_med)
    zip_med = zip_med.clip(lower=1.0)
    photos = df["photoCount"].fillna(0).clip(lower=0)
    ratio = (photos / zip_med.values).replace([np.inf, -np.inf], np.nan)
    return ratio.fillna(1.0)


def _apply_zone_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    zip_q50_price = np.maximum(np.expm1(out[ZIP_Q50].astype(np.float64)), 1.0)

    out["tax_to_zip_q50"] = out["taxAssessedValue"] / zip_q50_price
    out["tax_to_zip_q50"] = out["tax_to_zip_q50"].replace([np.inf, -np.inf], np.nan)

    listing = out["last_listing_price"]
    out["listing_to_zip_q50"] = listing / zip_q50_price
    out["listing_to_zip_q50"] = out["listing_to_zip_q50"].replace([np.inf, -np.inf], np.nan)

    out["zip_log_spread"] = out[ZIP_Q90] - out[ZIP_Q10]

    coastal = (out["dist_to_coast_km"] <= COAST_KM_THRESHOLD).astype(float)
    out["coastal_living_area"] = coastal * out["livingArea"].fillna(0)
    return out


def _apply_zone_encodings(
    tr: pd.DataFrame, va: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Cuantiles ZIP OOF-safe + ratios fiscal/barrio."""
    q_maps, q_globals = fit_zone_quantiles(tr, ZONE_QUANTILE_COL)
    for q in ZONE_QUANTILES:
        feat = f"zip_q{int(q * 100)}_log_price"
        tr[feat] = apply_zone_quantile(tr, ZONE_QUANTILE_COL, q_maps[q], q_globals[q])
        if va is not None:
            va[feat] = apply_zone_quantile(va, ZONE_QUANTILE_COL, q_maps[q], q_globals[q])

    photo_map, photo_global = fit_photo_zone_median(tr, ZONE_QUANTILE_COL)
    tr["photo_vs_zip_median"] = apply_photo_zone_median(tr, photo_map, photo_global)
    if va is not None:
        va["photo_vs_zip_median"] = apply_photo_zone_median(va, photo_map, photo_global)

    tr = _apply_zone_ratio_features(tr)
    if va is not None:
        va = _apply_zone_ratio_features(va)
    return tr, va


def _coords_rad(df: pd.DataFrame) -> np.ndarray:
    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    return np.radians(np.column_stack([lat, lon]))


def compute_knn_mean_log_price(
    query_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    k: int = KNN_K,
    exclude_loo: bool = False,
) -> np.ndarray:
    n_ref = len(ref_df)
    if n_ref == 0:
        return np.zeros(len(query_df))

    k_query = min(k + 1 if exclude_loo else k, n_ref)
    log_prices = ref_df[TARGET].values.astype(np.float64)
    global_mean = float(log_prices.mean())

    ref_coords = _coords_rad(ref_df)
    query_coords = _coords_rad(query_df)
    tree = BallTree(ref_coords, metric="haversine")
    _, ind = tree.query(query_coords, k=k_query)

    if k_query == 1 and exclude_loo:
        return np.full(len(query_df), global_mean)

    if exclude_loo:
        out = np.empty(len(query_df))
        for j in range(len(query_df)):
            nbrs = ind[j]
            nbrs = nbrs[nbrs != j][:k]
            out[j] = log_prices[nbrs].mean() if len(nbrs) else global_mean
        return out

    if k_query == 1:
        return log_prices[ind.ravel()]
    return log_prices[ind].mean(axis=1)


def apply_fold_encodings(tr: pd.DataFrame, va: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = tr.copy()
    va = va.copy()
    tr, va = _apply_zone_encodings(tr, va)
    tr[KNN_FEATURE] = compute_knn_mean_log_price(tr, tr, exclude_loo=True)
    va[KNN_FEATURE] = compute_knn_mean_log_price(va, tr, exclude_loo=False)
    return tr, va


def apply_full_encodings(
    train_df: pd.DataFrame, test_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    train = train_df.copy()
    test = test_df.copy() if test_df is not None else None
    train, test = _apply_zone_encodings(train, test)
    train[KNN_FEATURE] = compute_knn_mean_log_price(train, train, exclude_loo=True)
    if test is not None:
        test[KNN_FEATURE] = compute_knn_mean_log_price(test, train, exclude_loo=False)
    return train, test


def conservative_anchor(df: pd.DataFrame) -> np.ndarray:
    q50 = df[ZIP_Q50].values.astype(np.float64)
    knn = df[KNN_FEATURE].values.astype(np.float64)
    return (1.0 - ANCHOR_KNN_WEIGHT) * q50 + ANCHOR_KNN_WEIGHT * knn


def calibrate_predictions_conservative(
    pred_log: np.ndarray,
    df: pd.DataFrame,
    alpha: float = SHRINK_ALPHA,
) -> np.ndarray:
    """Shrink suave hacia barrio + clip asimétrico (más conservador abajo, suave arriba)."""
    anchor = conservative_anchor(df)
    out = alpha * pred_log + (1.0 - alpha) * anchor

    q10 = df[ZIP_Q10].values.astype(np.float64)
    q90 = df[ZIP_Q90].values.astype(np.float64)

    out = np.where(out > q90, q90 + CLIP_SOFT_RATE_ABOVE * (out - q90), out)
    out = np.where(out < q10, q10 + CLIP_SOFT_RATE_BELOW * (out - q10), out)
    return out


def apply_conservative_scale(pred_log: np.ndarray) -> np.ndarray:
    """Escala el precio final por CONSERVATIVE_SCALE (en espacio precio, no log)."""
    if CONSERVATIVE_SCALE == 1.0:
        return pred_log
    return np.log1p(CONSERVATIVE_SCALE * np.expm1(pred_log))


def fit_debias(pred_log: np.ndarray, real_log: np.ndarray) -> dict:
    """Ajusta un mapa monótono real_log ~ f(pred_log) sobre OOF para contrarrestar la
    regresión a la media (sube las caras, baja las baratas). Isotónica por defecto;
    fallback lineal (OLS) si hay muy pocos puntos o pred_log es casi constante."""
    pred_log = np.asarray(pred_log, dtype=np.float64)
    real_log = np.asarray(real_log, dtype=np.float64)
    mask = np.isfinite(pred_log) & np.isfinite(real_log)
    pred_log, real_log = pred_log[mask], real_log[mask]

    if len(pred_log) < 50 or np.ptp(pred_log) < 1e-9:
        b = float(np.polyfit(pred_log, real_log, 1)[0]) if len(pred_log) >= 2 else 1.0
        a = float(real_log.mean() - b * pred_log.mean()) if len(pred_log) else 0.0
        return {"method": "linear", "a": a, "b": b}

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(pred_log, real_log)
    return {
        "method": "isotonic",
        "x": [float(v) for v in iso.X_thresholds_],
        "y": [float(v) for v in iso.y_thresholds_],
    }


def apply_debias(pred_log: np.ndarray, params: dict | None) -> np.ndarray:
    """Aplica el mapa de de-sesgo ajustado por fit_debias. params=None → sin cambios."""
    if not params:
        return pred_log
    pred_log = np.asarray(pred_log, dtype=np.float64)
    if params.get("method") == "linear":
        return params["a"] + params["b"] * pred_log
    x = np.asarray(params["x"], dtype=np.float64)
    y = np.asarray(params["y"], dtype=np.float64)
    return np.interp(pred_log, x, y)


def finalize_predictions(
    pred_log: np.ndarray, df: pd.DataFrame, debias: dict | None = None
) -> np.ndarray:
    """De-sesgo opcional (debias) + calibración opcional (USE_CALIBRATION) + escala (CONSERVATIVE_SCALE)."""
    out = calibrate_predictions_conservative(pred_log, df) if USE_CALIBRATION else pred_log
    out = apply_debias(out, debias)
    return apply_conservative_scale(out)


def blend_predictions(
    pred_log: np.ndarray, df: pd.DataFrame, alpha: float = SHRINK_ALPHA
) -> np.ndarray:
    return calibrate_predictions_conservative(pred_log, df, alpha=alpha)


def cheap_sample_weight(
    real_log: np.ndarray, lo: float = 0.5, hi: float = 2.5, gamma: float = 1.5
) -> np.ndarray:
    """Peso por muestra que sube el gradiente de las casas baratas sin romper las caras.

    Se basa en el rank del precio (robusto a outliers): rank 0 (mas barata) -> peso alto,
    rank 1 (mas cara) -> peso bajo. gamma controla la curvatura; se clipea a [lo, hi] y se
    normaliza a media 1 para no cambiar la escala global del objetivo.
    """
    x = np.asarray(real_log, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.ones(0)
    ranks = pd.Series(x).rank(method="average").values / n  # (0, 1], 1 = mas cara
    w = lo + (hi - lo) * (1.0 - ranks) ** gamma
    w = np.clip(w, lo, hi)
    return w / w.mean()


def model_objective_params() -> dict:
    """Kwargs extra para LGBMRegressor: quantile si PREDICTION_QUANTILE está seteado."""
    if PREDICTION_QUANTILE is None:
        return {}
    return {"objective": "quantile", "alpha": float(PREDICTION_QUANTILE)}


def feature_columns() -> list[str]:
    return (
        RAW_FEATURES
        + ENGINEERED_FEATURES
        + PHOTO_FEATURES
        + PHOTO_ZONE_FEATURES
        + DESC_PRICE_FEATURES
        + TARGET_ENCODING_FEATURES
    )


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df
