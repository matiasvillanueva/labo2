"""
Entrena LightGBM sobre train_processed.csv con validación K-fold multi-seed.

Para cada configuración de hiperparámetros se promedia el MAE (en dólares) sobre
varias seeds y 10 folds, de modo que el resultado no dependa de un único split.
El target encoding por zona se calcula out-of-fold para evitar leakage.

El mejor modelo se imprime por consola como JSON y se guarda en
submissions_train/best_model.json SOLO si mejora el MAE del JSON existente.
Así se pueden hacer varias corridas con distintos grids y quedarse con el mejor.

Run from round1/:
    python scripts/train_lgbm.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

import features as F


def log(msg: str) -> None:
    """Print con timestamp y flush inmediato (visible en vivo aunque haya pipe/tee)."""
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

TRAIN_PATH = "data/tabular/train_processed.csv"
OUTPUT_DIR = Path("submissions_train")
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.json"

N_FOLDS = 10
SEEDS = [42, 99, 123, 456, 789]

# True = re-evaluar SOLO el modelo ganador guardado en best_model.json (rápido, para
# regenerar el pipeline). False = barrer todos los PARAM_CONFIGS (exploración completa).
ONLY_WINNER = True

# ── Corrida de exploración (10h) — SOLO params nuevos, sin repetir lo ya usado ──
# Umbral a batir: val_mae < 114124 (tight_cols_1500, ya en best_model.json).
# Hallazgo: subsample solo actúa con subsample_freq>0 (default 0) → todo el tuneo
# previo de subsample fue inerte. Acá se exploran de verdad: subsample_freq, max_depth,
# min_child_weight, min_split_gain, extra_trees, max_bin, num_leaves altos, lr bajo+más
# árboles, y combos. NO se repite tight_cols_1500 (es el baseline en best_model.json).
_BASE = {
    "n_estimators": 1500, "learning_rate": 0.015, "num_leaves": 47,
    "min_child_samples": 35, "colsample_bytree": 0.55,
    "reg_alpha": 1.5, "reg_lambda": 3.5,
}

PARAM_CONFIGS = []


def _cfg(name: str, **over):
    PARAM_CONFIGS.append({"name": name, "params": {**_BASE, **over}})


# A) Bagging REAL (subsample_freq>0) — antes estaba apagado.
_cfg("bag_sf1_sub80", subsample_freq=1, subsample=0.80)
_cfg("bag_sf1_sub70", subsample_freq=1, subsample=0.70)
_cfg("bag_sf1_sub60", subsample_freq=1, subsample=0.60)
_cfg("bag_sf1_sub50", subsample_freq=1, subsample=0.50)
_cfg("bag_sf3_sub75", subsample_freq=3, subsample=0.75)
_cfg("bag_sf5_sub70", subsample_freq=5, subsample=0.70)

# B) Learning rate bajo + más árboles (mejor convergencia).
_cfg("lr010_n2000", learning_rate=0.010, n_estimators=2000)
_cfg("lr008_n2500", learning_rate=0.008, n_estimators=2500)
_cfg("lr006_n3000", learning_rate=0.006, n_estimators=3000)
_cfg("lr012_n1800", learning_rate=0.012, n_estimators=1800)
_cfg("lr020_n1000", learning_rate=0.020, n_estimators=1000)

# C) num_leaves (con regularización acorde).
_cfg("leaves31", num_leaves=31, min_child_samples=25)
_cfg("leaves63", num_leaves=63, min_child_samples=45, reg_alpha=2.5, reg_lambda=5.0)
_cfg("leaves95", num_leaves=95, min_child_samples=60, reg_alpha=3.5, reg_lambda=7.0)
_cfg("leaves127", num_leaves=127, min_child_samples=80, reg_alpha=5.0, reg_lambda=8.0)

# D) max_depth (nunca lo limitamos).
_cfg("depth5_l31", max_depth=5, num_leaves=31)
_cfg("depth6", max_depth=6)
_cfg("depth8_l63", max_depth=8, num_leaves=63, reg_alpha=2.0, reg_lambda=5.0)
_cfg("depth10_l127", max_depth=10, num_leaves=127, min_child_samples=60, reg_alpha=4.0, reg_lambda=7.0)

# E) min_child_weight (min_sum_hessian) — regularización de hoja nueva.
_cfg("mcw_001", min_child_weight=0.01)
_cfg("mcw_01", min_child_weight=0.1)
_cfg("mcw_1", min_child_weight=1.0)
_cfg("mcw_5", min_child_weight=5.0)

# F) min_split_gain (min_gain_to_split) — poda por ganancia mínima.
_cfg("msg_001", min_split_gain=0.01)
_cfg("msg_01", min_split_gain=0.1)
_cfg("msg_05", min_split_gain=0.5)

# G) extra_trees (splits extremadamente aleatorizados → menos varianza).
_cfg("extratrees", extra_trees=True)
_cfg("extratrees_l63", extra_trees=True, num_leaves=63, reg_alpha=2.0, reg_lambda=5.0)
_cfg("extratrees_n2000", extra_trees=True, n_estimators=2000, learning_rate=0.010)

# H) max_bin (resolución de histograma).
_cfg("maxbin63", max_bin=63)
_cfg("maxbin127", max_bin=127)
_cfg("maxbin511", max_bin=511)

# I) colsample_bytree fuera de 0.55.
_cfg("col040", colsample_bytree=0.40)
_cfg("col070", colsample_bytree=0.70)
_cfg("col085", colsample_bytree=0.85)

# J) Regularización en regiones nuevas.
_cfg("reg_light", reg_alpha=0.5, reg_lambda=1.0)
_cfg("reg_l1heavy", reg_alpha=8.0, reg_lambda=3.5)
_cfg("reg_l2heavy", reg_alpha=1.5, reg_lambda=12.0)
_cfg("reg_both_heavy", reg_alpha=5.0, reg_lambda=10.0)

# K) Combos prometedores (bagging real + lr bajo + leaves/depth).
_cfg("combo_bag_lr010_l63", subsample_freq=1, subsample=0.70,
     learning_rate=0.010, n_estimators=2200, num_leaves=63, reg_alpha=2.0, reg_lambda=5.0)
_cfg("combo_bag_depth8", subsample_freq=1, subsample=0.70,
     max_depth=8, num_leaves=63, reg_alpha=2.0, reg_lambda=5.0)
_cfg("combo_extratrees_bag", extra_trees=True, subsample_freq=1, subsample=0.70)
_cfg("combo_deep_reg", learning_rate=0.010, n_estimators=2500, num_leaves=63,
     min_child_samples=50, reg_alpha=3.0, reg_lambda=6.0, subsample_freq=1, subsample=0.75)
_cfg("combo_msg_mcw", min_split_gain=0.05, min_child_weight=0.1, num_leaves=63,
     reg_alpha=2.0, reg_lambda=5.0)


def evaluate_config(df: pd.DataFrame, params: dict) -> dict:
    """Promedia MAE ($) y R² (log) sobre todas las seeds y folds para una config."""
    feature_cols = F.feature_columns()
    maes, r2s = [], []

    for seed in SEEDS:
        seed_start = time.time()
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_log = np.zeros(len(df))

        for tr_idx, va_idx in kf.split(df):
            tr = df.iloc[tr_idx].copy()
            va = df.iloc[va_idx].copy()
            tr, va = F.apply_fold_encodings(tr, va)

            tr = F.cast_categoricals(tr)
            va = F.cast_categoricals(va)

            model = lgb.LGBMRegressor(
                **params, **F.model_objective_params(),
                random_state=seed, n_jobs=-1, verbosity=-1,
            )
            model.fit(
                tr[feature_cols], tr[F.TARGET],
                categorical_feature=F.CATEGORICAL_FEATURES,
            )
            oof_log[va_idx] = model.predict(va[feature_cols])

        pred_price = np.expm1(oof_log)
        seed_mae = mean_absolute_error(df[F.PRICE_COL], pred_price)
        maes.append(seed_mae)
        r2s.append(r2_score(df[F.TARGET], oof_log))
        log(f"    seed {seed}: MAE ${seed_mae:,.0f} ({time.time() - seed_start:.0f}s)")

    return {
        "val_mae": float(np.mean(maes)),
        "val_mae_std": float(np.std(maes)),
        "val_r2_log": float(np.mean(r2s)),
    }


def select_configs() -> list[dict]:
    """Devuelve la lista de configs a evaluar: solo el ganador (ONLY_WINNER) o el grid completo."""
    if ONLY_WINNER and BEST_MODEL_PATH.exists():
        with open(BEST_MODEL_PATH) as f:
            winner = json.load(f)
        cfg = {"name": winner.get("config_name", "winner"), "params": winner["lgbm_params"]}
        log(f"Modo ONLY_WINNER: solo el ganador '{cfg['name']}' "
            f"(sin barrer los {len(PARAM_CONFIGS)} configs del grid).")
        return [cfg]
    if ONLY_WINNER:
        log(f"AVISO: ONLY_WINNER=True pero no existe {BEST_MODEL_PATH}; "
            f"se barre el grid completo ({len(PARAM_CONFIGS)} configs).")
    return PARAM_CONFIGS


def main():
    run_start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = F.build_features(pd.read_csv(TRAIN_PATH))

    configs = select_configs()
    total_fits = len(configs) * len(SEEDS) * N_FOLDS
    log(f"Train LightGBM — {len(df):,} propiedades × {len(SEEDS)} seeds × {N_FOLDS} folds")
    log(f"Features: {len(F.feature_columns())} | Configs: {len(configs)} | Fits totales: {total_fits}")

    results = []
    for i, config in enumerate(configs, 1):
        log(f"[{i}/{len(configs)}] Config '{config['name']}' → {config['params']}")
        metrics = evaluate_config(df, config["params"])
        results.append({**config, **metrics})
        log(
            f"  ✓ '{config['name']}': MAE ${metrics['val_mae']:,.0f} "
            f"± ${metrics['val_mae_std']:,.0f} | R² {metrics['val_r2_log']:.4f}"
        )

    log(f"Búsqueda terminada en {time.time() - run_start:.0f}s")
    print("\n" + "=" * 56)
    print(f"{'Config':<20} {'Val MAE ($)':>14} {'± std':>10} {'R² (log)':>10}")
    print("-" * 56)
    for r in sorted(results, key=lambda r: r["val_mae"]):
        print(
            f"{r['name']:<20} ${r['val_mae']:>12,.0f} "
            f"${r['val_mae_std']:>8,.0f} {r['val_r2_log']:>10.4f}"
        )

    winner = min(results, key=lambda r: r["val_mae"])
    winner_json = {
        "val_mae": winner["val_mae"],
        "val_mae_std": winner["val_mae_std"],
        "val_r2_log": winner["val_r2_log"],
        "config_name": winner["name"],
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "lgbm_params": winner["params"],
        "features": F.feature_columns(),
        "categorical_features": F.CATEGORICAL_FEATURES,
        "target": F.TARGET,
        "knn_k": F.KNN_K,
        "shrink_alpha": F.SHRINK_ALPHA,
        "anchor_knn_weight": F.ANCHOR_KNN_WEIGHT,
        "clip_soft_rate_below": F.CLIP_SOFT_RATE_BELOW,
        "clip_soft_rate_above": F.CLIP_SOFT_RATE_ABOVE,
        "zone_quantiles": F.ZONE_QUANTILES,
    }

    print("\n" + "=" * 56)
    print("MODELO GANADOR DE ESTA CORRIDA (JSON):")
    print("=" * 56)
    print(json.dumps(winner_json, indent=2))

    # Guardado condicional: solo si mejora el MAE del best_model.json existente.
    previous_mae = None
    if BEST_MODEL_PATH.exists():
        with open(BEST_MODEL_PATH) as f:
            previous_mae = json.load(f).get("val_mae")

    if previous_mae is None:
        with open(BEST_MODEL_PATH, "w") as f:
            json.dump(winner_json, f, indent=2)
        print(f"\nNo había modelo previo. Guardado en {BEST_MODEL_PATH} (MAE ${winner['val_mae']:,.0f}).")
    elif winner["val_mae"] < previous_mae:
        with open(BEST_MODEL_PATH, "w") as f:
            json.dump(winner_json, f, indent=2)
        print(
            f"\nMejora: ${winner['val_mae']:,.0f} < ${previous_mae:,.0f}. "
            f"Actualizado {BEST_MODEL_PATH}."
        )
    else:
        print(
            f"\nSin mejora: ${winner['val_mae']:,.0f} >= ${previous_mae:,.0f}. "
            f"Se mantiene {BEST_MODEL_PATH}."
        )


if __name__ == "__main__":
    main()
