"""
Genera la Practice Submission: predicciones out-of-fold (OOF) sobre el TRAIN.

Salida:
  - submissions_train/practice_submission.csv (calibrado, para subir)
  - submissions_train/comparison_oof_{config}_val{CV}_oof{OOF}_{N}f_raw.csv
  - submissions_train/comparison_oof_{config}_val{CV}_oof{OOF}_{N}f_calibrated.csv

Run from round1/:
    python scripts/predict_train_oof.py
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

TRAIN_PATH = "data/tabular/train_processed.csv"
BEST_MODEL_PATH = Path("submissions_train/best_model.json")
OUTPUT_DIR = Path("submissions_train")
OUTPUT_PATH = OUTPUT_DIR / "practice_submission.csv"
DEBIAS_PATH = OUTPUT_DIR / "debias.json"


def comparison_path(best: dict, oof_mae: float, suffix: str) -> Path:
    val = int(round(best["val_mae"]))
    oof = int(round(oof_mae))
    n_feat = len(F.feature_columns())
    return OUTPUT_DIR / f"comparison_oof_{best['config_name']}_val{val}_oof{oof}_{n_feat}f_{suffix}.csv"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(pred - actual).sum() / np.abs(actual).sum() * 100)


def crossfit_debias(pred_log: np.ndarray, real_log: np.ndarray,
                    n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """De-sesgo cross-fitted (fit en n-1 folds, aplica en el restante) para reportar
    métricas honestas, sin optimismo in-sample del mapa de calibración."""
    out = np.empty_like(pred_log)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(pred_log):
        params = F.fit_debias(pred_log[tr_idx], real_log[tr_idx])
        out[va_idx] = F.apply_debias(pred_log[va_idx], params)
    return out


def report_bias_by_decile(actual: np.ndarray, price_raw: np.ndarray,
                          price_deb: np.ndarray, n: int = 10) -> None:
    """Imprime sesgo% por decil de precio real, antes (raw) vs después (de-sesgo)."""
    d = pd.DataFrame({"real": actual, "raw": price_raw, "deb": price_deb})
    d["dec"] = pd.qcut(d["real"], n, labels=False)
    log("Sesgo% por decil de precio real (raw → de-sesgado):")
    for dec, sub in d.groupby("dec"):
        r = sub["real"].values
        bias_raw = (sub["raw"].values - r).mean() / r.mean() * 100
        bias_deb = (sub["deb"].values - r).mean() / r.mean() * 100
        print(f"    D{dec} | ${r.min():>9,.0f}-${r.max():>9,.0f} | "
              f"{bias_raw:+6.1f}% → {bias_deb:+6.1f}%", flush=True)


def main():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe {BEST_MODEL_PATH}. Corré primero: python scripts/train_lgbm.py"
        )

    with open(BEST_MODEL_PATH) as f:
        best = json.load(f)

    params = best["lgbm_params"]
    feature_cols = F.feature_columns()
    seeds = best["seeds"]
    n_folds = best["n_folds"]
    log(f"Modelo '{best['config_name']}' | Val MAE ${best['val_mae']:,.0f} | "
        f"{len(seeds)} seeds × {n_folds} folds")
    obj = f"quantile τ={F.PREDICTION_QUANTILE}" if F.PREDICTION_QUANTILE is not None else "media"
    log(f"Features: {len(feature_cols)} | objetivo {obj} | salida {'calibrada' if F.USE_CALIBRATION else 'raw'} | "
        f"k={F.CONSERVATIVE_SCALE} | α={F.SHRINK_ALPHA} (si cal)")
    log(f"Params: {params}")

    df = F.build_features(pd.read_csv(TRAIN_PATH))
    n = len(df)
    actual = df[F.PRICE_COL].values

    oof_sum = np.zeros(n)
    zone_cols = [F.KNN_FEATURE, F.ZIP_Q10, F.ZIP_Q50, F.ZIP_Q90]
    zone_sum = {c: np.zeros(n) for c in zone_cols}

    for seed in seeds:
        seed_start = time.time()
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof_log = np.zeros(n)

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
            model.fit(tr[feature_cols], tr[F.TARGET], categorical_feature=F.CATEGORICAL_FEATURES)
            oof_log[va_idx] = model.predict(va[feature_cols])
            for c in zone_cols:
                zone_sum[c][va_idx] += va[c].values

        oof_sum += oof_log
        log(f"    seed {seed}: MAE ${mean_absolute_error(actual, np.expm1(oof_log)):,.0f} "
            f"({time.time() - seed_start:.0f}s)")

    oof_log_raw = oof_sum / len(seeds)
    cal_df = pd.DataFrame({c: zone_sum[c] / len(seeds) for c in zone_cols})

    # De-sesgo: ajusta el mapa monótono sobre OOF y lo persiste para que predict_lgbm
    # lo aplique al test. Contrarresta la regresión a la media (sube caras, baja baratas).
    debias_params = F.fit_debias(oof_log_raw, df[F.TARGET].values)
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(DEBIAS_PATH, "w") as f:
        json.dump(debias_params, f, indent=2)
    log(f"De-sesgo ajustado ({debias_params['method']}) → guardado {DEBIAS_PATH}")

    oof_log_final = F.finalize_predictions(oof_log_raw, cal_df, debias=debias_params)
    # Métrica honesta del de-sesgo (cross-fitted, sin optimismo in-sample).
    oof_log_deb_cf = crossfit_debias(oof_log_raw, df[F.TARGET].values)

    price_raw = np.expm1(oof_log_raw)
    price_final = np.expm1(oof_log_final)
    price_deb_cf = np.expm1(oof_log_deb_cf)

    mae_raw = mean_absolute_error(actual, price_raw)
    mae_final = mean_absolute_error(actual, price_final)
    log(f"OOF raw         → MAE ${mae_raw:,.0f} | wMAPE {wmape(actual, price_raw):.2f}% | "
        f"medAPE {np.median(np.abs(price_raw - actual) / actual * 100):.2f}% | "
        f"R² {r2_score(df[F.TARGET], oof_log_raw):.4f}")
    log(f"OOF de-sesgado(cf) → MAE ${mean_absolute_error(actual, price_deb_cf):,.0f} | "
        f"wMAPE {wmape(actual, price_deb_cf):.2f}% | "
        f"medAPE {np.median(np.abs(price_deb_cf - actual) / actual * 100):.2f}% | "
        f"R² {r2_score(df[F.TARGET], oof_log_deb_cf):.4f}")
    log(f"OOF FINAL ({'cal+' if F.USE_CALIBRATION else ''}debias, k={F.CONSERVATIVE_SCALE}) "
        f"→ MAE ${mae_final:,.0f} | wMAPE {wmape(actual, price_final):.2f}%")
    report_bias_by_decile(actual, price_raw, price_deb_cf)

    comparison = pd.DataFrame({
        "zpid": df["zpid"],
        "valor_real": actual,
        "valor_calculado_raw": price_raw,
        "valor_calculado": price_final,
        "error_raw": price_raw - actual,
        "error": price_final - actual,
        "error_pct_raw": np.abs(price_raw - actual) / actual * 100,
        "error_pct": np.abs(price_final - actual) / actual * 100,
    })

    submission = comparison[["zpid", "valor_calculado"]].rename(
        columns={"valor_calculado": "predicted_price"}
    )
    suffix = "calibrated" if F.USE_CALIBRATION else "debiased"
    OUTPUT_DIR.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    comparison.to_csv(comparison_path(best, mae_final, suffix), index=False)

    log(f"Guardado {OUTPUT_PATH} — practice ({suffix})")
    log(f"Guardado {comparison_path(best, mae_final, suffix)}")
    print(submission["predicted_price"].describe())


if __name__ == "__main__":
    main()
