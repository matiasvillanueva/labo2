"""
Round 3 — OOF (practice). Modelo FINAL unico, sin variantes:

  tabular conservador (cuantil tau=0.35)  +  ajuste de imagen acotado (residual con cap+gate)

La imagen no predice precio desde cero: corrige el residual que el tabular no ve, con un
limite de +-8% y un gate que la apaga en listados escasos / satelite. Validado por ROI sim
del juego (game_mechanics_es.md): supera al tabular solo en ROI, win-rate y wMAPE.

Doble OOF (sin leakage): el residual se entrena sobre (real - tabular_oof), nunca sobre
predicciones in-sample.

Salidas (submissions_train/):
  - practice_submission.csv   OOF final (refined) para todos los zpid de train
  - comparison_oof.csv          real, tabular, refined, delta, gate, flags
  - oof_tabular.csv             zpid, tab_oof_log  (lo usa predict_test para el residual)
  - refined_config.json         tau, cap, alpha + metricas OOF (baseline vs refined)

Run from round3/:
    /home/matias/miniconda3/envs/labo2/bin/python scripts/train_oof.py
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
import features_img_residual as IR
from simulate_roi import simulate

# ── Config FINAL (sin variantes) ─────────────────────────────────────────────
TRAIN_PATH = "data/tabular/train_processed.csv"
OOF_DIR = Path("submissions_train")

SEEDS = [42, 99, 123, 456, 789]
N_FOLDS = 5
TAU = 0.35          # conservadurismo por-cuantil (maximiza ROI sim)
CAP = 0.08          # ajuste de imagen acotado a +-8% en log-precio
ALPHA = 1.0         # ganancia del residual

# Tabular: params ganadores round1 (col040), objetivo cuantil.
TABULAR_PARAMS = dict(
    n_estimators=1500, learning_rate=0.015, num_leaves=47,
    min_child_samples=35, colsample_bytree=0.40,
    reg_alpha=1.5, reg_lambda=3.5,
)
# Residual imagen: modelo chico y regularizado (solo corrige).
RESIDUAL_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=15,
    min_child_samples=50, colsample_bytree=0.8,
    reg_alpha=2.0, reg_lambda=4.0,
)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def wmape(a, p):
    return float(np.abs(p - a).sum() / np.abs(a).sum() * 100)


def roi(price, actual):
    return simulate(price, actual, 1.0, np.random.default_rng(12345))


def report_slices(actual, base_price, refined, gate):
    d = pd.DataFrame({"real": actual, "base": base_price, "ref": refined, "gate": gate})
    d["dec"] = pd.qcut(actual, 10, labels=False)
    log("Sesgo% por decil (tabular -> refined):")
    for dec, s in d.groupby("dec"):
        r = s["real"].values
        bb = (s["base"].values - r).mean() / r.mean() * 100
        br = (s["ref"].values - r).mean() / r.mean() * 100
        print(f"    D{dec} ${r.min():>9,.0f}-${r.max():>9,.0f} {bb:+6.1f}% -> {br:+6.1f}%",
              flush=True)
    g1 = d[d["gate"] == 1]
    if len(g1):
        a = g1["real"].values
        log(f"  [gate=1 n={len(g1):,}] wMAPE tabular {wmape(a, g1['base'].values):.2f}% "
            f"-> refined {wmape(a, g1['ref'].values):.2f}%")


def main():
    OOF_DIR.mkdir(exist_ok=True)
    tab_params = dict(TABULAR_PARAMS, objective="quantile", alpha=TAU)
    cols = F.feature_columns()
    log(f"FINAL | tabular cuantil tau={TAU} + residual imagen (cap={CAP}, alpha={ALPHA}) | "
        f"{len(SEEDS)} seeds x {N_FOLDS} folds")

    df = F.build_features(pd.read_csv(TRAIN_PATH))
    n = len(df)
    actual = df[F.PRICE_COL].values
    real_log = df[F.TARGET].values

    # Imagen interpretable (sin PCA), imputada por mediana train.
    img_train = IR.scalar_features("train")
    medians = IR.fit_impute(img_train)
    img = IR.apply_impute(img_train.reindex(df["zpid"].values), medians)
    img.index = df.index
    res_cols = list(img.columns)
    gate = IR.gate_mask(img).values
    log(f"Propiedades: {n:,} | gate activo (con foto, >=3, no satelite): {int(gate.sum()):,}")

    # ── Doble OOF: tabular (cuantil) + residual imagen (target = real - tab_oof) ──
    tab_oof_sum = np.zeros(n)
    res_oof_sum = np.zeros(n)
    for seed in SEEDS:
        t0 = time.time()
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        tab_oof = np.zeros(n)
        for tr_idx, va_idx in kf.split(df):
            tr = df.iloc[tr_idx].copy()
            va = df.iloc[va_idx].copy()
            tr, va = F.apply_fold_encodings(tr, va)
            tr = F.cast_categoricals(tr)
            va = F.cast_categoricals(va)
            m = lgb.LGBMRegressor(**tab_params, random_state=seed, n_jobs=-1, verbosity=-1)
            m.fit(tr[cols], tr[F.TARGET], categorical_feature=F.CATEGORICAL_FEATURES)
            tab_oof[va_idx] = m.predict(va[cols])
        tab_oof_sum += tab_oof

        res_target = real_log - tab_oof
        for tr_idx, va_idx in kf.split(df):
            rm = lgb.LGBMRegressor(**RESIDUAL_PARAMS, random_state=seed,
                                   n_jobs=-1, verbosity=-1)
            rm.fit(img.iloc[tr_idx][res_cols], res_target[tr_idx])
            res_oof_sum[va_idx] += rm.predict(img.iloc[va_idx][res_cols])
        log(f"  seed {seed}: tabular MAE ${mean_absolute_error(actual, np.expm1(tab_oof)):,.0f} "
            f"({time.time() - t0:.0f}s)")

    tab_oof_log = tab_oof_sum / len(SEEDS)
    res_oof = res_oof_sum / len(SEEDS)

    base_price = np.expm1(tab_oof_log)
    delta = gate * np.clip(ALPHA * res_oof, -CAP, CAP)
    refined = np.expm1(tab_oof_log + delta)

    base = roi(base_price, actual)
    ref = roi(refined, actual)
    log("=" * 78)
    log(f"tabular  -> MAE ${mean_absolute_error(actual, base_price):,.0f} | "
        f"wMAPE {wmape(actual, base_price):.2f}% | R2(log) {r2_score(real_log, tab_oof_log):.4f} | "
        f"ROI {base['roi_mean']:+.2f}% (win {base['win_pos']:.0f}%, %malas {base['trap_rate']:.0f})")
    log(f"refined  -> MAE ${mean_absolute_error(actual, refined):,.0f} | "
        f"wMAPE {wmape(actual, refined):.2f}% | "
        f"ROI {ref['roi_mean']:+.2f}% (win {ref['win_pos']:.0f}%, %malas {ref['trap_rate']:.0f})")
    log("=" * 78)
    report_slices(actual, base_price, refined, gate)

    # ── Salidas ──────────────────────────────────────────────────────────────
    pd.DataFrame({"zpid": df["zpid"].values, "predicted_price": refined}).to_csv(
        OOF_DIR / "practice_submission.csv", index=False)
    pd.DataFrame({
        "zpid": df["zpid"].values, "valor_real": actual,
        "predicted_tabular": base_price, "predicted_price": refined,
        "delta_pct": (refined - base_price) / base_price * 100, "gate": gate,
        "single_photo": img["single_photo"].values,
        "satellite_only": img["satellite_only"].values,
        "error_pct": np.abs(refined - actual) / actual * 100,
    }).to_csv(OOF_DIR / "comparison_oof.csv", index=False)
    pd.DataFrame({"zpid": df["zpid"].values, "tab_oof_log": tab_oof_log}).to_csv(
        OOF_DIR / "oof_tabular.csv", index=False)
    with open(OOF_DIR / "refined_config.json", "w") as f:
        json.dump(dict(
            tau=TAU, cap=CAP, alpha=ALPHA,
            roi_tabular=base["roi_mean"], roi_refined=ref["roi_mean"],
            win_tabular=base["win_pos"], win_refined=ref["win_pos"],
            wmape_tabular=wmape(actual, base_price), wmape_refined=wmape(actual, refined),
        ), f, indent=2)
    log("Guardado practice_submission.csv + comparison_oof.csv + oof_tabular.csv + "
        "refined_config.json")
    print(pd.Series(refined).describe())


if __name__ == "__main__":
    main()
