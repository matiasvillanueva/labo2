"""
Round 3 — Submission de test. Modelo FINAL unico (mismo que train_oof.py):

  tabular conservador (cuantil tau=0.35)  +  ajuste de imagen acotado (residual con cap+gate)

Usa submissions_train/oof_tabular.csv y refined_config.json que genera train_oof.py
(el residual full se entrena sobre real - tabular_oof, sin leakage). Corre train_oof.py
primero.

Salida: submissions/submission.csv

Run from round3/:
    /home/matias/miniconda3/envs/labo2/bin/python scripts/predict_test.py
"""

import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import features as F
import features_img_residual as IR

# Debe coincidir con train_oof.py ─────────────────────────────────────────────
TRAIN_PATH = "data/tabular/train_processed.csv"
TEST_PATH = "data/tabular/test_processed.csv"
TEMPLATE_PATH = "../participant/submissions/template.csv"
OOF_DIR = Path("submissions_train")
OUT_DIR = Path("submissions")

SEEDS = [42, 99, 123, 456, 789]
TABULAR_PARAMS = dict(
    n_estimators=1500, learning_rate=0.015, num_leaves=47,
    min_child_samples=35, colsample_bytree=0.40,
    reg_alpha=1.5, reg_lambda=3.5,
)
RESIDUAL_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=15,
    min_child_samples=50, colsample_bytree=0.8,
    reg_alpha=2.0, reg_lambda=4.0,
)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main():
    cfg_path = OOF_DIR / "refined_config.json"
    oof_path = OOF_DIR / "oof_tabular.csv"
    if not cfg_path.exists() or not oof_path.exists():
        raise FileNotFoundError(
            f"Faltan {cfg_path} / {oof_path}. Corre primero scripts/train_oof.py")
    with open(cfg_path) as f:
        cfg = json.load(f)
    tau, cap, alpha = cfg["tau"], cfg["cap"], cfg["alpha"]
    tab_params = dict(TABULAR_PARAMS, objective="quantile", alpha=tau)
    cols = F.feature_columns()
    log(f"FINAL | tabular cuantil tau={tau} + residual imagen (cap={cap}, alpha={alpha}) | "
        f"{len(SEEDS)} seeds")

    df = F.build_features(pd.read_csv(TRAIN_PATH))
    test = F.build_features(pd.read_csv(TEST_PATH))
    template = pd.read_csv(TEMPLATE_PATH)
    train, test = F.apply_full_encodings(df.copy(), test)
    train = F.cast_categoricals(train)
    test = F.cast_categoricals(test)
    log(f"Train: {len(train):,} | Test: {len(test):,} | Template: {len(template):,}")

    # ── Tabular fit full ──────────────────────────────────────────────────────
    test_log = np.zeros(len(test))
    train_log = np.zeros(len(train))
    for seed in SEEDS:
        m = lgb.LGBMRegressor(**tab_params, random_state=seed, n_jobs=-1, verbosity=-1)
        m.fit(train[cols], train[F.TARGET], categorical_feature=F.CATEGORICAL_FEATURES)
        test_log += m.predict(test[cols])
        train_log += m.predict(train[cols])
    test_log /= len(SEEDS)
    train_log /= len(SEEDS)

    # ── Residual full: target = real - tabular_oof (de train_oof, sin leakage) ──
    oof_tab = pd.read_csv(oof_path).set_index("zpid")["tab_oof_log"]
    tab_oof_log = oof_tab.reindex(df["zpid"].values).values
    res_target = df[F.TARGET].values - tab_oof_log

    medians = IR.fit_impute(IR.scalar_features("train"))
    res_cols = list(medians.index)
    img_tr = IR.apply_impute(
        IR.scalar_features("train").reindex(df["zpid"].values), medians)
    img_te = IR.apply_impute(
        IR.scalar_features("test").reindex(test["zpid"].values), medians)
    gate_te = IR.gate_mask(img_te).values

    res_test_sum = np.zeros(len(test))
    for seed in SEEDS:
        rm = lgb.LGBMRegressor(**RESIDUAL_PARAMS, random_state=seed,
                               n_jobs=-1, verbosity=-1)
        rm.fit(img_tr[res_cols], res_target)
        res_test_sum += rm.predict(img_te[res_cols])
    res_test = res_test_sum / len(SEEDS)

    delta_te = gate_te * np.clip(alpha * res_test, -cap, cap)
    test_pred = np.expm1(test_log + delta_te)

    # Diagnostico in-sample (referencia, no metrica honesta).
    train_pred = np.expm1(train_log)
    train_real = train[F.PRICE_COL].values
    log(f"Diagnostico train (in-sample): R2 {r2_score(train_real, train_pred):.4f} | "
        f"MAE ${mean_absolute_error(train_real, train_pred):,.0f}")
    log(f"Test: ajuste imagen medio {np.mean(np.abs(np.expm1(test_log + delta_te) - np.expm1(test_log)) / np.expm1(test_log)) * 100:.2f}% "
        f"| gate activo {int(gate_te.sum()):,}/{len(test):,}")

    pred = pd.DataFrame({"zpid": test["zpid"], "predicted_price": test_pred})
    sub = template[["zpid"]].merge(pred, on="zpid", how="left")
    if sub["predicted_price"].isna().any():
        raise ValueError("Faltan predicciones para zpids del template")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "submission.csv"
    sub.to_csv(out, index=False)
    log(f"Guardado {out} ({len(sub):,} filas)")
    print(sub["predicted_price"].describe())


if __name__ == "__main__":
    main()
