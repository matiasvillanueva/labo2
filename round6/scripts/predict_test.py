"""Round 6 — submission: base precisa + edge selectivo.

Reproduce la base tau=0.35 + residual imagen y entrena el edge final usando los
errores OOF. El edge solo recorta predicciones con alto riesgo de sobrevaluacion.

Salida: submissions/submission.csv

Run from round6/:
    /home/matias/miniconda3/envs/labo2/bin/python scripts/predict_test.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import features as F
import features_edge as E
import features_img_residual as IR

TRAIN_PATH = "data/tabular/train_processed.csv"
TEST_PATH = "data/tabular/test_processed.csv"
TEMPLATE_PATH = "../participant/submissions/template.csv"
OOF_DIR = Path("submissions_train")
OUT_DIR = Path("submissions")

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


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main() -> None:
    cfg_path = OOF_DIR / "round6_config.json"
    oof_path = OOF_DIR / "oof_tabular.csv"
    edge_oof_path = OOF_DIR / "oof_edge.csv"
    if not cfg_path.exists() or not oof_path.exists() or not edge_oof_path.exists():
        raise FileNotFoundError(
            "Faltan artefactos OOF. Corre primero scripts/train_oof.py")
    with open(cfg_path) as f:
        cfg = json.load(f)

    seeds = cfg["seeds"]
    cap, alpha = cfg["cap"], cfg["alpha"]
    tau = cfg["tau"]
    edge_threshold = cfg["edge_threshold"]
    edge_alpha = cfg["edge_alpha"]
    tab_params = dict(TABULAR_PARAMS, objective="quantile", alpha=tau)
    cols = F.feature_columns()
    log(f"FINAL round6 | base tau={tau} + edge t={edge_threshold} a={edge_alpha} | "
        f"imagen cap={cap} | {len(seeds)} seeds | {len(cols)} feats")

    df = F.build_features(pd.read_csv(TRAIN_PATH))
    test_raw = F.build_features(pd.read_csv(TEST_PATH))
    template = pd.read_csv(TEMPLATE_PATH)
    train, test = F.apply_full_encodings(df.copy(), test_raw.copy())
    train = F.cast_categoricals(train)
    test = F.cast_categoricals(test)
    log(f"Train: {len(train):,} | Test: {len(test):,} | Template: {len(template):,}")

    # ── Tabular preciso fit full ──────────────────────────────────────────────
    real_log = train[F.TARGET].values
    test_log = np.zeros(len(test))
    train_log = np.zeros(len(train))
    for seed in seeds:
        m = lgb.LGBMRegressor(**tab_params, random_state=seed, n_jobs=-1, verbosity=-1)
        m.fit(train[cols], train[F.TARGET],
              categorical_feature=F.CATEGORICAL_FEATURES)
        test_log += m.predict(test[cols])
        train_log += m.predict(train[cols])
    test_log /= len(seeds)
    train_log /= len(seeds)

    # ── Residual imagen: target = real - tab_oof (de train_oof, sin leakage) ──
    oof_tab = pd.read_csv(oof_path).set_index("zpid")["tab_oof_log"]
    tab_oof_log = oof_tab.reindex(df["zpid"].values).values
    res_target = df[F.TARGET].values - tab_oof_log

    medians = IR.fit_impute(IR.scalar_features("train"))
    res_cols = list(medians.index)
    img_tr = IR.apply_impute(IR.scalar_features("train").reindex(df["zpid"].values), medians)
    img_te = IR.apply_impute(IR.scalar_features("test").reindex(test["zpid"].values), medians)
    gate_te = IR.gate_mask(img_te).values

    res_test_sum = np.zeros(len(test))
    for seed in seeds:
        rm = lgb.LGBMRegressor(**RESIDUAL_PARAMS, random_state=seed,
                               n_jobs=-1, verbosity=-1)
        rm.fit(img_tr[res_cols], res_target)
        res_test_sum += rm.predict(img_te[res_cols])
    res_test = res_test_sum / len(seeds)

    delta_te = gate_te * np.clip(alpha * res_test, -cap, cap)
    base_test_log = test_log + delta_te
    base_test_price = np.expm1(base_test_log)

    # ── Edge final: se entrena en errores de la base OOF, nunca in-sample ─────
    edge_oof = pd.read_csv(edge_oof_path).set_index("zpid")
    base_oof_log = edge_oof["base_oof_log"].reindex(df["zpid"].values).values
    base_oof_price = np.expm1(base_oof_log)
    trap, severity_target = E.targets(base_oof_price, df[F.TARGET].values)
    edge_train = E.build_features(df, base_oof_price)
    edge_test = E.build_features(test_raw, base_test_price)
    classifier, severity_model = E.fit_models(
        edge_train, trap, severity_target, seed=2026
    )
    edge_risk, edge_severity = E.predict_models(
        classifier, severity_model, edge_test
    )
    test_pred, edge_cut_log = E.apply_edge(
        base_test_price, edge_risk, edge_severity, edge_threshold, edge_alpha
    )

    train_pred = np.expm1(train_log)
    train_real = train[F.PRICE_COL].values
    log(f"Diagnostico train (in-sample): R2 {r2_score(train_real, train_pred):.4f} | "
        f"MAE ${mean_absolute_error(train_real, train_pred):,.0f}")
    log(f"Test: gate imagen {int(gate_te.sum()):,}/{len(test):,} | edge afecta "
        f"{int((edge_cut_log > 0).sum()):,}/{len(test):,} | recorte medio "
        f"{np.mean(1.0 - test_pred/base_test_price)*100:.2f}%")

    pred = pd.DataFrame({"zpid": test["zpid"], "predicted_price": test_pred})
    sub = template[["zpid"]].merge(pred, on="zpid", how="left")
    if sub["predicted_price"].isna().any():
        raise ValueError("Faltan predicciones para zpids del template")

    OUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({
        "zpid": test["zpid"].values,
        "base_price": base_test_price,
        "edge_risk": edge_risk,
        "edge_severity": edge_severity,
        "edge_cut_pct": (1.0 - np.exp(-edge_cut_log)) * 100,
        "predicted_price": test_pred,
    }).to_csv(OUT_DIR / "test_edge_diagnostics.csv", index=False)
    out = OUT_DIR / "submission.csv"
    sub.to_csv(out, index=False)
    log(f"Guardado {out} ({len(sub):,} filas)")
    print(sub["predicted_price"].describe())


if __name__ == "__main__":
    main()
