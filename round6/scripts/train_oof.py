"""Round 6 — OOF autocontenido: base precisa + edge selectivo.

La base conserva el wMAPE fuerte: tabular quantile tau=0.35 + residual de imagen.
El edge aprende OOF el riesgo de sobrevaluacion peligrosa y solo recorta casos de
alto riesgo. Se maximiza ROI sujeto a:

    wMAPE_edge <= wMAPE_base + WMAPE_BUDGET

No se lee ni escribe round3.
"""

from __future__ import annotations

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
import features_edge as E
import features_img_residual as IR
from simulate_roi import simulate

TRAIN_PATH = "data/tabular/train_processed.csv"
OOF_DIR = Path("submissions_train")

SEEDS = [42, 99, 123]
N_FOLDS = 5
ROI_SIMS = 300
TAU = 0.35
CAP = 0.08
ALPHA = 1.0

# Preservar la precision de round5/base; el edge no puede gastar mas que esto.
WMAPE_BUDGET = 0.25
EDGE_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
EDGE_ALPHAS = [0.25, 0.50, 0.75, 1.00]

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


def wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(pred - actual).sum() / np.abs(actual).sum() * 100)


def roi(price: np.ndarray, actual: np.ndarray) -> dict:
    return simulate(price, actual, 1.0, np.random.default_rng(12345), n_sims=ROI_SIMS)


def quintile_report(actual: np.ndarray, price: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"real": actual, "pred": price})
    d["q"] = pd.qcut(actual, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    rows = []
    for q, s in d.groupby("q", observed=True):
        real = s["real"].values
        pred = s["pred"].values
        rows.append({
            "quintil": q,
            "n": len(s),
            "mape": float(np.mean(np.abs(pred - real) / real) * 100),
            "bias_pct": float(np.mean(pred - real) / np.mean(real) * 100),
            "over_rate": float(np.mean(pred > real) * 100),
        })
    return pd.DataFrame(rows).set_index("quintil")


def metrics(name: str, actual: np.ndarray, real_log: np.ndarray,
            price: np.ndarray) -> dict:
    pred_log = np.log1p(np.maximum(price, 0.0))
    sim = roi(price, actual)
    q = quintile_report(actual, price)
    return {
        "name": name,
        "mae": float(mean_absolute_error(actual, price)),
        "wmape": wmape(actual, price),
        "r2_log": float(r2_score(real_log, pred_log)),
        "roi_mean": float(sim["roi_mean"]),
        "win_pos": float(sim["win_pos"]),
        "trap_rate": float(sim["trap_rate"]),
        "q1_mape": float(q.loc["Q1", "mape"]),
        "q1_over_rate": float(q.loc["Q1", "over_rate"]),
        "q1_bias_pct": float(q.loc["Q1", "bias_pct"]),
    }


def show(m: dict) -> None:
    log(f"{m['name']:<23} MAE ${m['mae']:>9,.0f} | wMAPE {m['wmape']:5.2f}% | "
        f"ROI {m['roi_mean']:+.2f}% (win {m['win_pos']:.0f}%, traps {m['trap_rate']:.1f}%) | "
        f"Q1 MAPE {m['q1_mape']:5.1f}%")


def compute_tabular_oof(
    df: pd.DataFrame, actual: np.ndarray, cols: list[str],
) -> np.ndarray:
    params = dict(TABULAR_PARAMS, objective="quantile", alpha=TAU)
    oof_sum = np.zeros(len(df))
    for seed in SEEDS:
        start = time.time()
        oof = np.zeros(len(df))
        kf = KFold(N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in kf.split(df):
            tr, va = F.apply_fold_encodings(
                df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
            )
            tr, va = F.cast_categoricals(tr), F.cast_categoricals(va)
            model = lgb.LGBMRegressor(
                **params, random_state=seed, n_jobs=-1, verbosity=-1
            )
            model.fit(
                tr[cols], tr[F.TARGET],
                categorical_feature=F.CATEGORICAL_FEATURES,
            )
            oof[va_idx] = model.predict(va[cols])
        oof_sum += oof
        log(f"  base seed {seed}: MAE ${mean_absolute_error(actual, np.expm1(oof)):,.0f} "
            f"({time.time() - start:.0f}s)")
    return oof_sum / len(SEEDS)


def compute_residual_oof(
    df: pd.DataFrame,
    real_log: np.ndarray,
    tab_oof_log: np.ndarray,
    img: pd.DataFrame,
) -> np.ndarray:
    res_sum = np.zeros(len(df))
    res_target = real_log - tab_oof_log
    for seed in SEEDS:
        kf = KFold(N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in kf.split(df):
            model = lgb.LGBMRegressor(
                **RESIDUAL_PARAMS, random_state=seed, n_jobs=-1, verbosity=-1
            )
            model.fit(img.iloc[tr_idx], res_target[tr_idx])
            res_sum[va_idx] += model.predict(img.iloc[va_idx])
    return res_sum / len(SEEDS)


def main() -> None:
    OOF_DIR.mkdir(exist_ok=True)
    cols = F.feature_columns()
    log(f"Round6 EDGE | base tau={TAU} + residual imagen cap={CAP} | "
        f"{len(SEEDS)} seeds x {N_FOLDS} folds | {len(cols)} features")

    df = F.build_features(pd.read_csv(TRAIN_PATH))
    zpids = df["zpid"].values
    actual = df[F.PRICE_COL].values
    real_log = df[F.TARGET].values

    img_raw = IR.scalar_features("train")
    medians = IR.fit_impute(img_raw)
    img = IR.apply_impute(img_raw.reindex(zpids), medians)
    img.index = df.index
    gate = IR.gate_mask(img).values
    log(f"Propiedades: {len(df):,} | gate imagen activo: {int(gate.sum()):,}")

    # Base precisa autocontenida.
    tab_oof_log = compute_tabular_oof(df, actual, cols)
    res_oof = compute_residual_oof(df, real_log, tab_oof_log, img)
    base_log = tab_oof_log + gate * np.clip(ALPHA * res_oof, -CAP, CAP)
    base_price = np.expm1(base_log)
    baseline = metrics("base_precisa", actual, real_log, base_price)

    # Edge OOF.
    log("Cross-fitting edge risk + severidad ...")
    edge_risk, edge_severity, trap = E.crossfit(
        df, base_price, real_log, n_folds=N_FOLDS
    )
    log(f"Trap teorica (> {E.SAFE_PRED_TO_TRUE:.3f}x true): {trap.mean()*100:.1f}%")

    grid = []
    for threshold in EDGE_THRESHOLDS:
        for edge_alpha in EDGE_ALPHAS:
            price, cut_log = E.apply_edge(
                base_price, edge_risk, edge_severity, threshold, edge_alpha
            )
            row = metrics(
                f"edge_t{threshold:.2f}_a{edge_alpha:.2f}", actual, real_log, price
            )
            row.update({
                "edge_threshold": threshold,
                "edge_alpha": edge_alpha,
                "affected_pct": float(np.mean(cut_log > 0) * 100),
                "mean_cut_pct": float(np.mean(1.0 - price / base_price) * 100),
            })
            grid.append(row)

    wmape_limit = baseline["wmape"] + WMAPE_BUDGET
    eligible = [row for row in grid if row["wmape"] <= wmape_limit]
    winner = max(
        eligible,
        key=lambda row: (
            round(row["roi_mean"], 4), -row["wmape"], -row["q1_mape"]
        ),
    )
    winner_price, edge_cut_log = E.apply_edge(
        base_price, edge_risk, edge_severity,
        winner["edge_threshold"], winner["edge_alpha"],
    )

    log("=" * 104)
    show(baseline)
    for row in sorted(eligible, key=lambda x: -x["roi_mean"])[:8]:
        show(row)
    show({**winner, "name": "GANADOR EDGE"})
    log(f"wMAPE permitido <= {wmape_limit:.2f}% | afectadas "
        f"{winner['affected_pct']:.1f}% | recorte medio {winner['mean_cut_pct']:.2f}%")
    log("=" * 104)

    qb = quintile_report(actual, base_price)
    qw = quintile_report(actual, winner_price)
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        log(f"{q}: MAPE {qb.loc[q,'mape']:.1f}% -> {qw.loc[q,'mape']:.1f}% | "
            f"over {qb.loc[q,'over_rate']:.1f}% -> {qw.loc[q,'over_rate']:.1f}%")

    error_base = np.abs(base_price - actual) / actual * 100
    error_edge = np.abs(winner_price - actual) / actual * 100
    edge_cut_pct = (1.0 - np.exp(-edge_cut_log)) * 100
    quintiles = pd.qcut(actual, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    pd.DataFrame({"zpid": zpids, "predicted_price": winner_price}).to_csv(
        OOF_DIR / "practice_submission.csv", index=False
    )
    pd.DataFrame({
        "zpid": zpids,
        "valor_real": actual,
        "predicted_tabular": np.expm1(tab_oof_log),
        "predicted_baseline": base_price,
        "predicted_price": winner_price,
        "edge_risk": edge_risk,
        "edge_severity": edge_severity,
        "edge_cut_pct": edge_cut_pct,
        "quintil": quintiles,
        "error_pct": error_edge,
        "error_pct_baseline": error_base,
    }).to_csv(OOF_DIR / "comparison_oof.csv", index=False)
    pd.DataFrame({"zpid": zpids, "tab_oof_log": tab_oof_log}).to_csv(
        OOF_DIR / "oof_tabular.csv", index=False
    )
    pd.DataFrame({
        "zpid": zpids,
        "base_oof_log": base_log,
        "edge_risk": edge_risk,
        "edge_severity": edge_severity,
    }).to_csv(OOF_DIR / "oof_edge.csv", index=False)
    pd.DataFrame({
        "zpid": zpids,
        "valor_real": actual,
        "baseline_price": base_price,
        "round6_price": winner_price,
        "diff": winner_price - base_price,
        "diff_pct": (winner_price - base_price) / base_price * 100,
        "error_pct_baseline": error_base,
        "error_pct_round6": error_edge,
        "mejora": (error_edge < error_base).astype(int),
    }).to_csv(OOF_DIR / "diff_vs_baseline.csv", index=False)

    with open(OOF_DIR / "round6_config.json", "w") as file:
        json.dump({
            "self_contained": True,
            "strategy": "precise_base_plus_selective_edge",
            "seeds": SEEDS,
            "n_folds": N_FOLDS,
            "tau": TAU,
            "cap": CAP,
            "alpha": ALPHA,
            "wmape_budget": WMAPE_BUDGET,
            "safe_pred_to_true": E.SAFE_PRED_TO_TRUE,
            "edge_threshold": winner["edge_threshold"],
            "edge_alpha": winner["edge_alpha"],
            "tabular_params": TABULAR_PARAMS,
            "residual_params": RESIDUAL_PARAMS,
            "edge_classifier_params": E.CLASSIFIER_PARAMS,
            "edge_severity_params": E.SEVERITY_PARAMS,
            "metrics_baseline": baseline,
            "metrics_winner": winner,
            "edge_grid": grid,
        }, file, indent=2)

    log("Guardado practice + comparison + oof_tabular + oof_edge + config")
    log(f"Edge mejora error en {(error_edge < error_base).sum():,}/{len(df):,} props")


if __name__ == "__main__":
    main()
