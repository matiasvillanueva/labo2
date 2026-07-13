"""
Round 5 — OOF (practice). COMBINA las tres fuentes de senal de los rounds previos:

  tabular (round3)  +  imagen (round3)  +  texto/LLM (round4)

Se prueban dos formas de combinar y se elige la mejor por ROI de la simulacion del
juego (game_mechanics_es.md), comparando SIEMPRE contra round3:

  A) feature_fusion : un solo LightGBM (cuantil tau=0.35) que ve TODAS las columnas
     (tabular + imagen + TF-IDF/SVD + embeddings LLM/SVD + flags LLM).
  B) text_residual  : se mantiene round3 (tabular+imagen) como base y el texto/LLM
     entra como residual acotado sobre log(real) - log(round3_oof). alpha=0 => round3,
     asi la combinacion nunca queda peor que round3.

Decision: gana la estrategia con mayor ROI OOF. Si la fusion no supera a round3, se usa
el residual (que por construccion es >= round3). Se guarda todo en round5_config.json.

Salidas (submissions_train/):
  - practice_submission.csv   OOF final (estrategia ganadora) para todos los zpid train
  - comparison_oof.csv          real, round3, fusion, residual, ganador, error
  - diff_vs_round3.csv          diferencia por zpid vs round3 (precio y error)
  - round5_config.json          estrategia ganadora + params + metricas de las 3 opciones

Run from round5/:
    /home/matias/miniconda3/envs/labo2/bin/python scripts/train_oof.py
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
import features_fusion as FF
import features_text_llm as FTL
from features_fusion import LLM_KWARGS
from simulate_roi import simulate

TRAIN_PATH = "data/tabular/train_processed.csv"
ROUND3_COMPARISON = "../round3/submissions_train/comparison_oof.csv"
OOF_DIR = Path("submissions_train")

SEEDS = [42, 99, 123]
N_FOLDS = 5
TAU = 0.35                     # mismo conservadurismo cuantil que round3
ROI_SIMS = 300                 # simulaciones ROI (fijo por reproducibilidad)

# Grilla del residual de texto: alpha=0 => queda round3 (piso de seguridad)
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
CAP_GRID = [0.03, 0.05, 0.08]

# Tabular: params ganadores round1/round3 (objetivo cuantil).
TABULAR_PARAMS = dict(
    n_estimators=1500, learning_rate=0.015, num_leaves=47,
    min_child_samples=35, colsample_bytree=0.40,
    reg_alpha=1.5, reg_lambda=3.5,
)
# Texto/LLM (residual y rama de texto): params round4.
TEXT_PARAMS = dict(
    n_estimators=900, learning_rate=0.025, num_leaves=31,
    min_child_samples=30, subsample=0.85, subsample_freq=1,
    colsample_bytree=0.70, reg_alpha=1.8, reg_lambda=3.5,
    objective="huber", alpha=0.9,
)


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(pred - actual).sum() / np.abs(actual).sum() * 100)


def roi(price: np.ndarray, actual: np.ndarray, n_sims: int = ROI_SIMS) -> dict:
    return simulate(price, actual, 1.0, np.random.default_rng(12345), n_sims=n_sims)


def metrics(name: str, actual: np.ndarray, real_log: np.ndarray,
            pred_price: np.ndarray, pred_log: np.ndarray | None = None) -> dict:
    if pred_log is None:
        pred_log = np.log1p(np.maximum(pred_price, 0.0))
    r = roi(pred_price, actual)
    return {
        "name": name,
        "mae": float(mean_absolute_error(actual, pred_price)),
        "wmape": wmape(actual, pred_price),
        "r2_log": float(r2_score(real_log, pred_log)),
        "roi_mean": float(r["roi_mean"]),
        "win_pos": float(r["win_pos"]),
        "trap_rate": float(r["trap_rate"]),
    }


def show(m: dict) -> None:
    log(f"{m['name']:<14} MAE ${m['mae']:>10,.0f} | wMAPE {m['wmape']:6.2f}% | "
        f"R2(log) {m['r2_log']:.4f} | ROI {m['roi_mean']:+.2f}% "
        f"(win {m['win_pos']:.0f}%, %malas {m['trap_rate']:.0f})")


# ── Estrategia A: fusion por columnas ─────────────────────────────────────────
def run_fusion(df: pd.DataFrame, real_log: np.ndarray, actual: np.ndarray,
               img_all: pd.DataFrame) -> tuple[np.ndarray, dict]:
    n = len(df)
    tab_q = dict(TABULAR_PARAMS, objective="quantile", alpha=TAU)
    oof_sum = np.zeros(n)
    for seed in SEEDS:
        t0 = time.time()
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(n)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(df), start=1):
            tr = df.iloc[tr_idx].copy()
            va = df.iloc[va_idx].copy()
            x_tr, x_va, _ = FF.assemble_fold(tr, va, img_all, seed=seed + fold)
            m = lgb.LGBMRegressor(**tab_q, random_state=seed, n_jobs=-1, verbosity=-1)
            m.fit(x_tr, real_log[tr_idx], categorical_feature=F.CATEGORICAL_FEATURES)
            oof[va_idx] = m.predict(x_va)
        oof_sum += oof
        log(f"  [fusion] seed {seed}: MAE ${mean_absolute_error(actual, np.expm1(oof)):,.0f} "
            f"| {x_tr.shape[1]} cols ({time.time() - t0:.0f}s)")
    oof_log = oof_sum / len(SEEDS)
    return oof_log, metrics("feature_fusion", actual, real_log,
                            np.expm1(oof_log), oof_log)


# ── Estrategia B: texto/LLM como residual sobre round3 ────────────────────────
def run_text_residual(df: pd.DataFrame, real_log: np.ndarray, actual: np.ndarray,
                      base_log: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    n = len(df)
    res_target = real_log - base_log
    res_sum = np.zeros(n)
    for seed in SEEDS:
        t0 = time.time()
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(n)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(df), start=1):
            tr = df.iloc[tr_idx].copy()
            va = df.iloc[va_idx].copy()
            x_tr, store = FTL.fit_features("train", tr, seed=seed + fold, **LLM_KWARGS)
            x_va = FTL.transform_features("train", va, store)
            m = lgb.LGBMRegressor(**TEXT_PARAMS, random_state=seed, n_jobs=-1, verbosity=-1)
            m.fit(x_tr, res_target[tr_idx])
            oof[va_idx] = m.predict(x_va)
        res_sum += oof
        log(f"  [residual] seed {seed}: |res| medio {np.abs(oof).mean():.4f} "
            f"({time.time() - t0:.0f}s)")
    res_oof = res_sum / len(SEEDS)

    best = None
    for alpha in ALPHA_GRID:
        for cap in CAP_GRID:
            delta = np.clip(alpha * res_oof, -cap, cap)
            price = np.expm1(base_log + delta)
            cand = metrics(f"residual a{alpha}c{cap}", actual, real_log, price,
                           base_log + delta)
            cand.update(alpha=alpha, cap=cap)
            if best is None or cand["roi_mean"] > best["roi_mean"]:
                best = cand
            if alpha == 0.0:
                break  # alpha=0 no depende de cap (equivale a round3)
    best_delta = np.clip(best["alpha"] * res_oof, -best["cap"], best["cap"])
    best_log = base_log + best_delta
    final = metrics("text_residual", actual, real_log, np.expm1(best_log), best_log)
    final.update(alpha=best["alpha"], cap=best["cap"])
    return best_log, final, {"res_oof_abs_mean": float(np.abs(res_oof).mean())}


def main() -> None:
    OOF_DIR.mkdir(exist_ok=True)
    log(f"Round5 COMBINA tabular+imagen (round3) + texto/LLM (round4) | "
        f"{len(SEEDS)} seeds x {N_FOLDS} folds")

    df = F.build_features(pd.read_csv(TRAIN_PATH))
    zpids = df["zpid"].values
    real_log = df[F.TARGET].values
    actual = df[F.PRICE_COL].values
    img_all = FF.image_frame("train")
    log(f"Propiedades: {len(df):,} | imagen scalars: {img_all.shape[1]} cols")

    # ── Baseline round3 (tabular + imagen) desde su OOF ya calculado ──
    r3 = pd.read_csv(ROUND3_COMPARISON).set_index("zpid")
    r3_price = r3["predicted_price"].reindex(zpids).values
    if np.isnan(r3_price).any():
        raise ValueError("Faltan zpids de round3 comparison_oof.csv para el baseline")
    base_log = np.log1p(r3_price)
    m_r3 = metrics("round3", actual, real_log, r3_price, base_log)

    # ── A) fusion por columnas ──
    log("Estrategia A: fusion por columnas (tabular+imagen+texto/LLM)")
    fusion_log, m_fusion = run_fusion(df, real_log, actual, img_all)

    # ── B) texto/LLM como residual sobre round3 ──
    log("Estrategia B: texto/LLM como residual acotado sobre round3")
    residual_log, m_residual, res_info = run_text_residual(df, real_log, actual, base_log)

    log("=" * 88)
    for m in (m_r3, m_fusion, m_residual):
        show(m)
    log("=" * 88)

    # ── Decision: gana el mayor ROI OOF; si la fusion no supera round3, usar residual ──
    fusion_beats_round3 = m_fusion["roi_mean"] > m_r3["roi_mean"]
    if fusion_beats_round3 and m_fusion["roi_mean"] >= m_residual["roi_mean"]:
        strategy, winner_log, m_win = "feature_fusion", fusion_log, m_fusion
    else:
        strategy, winner_log, m_win = "text_residual", residual_log, m_residual
    winner_price = np.expm1(winner_log)
    log(f"GANADOR: {strategy} | ROI {m_win['roi_mean']:+.2f}% vs round3 "
        f"{m_r3['roi_mean']:+.2f}% | wMAPE {m_win['wmape']:.2f}% vs {m_r3['wmape']:.2f}%")

    # ── Salidas ──
    pd.DataFrame({"zpid": zpids, "predicted_price": winner_price}).to_csv(
        OOF_DIR / "practice_submission.csv", index=False)

    pd.DataFrame({
        "zpid": zpids,
        "valor_real": actual,
        "predicted_round3": r3_price,
        "predicted_fusion": np.expm1(fusion_log),
        "predicted_residual": np.expm1(residual_log),
        "predicted_price": winner_price,
        "error_pct": np.abs(winner_price - actual) / actual * 100,
        "error_pct_round3": np.abs(r3_price - actual) / actual * 100,
    }).to_csv(OOF_DIR / "comparison_oof.csv", index=False)

    diff = winner_price - r3_price
    err_r5 = np.abs(winner_price - actual) / actual * 100
    err_r3 = np.abs(r3_price - actual) / actual * 100
    pd.DataFrame({
        "zpid": zpids,
        "valor_real": actual,
        "round3_price": r3_price,
        "round5_price": winner_price,
        "diff": diff,
        "diff_pct": diff / r3_price * 100,
        "error_pct_round3": err_r3,
        "error_pct_round5": err_r5,
        "mejora": (err_r5 < err_r3).astype(int),
    }).to_csv(OOF_DIR / "diff_vs_round3.csv", index=False)

    with open(OOF_DIR / "round5_config.json", "w") as f:
        json.dump({
            "strategy": strategy,
            "seeds": SEEDS,
            "n_folds": N_FOLDS,
            "tau": TAU,
            "residual_alpha": m_residual.get("alpha"),
            "residual_cap": m_residual.get("cap"),
            "tabular_params": TABULAR_PARAMS,
            "text_params": TEXT_PARAMS,
            "llm_kwargs": LLM_KWARGS,
            "metrics": {"round3": m_r3, "feature_fusion": m_fusion,
                        "text_residual": m_residual},
            "res_oof_abs_mean": res_info["res_oof_abs_mean"],
        }, f, indent=2)

    n_better = int((err_r5 < err_r3).sum())
    log(f"Guardado practice_submission.csv + comparison_oof.csv + diff_vs_round3.csv + "
        f"round5_config.json")
    log(f"vs round3: {n_better:,}/{len(df):,} propiedades con menor error | "
        f"diff medio ${np.mean(diff):,.0f} | |diff| medio ${np.mean(np.abs(diff)):,.0f}")
    print(pd.Series(winner_price).describe())


if __name__ == "__main__":
    main()
