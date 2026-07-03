"""
Round 2 — Correccion cross-fitted para listados escasos (patron de error #1).

Sobreestimacion masiva cuando: 1 sola foto, satelite-only, o satelite unico.
Encoge pred_log hacia la media OOF solo en esos casos:
    pred' = center + alpha * (pred - center),  alpha in (0, 1]
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

ALPHA_GRID = [0.55, 0.65, 0.75, 0.85, 0.92, 1.0]


def sparse_risk(flags) -> np.ndarray:
    """Mascara booleana: propiedades de alto riesgo de sobreestimacion."""
    return (
        (flags["single_photo"] == 1)
        | (flags["single_satellite"] == 1)
        | (flags["satellite_only"] == 1)
    ).values.astype(bool)


def apply_sparse_shrink(pred_log: np.ndarray, risk: np.ndarray,
                        alpha: float, center: float) -> np.ndarray:
    if alpha >= 1.0:
        return pred_log
    out = pred_log.copy()
    out[risk] = center + alpha * (pred_log[risk] - center)
    return out


def wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(pred - actual).sum() / actual.sum() * 100)


def crossfit_sparse_shrink(pred_log: np.ndarray, actual_price: np.ndarray,
                           risk: np.ndarray, n_splits: int = 5,
                           seed: int = 0) -> tuple[np.ndarray, float]:
    center = float(pred_log.mean())
    best_alpha, best_score = 1.0, float("inf")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for alpha in ALPHA_GRID:
        out = np.empty_like(pred_log)
        for tr, va in kf.split(pred_log):
            out[va] = apply_sparse_shrink(pred_log[va], risk[va], alpha, center)
        score = wmape(actual_price, np.expm1(out))
        if score < best_score:
            best_score, best_alpha = score, alpha

    final = apply_sparse_shrink(pred_log, risk, best_alpha, center)
    return final, best_alpha


def save_params(path: Path, alpha: float, center: float, n_risk: int) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "method": "sparse_shrink",
            "alpha": alpha,
            "center": center,
            "n_risk_train": n_risk,
        }, f, indent=2)


def load_params(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def apply_saved(pred_log: np.ndarray, risk: np.ndarray, params: dict | None) -> np.ndarray:
    if params is None:
        return pred_log
    return apply_sparse_shrink(pred_log, risk, params["alpha"], params["center"])
