"""
Simulacion local del juego de inversion (game_mechanics_es.md) sobre las predicciones
OOF de train (donde SI conocemos el valor real). Sirve para medir ROI — no wMAPE — y
asi decidir cuanto conviene ser conservador (escalar la prediccion antes de pujar).

Mecanica replicada:
  asking = true * (1 + N(market_bias, market_noise))
  compra si  pred_submitted > asking*(1+buy_threshold)  y  asking*(1+tx) <= cap*max_pos
  puja = pred_submitted * bid_fraction   (bid_fraction lo fija el motor, no nosotros)
  subasta Vickrey 2do precio vs N competidores sinteticos (pred = true*(1+N(0,comp_sigma)))
  profit = true - paid*(1+tx) ;  penaliza capital ocioso al cierre de ronda

OJO: los competidores son sinteticos (asuncion). El resultado es comparativo entre
escalas, no un ROI absoluto exacto.

Uso:
    python scripts/simulate_roi.py                # barre escalas sobre fusion y tabular
    python scripts/simulate_roi.py predicted_tabular_only
"""

import sys

import numpy as np
import pandas as pd

COMPARISON = "submissions_train/comparison_oof.csv"

# Defaults de la competencia (game_mechanics_es.md)
P = dict(
    market_bias=-0.07, market_noise=0.35,
    buy_threshold=0.08, transaction_cost=0.02, max_position=0.25,
    bid_fraction=0.85, opportunity_cost=0.04,
    n_rounds=4, props_per_round=250, start_capital=5_000_000.0,
)
N_SIMS = 400
N_COMP = 3          # competidores sinteticos
COMP_SIGMA = 0.18   # ~18% de ruido de valuacion de los rivales
SCALES = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]


def simulate(pred, true, scale, rng, n_sims=N_SIMS):
    sub = pred * scale
    n = len(true)
    tx = P["transaction_cost"]
    rois, buys, traps, profits = [], [], [], []
    for _ in range(n_sims):
        net = 0.0
        total_cap = P["start_capital"] * P["n_rounds"]
        nb = nt = 0
        gross = 0.0
        for _r in range(P["n_rounds"]):
            idx = rng.choice(n, P["props_per_round"], replace=False)
            t = true[idx]
            s = sub[idx]
            asking = t * (1 + rng.normal(P["market_bias"], P["market_noise"], len(idx)))
            asking = np.maximum(asking, 1000.0)
            # competidores: pred ~ true*(1+N(0,sigma)), puja = pred*bid_fraction si les sirve
            comp_pred = t[:, None] * (1 + rng.normal(0, COMP_SIGMA, (len(idx), N_COMP)))
            comp_want = comp_pred > asking[:, None] * (1 + P["buy_threshold"])
            comp_bid = np.where(comp_want, comp_pred * P["bid_fraction"], -np.inf)
            best_comp = comp_bid.max(axis=1)
            second_comp = np.sort(comp_bid, axis=1)[:, -2] if N_COMP >= 2 else np.full(len(idx), -np.inf)

            cap = P["start_capital"]
            for j in range(len(idx)):
                want = s[j] > asking[j] * (1 + P["buy_threshold"])
                if not want:
                    continue
                if asking[j] * (1 + tx) > cap * P["max_position"]:
                    continue
                our_bid = s[j] * P["bid_fraction"]
                bc = best_comp[j]
                if bc == -np.inf:                       # nadie mas la quiere
                    paid = asking[j]
                elif our_bid > bc:                       # ganamos la subasta
                    paid = max(asking[j], bc)
                else:                                     # perdemos
                    continue
                cost = paid * (1 + tx)
                if cost > cap:
                    continue
                cap -= cost
                profit = t[j] - cost
                net += profit
                gross += profit
                nb += 1
                if profit < 0:
                    nt += 1
            net -= cap * (P["opportunity_cost"] / P["n_rounds"])  # capital ocioso
        rois.append(net / total_cap * 100)
        buys.append(nb)
        traps.append(nt)
        profits.append(gross)
    rois = np.array(rois)
    buys = np.array(buys)
    traps = np.array(traps)
    return dict(
        roi_mean=rois.mean(), roi_med=np.median(rois), roi_std=rois.std(),
        win_pos=(rois > 0).mean() * 100,
        buys=buys.mean(), trap_rate=(traps.sum() / max(buys.sum(), 1)) * 100,
    )


def run(col):
    d = pd.read_csv(COMPARISON)
    pred = d[col].values.astype(float)
    true = d["valor_real"].values.astype(float)
    print(f"\n===== {col}  (N_SIMS={N_SIMS}, competidores={N_COMP}, sigma={COMP_SIGMA}) =====")
    print(f"{'scale':>6} {'ROI%':>8} {'medROI%':>8} {'std':>7} {'ROI>0%':>7} "
          f"{'buys/sim':>9} {'%malas':>7}")
    for k in SCALES:
        rng = np.random.default_rng(12345)
        m = simulate(pred, true, k, rng)
        print(f"{k:>6.2f} {m['roi_mean']:>8.2f} {m['roi_med']:>8.2f} {m['roi_std']:>7.2f} "
              f"{m['win_pos']:>7.1f} {m['buys']:>9.1f} {m['trap_rate']:>7.1f}")


def main():
    cols = sys.argv[1:] or ["predicted_price", "predicted_tabular"]
    print("Pasivo (no comprar): ROI ~ -4.0% (opportunity cost)")
    for c in cols:
        run(c)


if __name__ == "__main__":
    main()
