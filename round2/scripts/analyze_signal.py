"""
Round 2 — Reporte: cuanto aporta cada senal de imagen al precio.

Imprime:
  - correlacion de fracciones de tipo / materiales / sufijo con log_price
  - precio medio por tipo de foto dominante
  - relacion sufijo del filename ("otro id") vs tipo de foto (a nivel imagen)

Entorno: env con pandas (p.ej. labo2):
  cd round2
  /home/matias/miniconda3/envs/labo2/bin/python scripts/analyze_signal.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import features_img as FI

TRAIN_TAB = Path("../participant/data/tabular/train_processed.csv")
TARGET = "log_price"
PRICE = "lastSoldPrice_hpi_adjusted"


def main():
    *_, scal, flags = FI.pooled_matrices("train")
    tab = pd.read_csv(TRAIN_TAB, usecols=["zpid", TARGET, PRICE])
    df = scal.join(flags, how="outer").join(tab.set_index("zpid"), how="inner").dropna(subset=[TARGET])
    print(f"Propiedades con foto + target: {len(df):,}\n")

    # 1) Correlaciones con log_price
    feat_cols = [c for c in scal.columns if c != "n_images"]
    corr = df[feat_cols].corrwith(df[TARGET]).sort_values(key=np.abs, ascending=False)
    print("=== Correlacion de senales de imagen con log_price (top 20) ===")
    for name, v in corr.head(20).items():
        print(f"  {name:24s} {v:+.3f}")

    # 2) Precio medio por tipo dominante
    frac_cols = [f"frac_{k}" for k in FI.TYPE_KEYS]
    df["tipo_dominante"] = df[frac_cols].idxmax(axis=1).str.replace("frac_", "")
    print("\n=== Precio medio por tipo de foto dominante ===")
    g = df.groupby("tipo_dominante")[PRICE].agg(["count", "mean", "median"])
    for t, r in g.iterrows():
        print(f"  {t:10s} n={int(r['count']):>6,} | "
              f"media ${r['mean']:>10,.0f} | mediana ${r['median']:>10,.0f}")

    # 3) Sufijo ("otro id") vs tipo de foto, a nivel imagen
    meta = pd.read_csv(FI._emb_path("train")[1])
    meta = meta[meta["ok"] == 1]
    print("\n=== Sufijo del filename por tipo de foto (nivel imagen) ===")
    s = meta.groupby("type_argmax")["suffix"].agg(["count", "mean", "median"])
    for t, r in s.iterrows():
        print(f"  {t:10s} n={int(r['count']):>7,} | "
              f"suffix medio {r['mean']:>7.1f} | mediana {r['median']:>6.0f}")

    print("\n=== Tipo de foto por posicion (image_index 0..5) ===")
    sub = meta[meta["image_index"] <= 5]
    ct = pd.crosstab(sub["image_index"], sub["type_argmax"], normalize="index")
    print((ct * 100).round(1).to_string())


if __name__ == "__main__":
    main()
