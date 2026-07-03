# Round 3 — Tabular (round1) + Imágenes (round2)

Modelo FINAL único (sin variantes): **tabular conservador + ajuste de imagen acotado**.

1. **Tabular** (`features.py`, round1) con objetivo cuantil `tau=0.35`: predice un
   percentil bajo del precio. Apostar un poco más bajo maximiza el ROI de la simulación
   del juego (`game_mechanics_es.md`), que es lo que decide la competencia.
2. **Ajuste de imagen** (`features_img_residual.py`): un modelo chico corrige el
   **residual** que el tabular no ve (lujo, renovado, piscina), con límite duro de ±8%
   (`cap`) y un `gate` que lo apaga en listados escasos (1 foto, <3 fotos, satélite-only),
   donde la imagen empeoraba.

La fusión "early" (concatenar 351 columnas de imagen en un solo LGBM) se probó y
**empeoró**; por eso la imagen entra solo como corrección acotada del residual.

## Estructura

```
round3/
  data/tabular/        -> symlinks a round1/data/tabular/
  embeddings/          embeddings DINOv2 + meta + shards (movidos desde round2)
  scripts/
    extract_embeddings.py    extracción (DINOv2+SigLIP), solo si faltan fotos
    features.py                tabular (copia de round1)
    features_img.py            agregación imagen por zpid (provee pooled_matrices)
    features_img_residual.py   escalares de imagen interpretables (sin PCA) + gate
    simulate_roi.py            simulación del juego para medir ROI (no wMAPE)
    train_oof.py                 OOF tabular + residual -> practice + comparison + config
    predict_test.py            fit full + residual -> submission de test
  submissions_train/   practice_submission.csv, comparison_oof.csv,
                       oof_tabular.csv, refined_config.json
  submissions/         submission.csv
```

## Comandos

```bash
cd round3

# (opcional) re-extraer embeddings — solo si faltan fotos. Ya están en embeddings/.
MAX_IMAGES_PER_PROP=9999 /home/matias/miniconda3/envs/llms/bin/python scripts/extract_embeddings.py

# 1) OOF: practice_submission.csv + comparison_oof.csv + oof_tabular.csv + refined_config.json
/home/matias/miniconda3/envs/labo2/bin/python scripts/train_oof.py

# 2) Submission de test (usa oof_tabular.csv + refined_config.json del paso 1)
/home/matias/miniconda3/envs/labo2/bin/python scripts/predict_test.py
```

## Qué mirar

`train_oof.py` imprime, en OOF y medido por ROI sim:

- `tabular` (solo cuantil tau=0.35) como baseline
- `refined` (tabular + ajuste de imagen) — debe ganar en ROI, win-rate y wMAPE

Más el sesgo por decil (tabular → refined) y el wMAPE en el slice `gate=1` (donde la
imagen sí puede opinar). `comparison_oof.csv` trae por zpid el precio tabular, el refined,
el `delta_pct` aplicado y los flags de imagen para inspección fina.
