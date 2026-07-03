# Round 2 — Modelo de precio SOLO con imagenes

Predice el precio de venta usando **unicamente** las fotos del listado (embeddings
CLIP + tipo de foto + materiales). No usa features tabulares ni nada de `round1`
(el target `log_price` se usa solo como etiqueta de entrenamiento).

## Pipeline

```
extract_embeddings.py  (1 sola vez, lento)  ->  embeddings/img_emb_*.npy + img_meta_*.csv
        |
features_img.py  (agrega por propiedad + PCA, importado por los de abajo)
        |
train_oof.py    ->  submissions_train/practice_submission.csv   (OOF, para SUBIR)
predict_test.py ->  submissions/submission.csv                  (test, para SUBIR)
analyze_signal.py -> reporte de impacto (tipo / materiales / sufijo)
```

## Como correr

Desde la carpeta `round2/`.

### 1) Extraer embeddings (UNA vez, ~1.5h en CPU; resumible)

```bash
cd round2
MAX_IMAGES_PER_PROP=4 /home/matias/miniconda3/envs/llms/bin/python scripts/extract_embeddings.py
```

- Es **resumible**: si lo cortas (Ctrl-C), al re-ejecutarlo saltea los shards ya
  hechos en `embeddings/shards/`.
- Avance: imprime `img/s` y `ETA` por shard. Tambien podes ver cuantos shards hay:
  `ls embeddings/shards | wc -l`.
- Ajustes por variable de entorno: `MAX_IMAGES_PER_PROP` (fotos por casa, default 4),
  `BATCH_SIZE` (64), `CHUNK_SIZE` (2000). Si tu CPU va lento, bajalo a `MAX_IMAGES_PER_PROP=3`.
- Si conseguis GPU (Colab), el mismo script autodetecta `cuda` y vuela.

### 2) OOF sobre train (Practice Submission)

```bash
/home/matias/miniconda3/envs/labo2/bin/python scripts/train_oof.py
```

Genera `submissions_train/practice_submission.csv` (`zpid,predicted_price`) e
imprime metricas OOF honestas (R² log, MAE $, wMAPE).

### 3) Prediccion sobre test (Submission)

```bash
/home/matias/miniconda3/envs/labo2/bin/python scripts/predict_test.py
```

Genera `submissions/submission.csv` (`zpid,predicted_price`).

### 4) (opcional) Analisis de impacto

```bash
/home/matias/miniconda3/envs/labo2/bin/python scripts/analyze_signal.py
```

## Notas

- `extract_embeddings.py` necesita `torch`+`transformers` (env `llms`); el resto
  necesita `lightgbm`+`sklearn` (env `labo2`).
- La 1a corrida descarga los pesos de CLIP (~600MB) desde HuggingFace (requiere internet).
- Propiedades sin foto se rellenan con la mediana de prediccion (el modelo es solo-imagen).
