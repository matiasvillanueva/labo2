# Round 5 — Combina tabular + imagen (round3) + texto/LLM (round4)

`round5` **combina** las tres fuentes de senal de los rounds previos, sin regenerar
nada pesado: reutiliza por symlink los embeddings de imagen de `round3` y el cache
LLM (embeddings semanticos + flags) de `round4`.

No reemplaza a `round3`: lo toma como base fuerte y evalua si sumar texto/LLM predice
mejor. `train_oof.py` prueba dos formas de combinar y elige por ROI de la simulacion
del juego (`game_mechanics_es.md`), comparando siempre contra `round3`:

1. **feature_fusion**: un unico `LightGBM` cuantil (`tau=0.35`) que ve TODAS las
   columnas a la vez: tabular (round3) + imagen interpretable (round3) + `TF-IDF/SVD`
   + embeddings LLM/SVD + flags LLM (round4).
2. **text_residual** (fallback): se mantiene `round3` (tabular + imagen) como base y el
   texto/LLM entra como **residual acotado** sobre `log(real) - log(round3_oof)`, con
   `alpha` y `cap` tuneados en OOF. `alpha=0` equivale a `round3`, asi la combinacion
   nunca queda peor que `round3`.

Si la fusion no supera a `round3` en ROI, gana el residual. La estrategia elegida queda
en `round5_config.json`.

## Estructura

```bash
round5/
  data/tabular/        -> symlinks a participant/data/tabular/
  embeddings/          symlinks a los .npy/.csv/shards de round3/embeddings
                       (los artefactos chicos de runtime se escriben aca, no en round3)
  llm_cache/           -> symlink a round4/llm_cache (solo lectura)
  scripts/
    features.py                tabular (copia de round3)
    features_img.py            agregacion imagen por zpid (copia de round3)
    features_img_residual.py   escalares de imagen interpretables + gate (copia de round3)
    features_text.py           TF-IDF + SVD (copia de round4)
    features_llm.py            cache/carga de embeddings + flags LLM (copia de round4)
    features_text_llm.py       combo TF-IDF + emb LLM + flags (copia de round4)
    features_fusion.py         ensambla tabular + imagen + texto/LLM (nuevo, round5)
    build_llm_cache.py         construccion de cache LLM (copia de round4, solo si falta)
    simulate_roi.py            simulacion ROI del juego (copia de round3)
    train_oof.py               OOF: fusion vs residual, decide y compara vs round3 (nuevo)
    predict_test.py            fit final segun la estrategia elegida (nuevo)
  submissions_train/   practice_submission.csv, comparison_oof.csv,
                       diff_vs_round3.csv, diff_vs_round3_test.csv, round5_config.json
  submissions/         submission.csv
```

## Comandos

```bash
cd round5

# 1) OOF: elige estrategia (fusion vs residual) y compara contra round3
/home/matias/miniconda3/envs/labo2/bin/python scripts/train_oof.py

# 2) Submission de test (usa round5_config.json del paso 1)
/home/matias/miniconda3/envs/labo2/bin/python scripts/predict_test.py
```

Los embeddings de imagen (`round3`) y el cache LLM (`round4`) ya existen y se reutilizan
por symlink; no hay que reextraer nada ni volver a llamar al LLM.

## Que mirar

`train_oof.py` imprime, en OOF y medido por ROI sim, tres filas comparables:

- `round3` — base tabular + imagen (baseline a superar)
- `feature_fusion` — todas las columnas en un solo modelo
- `text_residual` — round3 + texto/LLM acotado (mejor `alpha`/`cap`)

Salidas para inspeccion fina:

- `comparison_oof.csv`: por zpid, el precio real, el de round3, el de cada estrategia y
  el del ganador, con el error porcentual.
- `diff_vs_round3.csv`: diferencia por zpid vs round3 (precio y error), y una bandera
  `mejora` (1 = round5 tiene menor error que round3 en esa propiedad).
- `diff_vs_round3_test.csv`: mismo diff pero sobre la submission de test.
- `round5_config.json`: estrategia ganadora, `alpha`/`cap` del residual y las metricas
  de las tres opciones.

## Resultado de esta corrida

OOF (3 seeds x 5 folds), medido por ROI sim del juego:

| estrategia       | MAE        | wMAPE   | R2(log) | ROI     |
|------------------|-----------:|--------:|--------:|--------:|
| round3 (base)    | $117,491   | 21.02%  | 0.7481  | +1.85%  |
| feature_fusion   | $126,135   | 22.56%  | 0.7278  | +1.27%  |
| text_residual    | $117,491   | 21.02%  | 0.7481  | +1.85%  |

Conclusion honesta: **el texto/LLM no mejora a round3 en este dataset**. La fusion por
columnas empeora (diluye la senal tabular fuerte con 512 columnas de SVD), y el residual
de texto elige `alpha=0`, es decir, `round5 == round3` (sin regresion). Verificaciones:

- Correlacion de features densas de texto (lexicos, `$` mencionados, mayusculas) con el
  residual de round3: maximo |corr| ~0.04.
- OOF R2 sobre el residual: flags LLM viejos (17, densos) `-0.0025`; embeddings LLM
  (1536) `-0.0078`. Negativo => no explican el residual.
- Incluso en el slice `gate=0` (imagen apagada, fotos escasas), embeddings LLM dan R2
  `-0.025`. El texto no sustituye a la imagen ahi.

Motivo: la tabular de round1/round3 ya incorpora senales de la descripcion
(`desc_length`, `desc_mentions_renovated/pool/view/new`, `desc_is_boilerplate`) y el
precio queda bien determinado por tabular (m2, ambientes, ZIP quantiles, KNN, impuestos,
last_listing) + imagen. El residual restante (~33% en log) es ruido idiosincratico de la
venta que el texto no predice.

El pipeline queda listo para incorporar el texto automaticamente si en el futuro se
agregan columnas con senal real: `train_oof.py` re-tunea `alpha` y lo activaria solo si
sube el ROI OOF.
