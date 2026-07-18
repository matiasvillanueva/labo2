# Round 6 — valuacion precisa + edge selectivo

Pipeline autocontenido dentro de `round6/`. Conserva la precision de round5/base y
agrega un **edge selectivo** contra sobrevaluaciones peligrosas. No baja todas las
predicciones: aprende OOF cuales tienen mayor riesgo y solo recorta esas.

## Motivacion: resultado real de round4

El export `2026-07-18T14-39_export.csv` corresponde a **round4**, no round5:

- 250 propiedades ofrecidas
- 11 ganadas
- 10 perdidas y solo 1 rentable
- profit total: **-$1,977,004**

En numerosas subastas otro competidor empato exactamente nuestra oferta. En ese caso
el costo queda aproximadamente:

```text
cost ~= predicted_price * 0.85 * 1.02
```

Por lo tanto, una prediccion es peligrosa si:

```text
predicted_price / true_value > 1 / (0.85 * 1.02) = 1.153
```

En las 11 compras reveladas, round4 tenia predicciones extremadamente altas. Round5
las reduce mucho, pero todavia deja varios outliers. Esto explica por que un wMAPE
global bueno no garantiza un buen P&L: las propiedades efectivamente ganadas son una
muestra sesgada hacia las sobrevaluaciones del modelo.

## Solucion

### 1. Base precisa

- LightGBM quantile `tau=0.35`
- features tabulares + ZIP/KNN
- residual de imagen gated y limitado a +/-8%
- wMAPE OOF cercano al excelente resultado de round5

Se descarto como salida final el experimento `tau=0.30 + sample_weight`: daba buen ROI
en el simulador local, pero empeoraba wMAPE de 21.04% a 25.67%. El feedback real indica
que no conviene destruir la valuacion general para optimizar un simulador sintetico.

### 2. Edge selectivo

`features_edge.py` entrena dos modelos sobre predicciones OOF de la base:

1. clasificador: probabilidad de que `prediction / true > 1.153`;
2. regresor: severidad esperada de la sobrevaluacion en log-precio.

Features principales:

- prediccion base;
- ratios contra tax assessment, ultimo tax y listing valido;
- precio predicho por sqft;
- propiedad, ubicacion, fotos y banda lexica de descripcion.

La correccion final es:

```text
si edge_risk >= threshold:
    final_log_price = base_log_price - alpha * predicted_overvaluation
si no:
    final_price = base_price
```

La grilla OOF busca `threshold` y `alpha`, pero impone:

```text
wMAPE_edge <= wMAPE_base + 0.25 puntos
```

Ganador: `threshold=0.25`, `alpha=0.50`.

### 3. Texto y fotos

Se mantienen las features de la iteracion anterior:

- banda baja: mobile/manufactured, starter, fixer, foreclosure, investor;
- banda alta: luxury y waterfront;
- tono: price-cut, priced-to-sell, below-market;
- `log_photo_count`, `photos_per_100sqft`, `photo_vs_zip_median`;
- interaccion pocas fotos x casa grande.

No se extraen `last sold for $X` ni `Zestimate $X`: son leakage.

## Resultado OOF

3 seeds x 5 folds:

| estrategia | MAE | wMAPE | ROI sim | win positivo | traps | Q1 MAPE |
|---|---:|---:|---:|---:|---:|---:|
| base precisa | $117,636 | **21.04%** | +1.74% | 80% | 47.1% | 49.7% |
| **base + edge** | $118,790 | **21.25%** | **+1.93%** | **85%** | **46.4%** | **47.5%** |

El edge:

- afecta 24.4% de las propiedades;
- recorta solo 1.60% en promedio global;
- conserva el wMAPE bajo;
- mejora ROI local y Q1;
- reduce la tasa de sobreestimacion Q1 de 65.2% a 62.2%.

## Comparacion sobre las 11 compras reveladas de round4

Es una muestra chica y sesgada (solo conocemos `true_value` de propiedades ganadas),
por lo que sirve como diagnostico, no como validacion:

| modelo | MAE | MAPE | sobrevaluaciones peligrosas | P&L teorico si paga su oferta empatada |
|---|---:|---:|---:|---:|
| round4 export | $261,602 | 181.0% | 10/11 | -$1,974,576 |
| round5 | $123,410 | 101.9% | 7/11 | -$547,131 |
| round6 base | $123,508 | 102.0% | 7/11 | -$547,901 |
| **round6 edge** | **$113,281** | **97.7%** | **6/11** | **-$435,301** |

Round6 edge mejora round5 en este slice, pero no garantiza P&L positivo: quedan
outliers que no se pueden detectar perfectamente con las columnas disponibles. La
mejora es deliberadamente moderada para no sobreajustar 11 etiquetas reveladas.

## Archivos

```text
scripts/features_desc_price.py   banda de precio desde texto
scripts/features.py              tabular + fotos + encodings OOF-safe
scripts/features_img_residual.py residual de imagen
scripts/features_edge.py         riesgo y severidad de sobrevaluacion
scripts/train_oof.py              base OOF + tune de edge con limite de wMAPE
scripts/predict_test.py           fit final + edge

submissions_train/comparison_oof.csv
submissions_train/oof_tabular.csv
submissions_train/oof_edge.csv
submissions_train/round6_config.json
submissions/test_edge_diagnostics.csv
submissions/submission.csv
```

## Ejecucion

```bash
cd round6
/home/matias/miniconda3/envs/labo2/bin/python scripts/train_oof.py
/home/matias/miniconda3/envs/labo2/bin/python scripts/predict_test.py
```
