# Round 4 — Texto + LLM

Modelo equivalente a `round2` en formato, pero usando solamente la columna
`description`. El modelo final usa:

1. `features_text.py`: `TF-IDF` sobre texto crudo + `TruncatedSVD`.
2. `features_llm.py`: embeddings semanticos + flags estructurados del LLM, incluyendo indicios de casas baratas.
3. `features_text_llm.py`: concatena `TF-IDF + embeddings + flags`.
4. `train_oof.py`: OOF , ajustando SVD/vectorizadores dentro de cada fold.
5. `predict_test.py`: fit full sobre train y submission de test.

## Estructura

```bash
round4/
  scripts/
    features_text.py
    features_llm.py
    features_text_llm.py
    build_llm_cache.py
    train_oof.py
    predict_test.py
  llm_cache/
    *_emb_*.npy
    *_flags_*.csv
  submissions_train/
    practice_submission.csv
    comparison_oof.csv
    text_model_config.json
  submissions/
    submission.csv
```

## Comandos

```bash
cd round4

# 0) Setear key de OpenAI en la terminal, no en el codigo
export OPENAI_API_KEY="tu_key"

# 1) Smoke test chico, no toca el cache default
/home/matias/miniconda3/envs/labo2/bin/python scripts/build_llm_cache.py \
  --splits train --limit 8 --cache-tag smoke

# 2) Cache completo train+test
/home/matias/miniconda3/envs/labo2/bin/python scripts/build_llm_cache.py \
  --splits train test --flag-batch-size 64 --flag-workers 4

# 3) Practice OOF + diagnostico
/home/matias/miniconda3/envs/labo2/bin/python scripts/train_oof.py

# 4) Submission de test
/home/matias/miniconda3/envs/labo2/bin/python scripts/predict_test.py
```

Por defecto usa OpenAI barato:

- embeddings: `text-embedding-3-small`
- flags: `gpt-4.1-nano`

El cache queda guardado por chunks y se puede reanudar. Si se quiere volver a
Ollama local, agregar `--provider ollama --embedding-model llama3.2:latest
--llm-model llama3.2:latest`.

Para acelerar los flags con OpenAI, el script usa JSON compacto y puede correr
varios batches en paralelo con `--flag-workers`. Si aparece rate limit, bajar a
`--flag-workers 2`; si va estable, se puede probar `--flag-workers 6`.
