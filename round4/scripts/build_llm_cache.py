#!/home/matias/miniconda3/envs/labo2/bin/python
"""
Construye caches LLM para round4.

Default: OpenAI barato para todo. Requiere OPENAI_API_KEY en el entorno.
Fallback opcional: --provider ollama.

Run from round4/:
    scripts/build_llm_cache.py --splits train test

Opciones utiles:
    --skip-flags        solo embeddings semanticos
    --skip-embeddings   solo flags estructurados
    --limit 100         prueba chica sin tocar el cache final si se usa --cache-tag smoke
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

import features_llm as FL
import features_text as FT

TRAIN_TAB = Path("../participant/data/tabular/train_processed.csv")
TEST_TAB = Path("../participant/data/tabular/test_processed.csv")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_split(split: str, limit: int = 0) -> pd.DataFrame:
    path = TRAIN_TAB if split == "train" else TEST_TAB
    df = pd.read_csv(path, usecols=["zpid", FT.TEXT_COL])
    if limit:
        df = df.head(limit).copy()
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    p.add_argument("--provider", choices=["openai", "ollama"], default=FL.DEFAULT_PROVIDER)
    p.add_argument("--embedding-model", default=FL.DEFAULT_EMBED_MODEL)
    p.add_argument("--llm-model", default=FL.DEFAULT_LLM_MODEL)
    p.add_argument("--embedding-batch-size", type=int, default=256)
    p.add_argument("--flag-batch-size", type=int, default=64)
    p.add_argument("--flag-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cache-tag", default="")
    p.add_argument("--skip-embeddings", action="store_true")
    p.add_argument("--skip-flags", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log(
        f"provider={args.provider} | embed={args.embedding_model} | "
        f"llm={args.llm_model} | tag={args.cache_tag or 'default'}"
    )
    for split in args.splits:
        df = load_split(split, limit=args.limit)
        log(f"Split {split}: {len(df):,} filas")
        if not args.skip_embeddings:
            FL.build_embedding_cache(
                split,
                df,
                model=args.embedding_model,
                batch_size=args.embedding_batch_size,
                cache_tag=args.cache_tag,
                provider=args.provider,
            )
        if not args.skip_flags:
            FL.build_flag_cache(
                split,
                df,
                model=args.llm_model,
                batch_size=args.flag_batch_size,
                cache_tag=args.cache_tag,
                provider=args.provider,
                workers=args.flag_workers,
            )


if __name__ == "__main__":
    main()
