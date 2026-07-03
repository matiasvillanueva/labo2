"""
Round 4 — Cache y carga de features LLM.

Default: OpenAI barato para todo:
  - embeddings: text-embedding-3-small
  - flags: gpt-4.1-nano

Fallback opcional: Ollama local con --provider ollama.
Los caches viven en round4/llm_cache para no recalcular llamadas.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import features_text as FT

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DEFAULT_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-nano")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
CACHE_DIR = Path("llm_cache")

FLAG_COLUMNS = [
    "llm_renovated",
    "llm_needs_work",
    "llm_luxury",
    "llm_waterfront_view",
    "llm_pool_spa",
    "llm_gated_security",
    "llm_new_construction",
    "llm_investment_distressed",
    "llm_furnished",
    "llm_hoa_condo_amenities",
    "llm_garage_parking",
    "llm_outdoor_space",
    "llm_walkable_location",
    "llm_quiet_private",
    "llm_condition_score",
    "llm_luxury_score",
    "llm_location_score",
]
FLAG_SHORT_NAMES = [
    "ren",
    "work",
    "lux",
    "water",
    "pool",
    "gated",
    "new",
    "distress",
    "furn",
    "hoa",
    "park",
    "outdoor",
    "walk",
    "quiet",
    "cond",
    "lux_score",
    "loc_score",
]


def slug_model(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def cache_suffix(provider: str, model: str, cache_tag: str = "") -> str:
    suffix = f"{slug_model(provider)}_{slug_model(model)}"
    if cache_tag:
        suffix = f"{suffix}_{slug_model(cache_tag)}"
    return suffix


def embedding_paths(
    split: str,
    model: str = DEFAULT_EMBED_MODEL,
    cache_tag: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> tuple[Path, Path]:
    suffix = cache_suffix(provider, model, cache_tag)
    return (
        CACHE_DIR / f"{split}_emb_{suffix}.npy",
        CACHE_DIR / f"{split}_emb_{suffix}_meta.csv",
    )


def embedding_chunk_dir(
    split: str,
    model: str = DEFAULT_EMBED_MODEL,
    cache_tag: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> Path:
    return CACHE_DIR / "embedding_chunks" / f"{split}_{cache_suffix(provider, model, cache_tag)}"


def flag_path(
    split: str,
    model: str = DEFAULT_LLM_MODEL,
    cache_tag: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> Path:
    return CACHE_DIR / f"{split}_flags_{cache_suffix(provider, model, cache_tag)}.csv"


def flag_chunk_dir(
    split: str,
    model: str = DEFAULT_LLM_MODEL,
    cache_tag: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> Path:
    return CACHE_DIR / "flag_chunks" / f"{split}_{cache_suffix(provider, model, cache_tag)}"


def _post_json_url(
    url: str,
    payload: dict,
    headers: dict | None = None,
    timeout: int = 300,
    retries: int = 4,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", **(headers or {})}
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt == retries:
                raise RuntimeError(f"HTTP {exc.code} calling {url}: {error_body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Error calling {url}: {exc}") from exc
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Error calling {url}")


def _openai_headers() -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY. Exportala antes de correr build_llm_cache.py")
    return {"Authorization": f"Bearer {api_key}"}


def _ollama_post(endpoint: str, payload: dict, timeout: int = 300) -> dict:
    return _post_json_url(f"{OLLAMA_HOST}{endpoint}", payload, timeout=timeout)


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_EMBED_MODEL,
    provider: str = DEFAULT_PROVIDER,
) -> np.ndarray:
    provider = provider.lower()
    if provider == "openai":
        data = _post_json_url(
            f"{OPENAI_BASE_URL}/embeddings",
            {"model": model, "input": texts},
            headers=_openai_headers(),
            timeout=300,
        )
        ordered = sorted(data["data"], key=lambda item: item["index"])
        return np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
    if provider == "ollama":
        data = _ollama_post(
            "/api/embed",
            {"model": model, "input": texts, "keep_alive": "30m"},
            timeout=600,
        )
        return np.asarray(data["embeddings"], dtype=np.float32)
    raise ValueError(f"Provider no soportado: {provider}")


def _flag_prompt(items: list[dict]) -> str:
    return (
        "Extract numeric real-estate signals from listing descriptions.\n"
        "Use only explicit text. Unknown/not mentioned = 0. Scores are integers 0..3.\n"
        "Return ONLY compact JSON: {\"rows\":[[zpid,ren,work,lux,water,pool,gated,new,"
        "distress,furn,hoa,park,outdoor,walk,quiet,cond,lux_score,loc_score],...]}\n"
        "Meanings: ren=renovated/updated; work=TLC/fixer/as-is; lux=luxury/high-end; "
        "water=water/ocean/lake/canal view/front; pool=pool/spa; gated=gated/security; "
        "new=new construction; distress=foreclosure/short sale/investor/cash only; "
        "furn=furnished/turnkey; hoa=condo/HOA amenities; park=garage/parking; "
        "outdoor=patio/balcony/yard/deck; walk=walkable/near beach/downtown/shops/transit; "
        "quiet=private/quiet/secluded; cond=overall condition score; lux_score=luxury score; "
        "loc_score=location/view amenity score.\n"
        f"Descriptions:\n{json.dumps({'items': items}, ensure_ascii=True)}"
    )


def _coerce_number(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y"}:
                return 1.0
            if lowered in {"false", "no", "n", ""}:
                return 0.0
        return default


def _normalize_flag_rows(items: list[dict], zpids: Iterable[int]) -> pd.DataFrame:
    rows = {int(z): {col: 0.0 for col in FLAG_COLUMNS} for z in zpids}
    for item in items:
        try:
            zpid = int(item["zpid"])
        except (KeyError, TypeError, ValueError):
            continue
        if zpid not in rows:
            continue
        for col in FLAG_COLUMNS:
            short = col.removeprefix("llm_")
            rows[zpid][col] = _coerce_number(item.get(col, item.get(short)), 0.0)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "zpid"
    return out.reset_index()


def _normalize_compact_flag_rows(rows: list, zpids: Iterable[int]) -> pd.DataFrame:
    expected = 1 + len(FLAG_COLUMNS)
    by_zpid = {int(z): {col: 0.0 for col in FLAG_COLUMNS} for z in zpids}
    for row in rows:
        if not isinstance(row, list) or len(row) < expected:
            continue
        try:
            zpid = int(row[0])
        except (TypeError, ValueError):
            continue
        if zpid not in by_zpid:
            continue
        for col, value in zip(FLAG_COLUMNS, row[1:expected]):
            by_zpid[zpid][col] = _coerce_number(value, 0.0)
    out = pd.DataFrame.from_dict(by_zpid, orient="index")
    out.index.name = "zpid"
    return out.reset_index()


def _normalize_flags(parsed: dict, zpids: Iterable[int]) -> pd.DataFrame:
    if "rows" in parsed:
        return _normalize_compact_flag_rows(parsed.get("rows", []), zpids)
    return _normalize_flag_rows(parsed.get("items", []), zpids)


def extract_flags(
    batch: pd.DataFrame,
    model: str = DEFAULT_LLM_MODEL,
    provider: str = DEFAULT_PROVIDER,
) -> pd.DataFrame:
    items = [
        {"zpid": int(row.zpid), "description": str(row.description)[:1400]}
        for row in batch[["zpid", FT.TEXT_COL]].itertuples(index=False)
    ]
    prompt = _flag_prompt(items)
    provider = provider.lower()
    if provider == "openai":
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You extract structured real-estate features as JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
        }
        data = _post_json_url(
            f"{OPENAI_BASE_URL}/chat/completions",
            payload,
            headers=_openai_headers(),
            timeout=300,
        )
        parsed = json.loads(data["choices"][0]["message"]["content"])
        return _normalize_flags(parsed, batch["zpid"].values)
    if provider == "ollama":
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": 4096},
        }
        data = _ollama_post("/api/generate", payload, timeout=900)
        parsed = json.loads(data.get("response", "{}"))
        return _normalize_flags(parsed, batch["zpid"].values)
    raise ValueError(f"Provider no soportado: {provider}")


def build_embedding_cache(
    split: str,
    df: pd.DataFrame,
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 256,
    cache_tag: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    chunk_dir = embedding_chunk_dir(split, model, cache_tag, provider)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    texts = FT.clean_text(df).tolist()
    zpids = df["zpid"].astype(np.int64).to_numpy()
    for start in range(0, len(df), batch_size):
        out = chunk_dir / f"{start:06d}.npz"
        if out.exists():
            continue
        stop = min(start + batch_size, len(df))
        t0 = time.time()
        emb = embed_texts(texts[start:stop], model=model, provider=provider)
        np.savez_compressed(out, zpid=zpids[start:stop], embedding=emb)
        print(f"[embed {split}] {stop:,}/{len(df):,} ({time.time() - t0:.1f}s)", flush=True)
    materialize_embedding_cache(split, model, cache_tag, provider)


def materialize_embedding_cache(
    split: str,
    model: str = DEFAULT_EMBED_MODEL,
    cache_tag: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> None:
    emb_path, meta_path = embedding_paths(split, model, cache_tag, provider)
    chunks = sorted(embedding_chunk_dir(split, model, cache_tag, provider).glob("*.npz"))
    if not chunks:
        raise FileNotFoundError(f"No hay chunks de embeddings para {split}")
    zpids, arrays = [], []
    for path in chunks:
        data = np.load(path)
        zpids.append(data["zpid"])
        arrays.append(data["embedding"])
    emb = np.vstack(arrays).astype(np.float32)
    zpid = np.concatenate(zpids)
    np.save(emb_path, emb)
    pd.DataFrame({"zpid": zpid}).to_csv(meta_path, index=False)
    print(f"[embed {split}] materializado {emb_path} {emb.shape}", flush=True)


def _build_flag_batch(
    batch: pd.DataFrame,
    model: str,
    provider: str,
    min_batch_size: int = 1,
) -> pd.DataFrame:
    try:
        return extract_flags(batch, model=model, provider=provider)
    except Exception as exc:
        if len(batch) <= min_batch_size:
            print(f"[flags] fallback zeros para {len(batch)} filas: {exc}", flush=True)
            return _normalize_flag_rows([], batch["zpid"].values)
        mid = len(batch) // 2
        left = _build_flag_batch(batch.iloc[:mid], model, provider, min_batch_size)
        right = _build_flag_batch(batch.iloc[mid:], model, provider, min_batch_size)
        return pd.concat([left, right], ignore_index=True)


def build_flag_cache(
    split: str,
    df: pd.DataFrame,
    model: str = DEFAULT_LLM_MODEL,
    batch_size: int = 16,
    cache_tag: str = "",
    provider: str = DEFAULT_PROVIDER,
    workers: int = 4,
) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    chunk_dir = flag_chunk_dir(split, model, cache_tag, provider)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    cached_zpids: set[int] = set()
    for path in chunk_dir.glob("*.csv"):
        try:
            cached_zpids.update(pd.read_csv(path, usecols=["zpid"])["zpid"].astype(int).tolist())
        except Exception:
            continue

    pending_pos = [
        i for i, zpid in enumerate(df["zpid"].astype(int).tolist())
        if zpid not in cached_zpids
    ]
    jobs = []
    for offset in range(0, len(pending_pos), batch_size):
        positions = pending_pos[offset:offset + batch_size]
        if not positions:
            continue
        start = positions[0]
        out = chunk_dir / f"{start:06d}.csv"
        jobs.append((positions, out))

    def run_job(job: tuple[list[int], Path]) -> tuple[int, float]:
        positions, out = job
        stop = positions[-1] + 1
        t0 = time.time()
        flags = _build_flag_batch(df.iloc[positions], model=model, provider=provider)
        flags.to_csv(out, index=False)
        return stop, time.time() - t0

    max_workers = max(1, workers if provider.lower() == "openai" else 1)
    if max_workers == 1:
        for job in jobs:
            stop, elapsed = run_job(job)
            print(f"[flags {split}] {stop:,}/{len(df):,} ({elapsed:.1f}s)", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_job, job) for job in jobs]
            for fut in as_completed(futures):
                stop, elapsed = fut.result()
                print(
                    f"[flags {split}] {stop:,}/{len(df):,} ({elapsed:.1f}s, workers={max_workers})",
                    flush=True,
                )
    materialize_flag_cache(split, model, cache_tag, provider)


def materialize_flag_cache(
    split: str,
    model: str = DEFAULT_LLM_MODEL,
    cache_tag: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> None:
    out = flag_path(split, model, cache_tag, provider)
    chunks = sorted(flag_chunk_dir(split, model, cache_tag, provider).glob("*.csv"))
    if not chunks:
        raise FileNotFoundError(f"No hay chunks de flags para {split}")
    flags = pd.concat([pd.read_csv(path) for path in chunks], ignore_index=True)
    flags = flags.drop_duplicates("zpid", keep="last")
    flags.to_csv(out, index=False)
    print(f"[flags {split}] materializado {out} {flags.shape}", flush=True)


def load_embedding_matrix(
    split: str,
    df: pd.DataFrame,
    model: str = DEFAULT_EMBED_MODEL,
    cache_tag: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> np.ndarray:
    emb_path, meta_path = embedding_paths(split, model, cache_tag, provider)
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Falta cache de embeddings {split}. Corre: "
            f"scripts/build_llm_cache.py --splits {split} --skip-flags"
        )
    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    pos = pd.Series(np.arange(len(meta)), index=meta["zpid"].astype(np.int64))
    idx = df["zpid"].astype(np.int64).map(pos)
    if idx.isna().any():
        missing = int(idx.isna().sum())
        raise ValueError(f"Faltan embeddings para {missing} zpids de {split}")
    return emb[idx.astype(int).values]


def load_flag_features(
    split: str,
    df: pd.DataFrame,
    model: str = DEFAULT_LLM_MODEL,
    cache_tag: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> pd.DataFrame:
    path = flag_path(split, model, cache_tag, provider)
    if not path.exists():
        raise FileNotFoundError(
            f"Falta cache de flags {split}. Corre: "
            f"scripts/build_llm_cache.py --splits {split} --skip-embeddings"
        )
    flags = pd.read_csv(path).set_index("zpid")
    out = flags.reindex(df["zpid"].values)[FLAG_COLUMNS].fillna(0.0)
    out.index = df.index
    return out.astype(np.float32)
