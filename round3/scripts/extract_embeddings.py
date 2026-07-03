"""
Round 3 — Extraccion de embeddings de imagen (DINOv2 + SigLIP).

Identico a round2 (los embeddings ya generados se movieron a round3/embeddings/).
Solo re-correr si faltan fotos o se cambia MAX_IMAGES_PER_PROP.

Por cada foto (cap configurable por propiedad):
  - embedding visual 768-d (DINOv2-base, CLS token, L2-normalizado, float16)
  - tipo de foto via zero-shot SigLIP: interior / exterior / satellite / floorplan
  - scores de materiales/calidad via zero-shot SigLIP

Es RESUMIBLE: shards en embeddings/shards/ con prefijo v2_.

Entorno:
  cd round3
  MAX_IMAGES_PER_PROP=9999 /home/matias/miniconda3/envs/llms/bin/python scripts/extract_embeddings.py
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoImageProcessor, SiglipModel, SiglipProcessor

PARTICIPANT = Path("../participant")
DATA_DIR = PARTICIPANT / "data"
META = {
    "train": DATA_DIR / "train_photo_metadata.csv",
    "test": DATA_DIR / "test_photo_metadata.csv",
}
OUT_DIR = Path("embeddings")
SHARD_DIR = OUT_DIR / "shards"
DINO_NAME = "facebook/dinov2-base"
SIGLIP_NAME = "google/siglip-base-patch16-224"
EMB_DIM = 768
SHARD_PREFIX = "v2"

MAX_IMAGES_PER_PROP = int(os.environ.get("MAX_IMAGES_PER_PROP", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "0"))  # 0 = auto (32 cpu / 64 cuda)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "2000"))
SPLITS = os.environ.get("SPLITS", "train,test").split(",")
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))

TYPE_PROMPTS = {
    "interior": "a photo of the interior of a house, a room inside a home",
    "exterior": "a photo of the exterior of a house, the front of a building",
    "satellite": "an aerial satellite top-down view of a property and land",
    "floorplan": "a floor plan diagram of a house layout",
}
TYPE_KEYS = list(TYPE_PROMPTS.keys())

MATERIAL_PROMPTS = {
    "granite": "a kitchen with granite or marble countertops",
    "stainless": "a kitchen with stainless steel appliances",
    "hardwood": "a room with hardwood wood floors",
    "tile": "a room with tile or marble floors",
    "modern_kitchen": "a modern renovated luxury kitchen",
    "luxury": "a luxurious expensive high-end home interior",
    "outdated": "an old outdated home interior in need of renovation",
    "pool": "a backyard with a swimming pool",
    "waterfront": "a waterfront property with an ocean or lake view",
    "spacious": "a large spacious bright room with high ceilings",
}
MAT_KEYS = list(MATERIAL_PROMPTS.keys())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def suffix_from_filename(fn: str) -> int:
    try:
        return int(os.path.splitext(fn)[0].split("_")[-1])
    except Exception:
        return -1


def build_tasks(split: str) -> pd.DataFrame:
    df = pd.read_csv(META[split], usecols=["zpid", "image_index", "filename", "image_path"])
    df = df.sort_values(["zpid", "image_index"]).groupby("zpid", sort=False).head(MAX_IMAGES_PER_PROP)
    df = df.reset_index(drop=True)
    df["suffix"] = df["filename"].map(suffix_from_filename)
    df["abs_path"] = df["image_path"].map(lambda p: str(PARTICIPANT / p))
    if MAX_TASKS > 0:
        df = df.head(MAX_TASKS).reset_index(drop=True)
    return df


def load_models(device):
    log(f"Cargando DINOv2 {DINO_NAME} + SigLIP {SIGLIP_NAME} (1a vez descarga pesos)...")
    dino = AutoModel.from_pretrained(DINO_NAME).to(device).eval()
    dino_proc = AutoImageProcessor.from_pretrained(DINO_NAME)
    siglip = SiglipModel.from_pretrained(SIGLIP_NAME).to(device).eval()
    sig_proc = SiglipProcessor.from_pretrained(SIGLIP_NAME)
    return dino, dino_proc, siglip, sig_proc


@torch.no_grad()
def encode_siglip_text(siglip, sig_proc, device):
    prompts = list(TYPE_PROMPTS.values()) + list(MATERIAL_PROMPTS.values())
    inp = sig_proc(text=prompts, padding="max_length", return_tensors="pt").to(device)
    feats = siglip.get_text_features(**inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    n_type = len(TYPE_PROMPTS)
    return feats[:n_type], feats[n_type:]


@torch.no_grad()
def process_batch(dino, dino_proc, siglip, sig_proc, device, paths,
                  type_text, mat_text, logit_scale):
    imgs, ok = [], []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
            ok.append(1)
        except Exception:
            imgs.append(Image.new("RGB", (224, 224)))
            ok.append(0)

    # DINOv2 embedding (768-d CLS)
    dino_in = dino_proc(images=imgs, return_tensors="pt").to(device)
    dino_out = dino(**dino_in)
    dino_emb = dino_out.last_hidden_state[:, 0]
    dino_emb = dino_emb / dino_emb.norm(dim=-1, keepdim=True)

    # SigLIP zero-shot scores
    sig_in = sig_proc(images=imgs, return_tensors="pt").to(device)
    sig_emb = siglip.get_image_features(**sig_in)
    sig_emb = sig_emb / sig_emb.norm(dim=-1, keepdim=True)

    type_logits = logit_scale * sig_emb @ type_text.T
    type_probs = type_logits.softmax(dim=-1)
    mat_sims = sig_emb @ mat_text.T

    return (
        dino_emb.cpu().numpy().astype(np.float16),
        type_probs.cpu().numpy().astype(np.float32),
        mat_sims.cpu().numpy().astype(np.float32),
        np.array(ok, dtype=np.int8),
    )


def shard_paths(split: str, idx: int):
    p = SHARD_PREFIX
    return (
        SHARD_DIR / f"{split}_{p}_emb_{idx:05d}.npy",
        SHARD_DIR / f"{split}_{p}_meta_{idx:05d}.csv",
    )


def process_split(split, dino, dino_proc, siglip, sig_proc, device,
                  type_text, mat_text, logit_scale, batch_size):
    tasks = build_tasks(split)
    n = len(tasks)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    n_props = tasks["zpid"].nunique()
    log(f"[{split}] {n_props:,} propiedades -> {n:,} imagenes "
        f"(cap {MAX_IMAGES_PER_PROP}/prop, DINOv2+SigLIP) en {n_chunks} shards")

    pending = [c for c in range(n_chunks) if not shard_paths(split, c)[0].exists()]
    done = n_chunks - len(pending)
    if done:
        log(f"[{split}] {done}/{n_chunks} shards ya existen -> reanudando")
    if not pending:
        log(f"[{split}] todos los shards listos")
        finalize_split(split, n_chunks)
        return

    t0 = time.time()
    seen = 0
    for ci, c in enumerate(pending):
        lo, hi = c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, n)
        chunk = tasks.iloc[lo:hi]
        embs, tprobs, msims, oks = [], [], [], []
        paths = chunk["abs_path"].tolist()
        for b in tqdm(range(0, len(paths), batch_size),
                      desc=f"{split} shard {c+1}/{n_chunks}", leave=False):
            bp = paths[b:b + batch_size]
            e, tp, ms, ok = process_batch(
                dino, dino_proc, siglip, sig_proc, device, bp,
                type_text, mat_text, logit_scale)
            embs.append(e); tprobs.append(tp); msims.append(ms); oks.append(ok)

        emb = np.vstack(embs)
        tprob = np.vstack(tprobs)
        msim = np.vstack(msims)
        ok = np.concatenate(oks)

        meta = chunk[["zpid", "image_index", "suffix"]].copy()
        meta["ok"] = ok
        meta["type_argmax"] = [TYPE_KEYS[i] for i in tprob.argmax(axis=1)]
        for j, k in enumerate(TYPE_KEYS):
            meta[f"prob_{k}"] = tprob[:, j]
        for j, k in enumerate(MAT_KEYS):
            meta[f"mat_{k}"] = msim[:, j]

        emb_path, meta_path = shard_paths(split, c)
        np.save(emb_path, emb)
        meta.to_csv(meta_path, index=False)

        seen += len(chunk)
        rate = seen / (time.time() - t0)
        remaining = sum(min((cc + 1) * CHUNK_SIZE, n) - cc * CHUNK_SIZE
                        for cc in pending[ci + 1:])
        eta_min = (remaining / rate / 60) if rate > 0 else float("nan")
        log(f"[{split}] shard {c+1}/{n_chunks} ok | {rate:.1f} img/s | "
            f"ETA split ~{eta_min:.1f} min")

    finalize_split(split, n_chunks)


def finalize_split(split, n_chunks):
    if any(not shard_paths(split, c)[0].exists() for c in range(n_chunks)):
        log(f"[{split}] faltan shards, no concateno todavia")
        return
    embs, metas = [], []
    for c in range(n_chunks):
        ep, mp = shard_paths(split, c)
        embs.append(np.load(ep))
        metas.append(pd.read_csv(mp))
    emb = np.vstack(embs)
    meta = pd.concat(metas, ignore_index=True)
    np.save(OUT_DIR / f"img_emb_{split}.npy", emb)
    meta.to_csv(OUT_DIR / f"img_meta_{split}.csv", index=False)
    log(f"[{split}] FINAL -> img_emb_{split}.npy {emb.shape} (DINOv2 {EMB_DIM}-d) | "
        f"img_meta_{split}.csv ({len(meta):,} filas, {meta['ok'].mean():.1%} ok)")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = BATCH_SIZE or (64 if device.type == "cuda" else 24)
    log(f"Device: {device} | batch={batch_size} | cap={MAX_IMAGES_PER_PROP} | "
        f"DINOv2+SigLIP")

    dino, dino_proc, siglip, sig_proc = load_models(device)
    type_text, mat_text = encode_siglip_text(siglip, sig_proc, device)
    logit_scale = siglip.logit_scale.exp().item()

    for split in SPLITS:
        split = split.strip()
        if split not in META:
            continue
        process_split(split, dino, dino_proc, siglip, sig_proc, device,
                      type_text, mat_text, logit_scale, batch_size)

    log("Listo. Proximos pasos: train_oof.py y predict_test.py (env labo2).")


if __name__ == "__main__":
    main()
