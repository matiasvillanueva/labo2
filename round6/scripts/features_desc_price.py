"""
Round 6 — Features lexicas de "banda de precio" desde la descripcion.

Regex/lexico puro (sin LLM, sin TF-IDF): banderas densas y baratas de calcular que
ubican una propiedad en una banda de precio (baja / alta) y el tono comercial del
anuncio. La idea es dar al modelo pistas de mercado que la tabular no captura, sobre
todo para no sobreestimar las casas baratas (mobile/manufactured, fixer, distress).

IMPORTANTE — sin leakage: NO se parsea ningun monto en dolares del texto
(`last sold for $X`, `Zestimate $X`, etc.). Ese boilerplate de Zillow es casi el
target y filtrarlo seria trampa. Aca solo miramos vocabulario cualitativo.

API:
  FEATURE_COLUMNS               -> list[str] con los nombres de columnas que genera
  build(df) -> DataFrame        -> mismas filas/index que df, columnas = FEATURE_COLUMNS
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

# ── Banda BAJA (asociada a precios por debajo de la mediana del barrio) ──────────
LOW_BAND = {
    "desc_mobile_manufactured": (
        r"mobile\s+home|manufactured\s+(home|hous)|trailer\b|"
        r"single[\s-]?wide|double[\s-]?wide|hud\s*code|on\s+leased\s+land"
    ),
    "desc_starter_affordable": (
        r"starter\s+home|first[\s-]?time\s+(home\s*)?buyer|affordable|"
        r"budget[\s-]?friendly|entry[\s-]?level|great\s+starter"
    ),
    "desc_fixer_tlc": (
        r"fixer[\s-]?upper|needs?\s+(some\s+)?(work|tlc|updating|update|repair|love)|"
        r"handyman|as[\s-]is|\btlc\b|bring\s+your\s+(tools|imagination|contractor)|"
        r"sweat\s+equity|diamond\s+in\s+the\s+rough|potential\s+galore"
    ),
    "desc_foreclosure_distress": (
        r"foreclosure|short\s+sale|bank[\s-]?owned|\breo\b|auction|distress(ed)?|"
        r"estate\s+sale|must\s+sell|motivated\s+seller|probate"
    ),
    "desc_investor_language": (
        r"investor(s)?\s+(special|opportunity|dream|alert)|investment\s+(opportunity|property)|"
        r"rental\s+income|cash\s+flow|great\s+rental|fix\s+(and|&|n)\s+flip|"
        r"buy\s+and\s+hold|cap\s+rate|income\s+producing"
    ),
}

# ── Banda ALTA (asociada a precios por encima de la mediana del barrio) ──────────
HIGH_BAND = {
    "desc_luxury_language": (
        r"luxur(y|ious)|exclusive|prestigious|gourmet|resort[\s-]?style|high[\s-]?end|"
        r"custom[\s-]?built|\bestate\b|elegant|opulent|designer|top[\s-]?of[\s-]?the[\s-]?line|"
        r"chef'?s\s+kitchen|entertainer'?s\s+dream|one[\s-]?of[\s-]?a[\s-]?kind"
    ),
    "desc_waterfront_estate": (
        r"waterfront|ocean(front|\s+view)|beachfront|lake\s*front|river\s*front|"
        r"water\s+view|deep\s+water|private\s+dock|intracoastal|canal\s+front|"
        r"gulf\s+access|boat\s+lift"
    ),
}

# ── Tono comercial del precio (sin monto) ───────────────────────────────────────
PRICE_TONE = {
    "desc_price_cut_language": (
        r"price\s+(reduc|cut|drop|improv|adjust)|reduced\b|new\s+price|priced\s+below|"
        r"bring\s+(all\s+)?offers|seller\s+financ|owner\s+financ"
    ),
    "desc_priced_to_sell": (
        r"priced\s+to\s+sell|won'?t\s+last|act\s+fast|move[\s-]?in\s+ready|turn[\s-]?key"
    ),
    "desc_below_market_language": (
        r"below\s+market|under\s+market|great\s+(deal|value|price|buy)|"
        r"best\s+(deal|value|price)|bargain|\bsteal\b|priced\s+right"
    ),
}

ALL_PATTERNS = {**LOW_BAND, **HIGH_BAND, **PRICE_TONE}

# Cues que empujan hacia banda BAJA (para el conteo denso).
CHEAP_CUE_COLS = list(LOW_BAND.keys()) + [
    "desc_price_cut_language", "desc_below_market_language"
]
# Cues que empujan hacia banda ALTA.
PREMIUM_CUE_COLS = list(HIGH_BAND.keys())

FEATURE_COLUMNS = (
    list(ALL_PATTERNS.keys())
    + ["desc_cheap_cue_count", "desc_premium_cue_count", "desc_price_band_score"]
)

_COMPILED = {name: re.compile(pat, flags=re.IGNORECASE) for name, pat in ALL_PATTERNS.items()}


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Banderas lexicas de banda de precio. Devuelve un DataFrame con el mismo index."""
    text = df.get("description")
    if text is None:
        text = pd.Series("", index=df.index)
    text = text.fillna("").astype(str)

    out = pd.DataFrame(index=df.index)
    with warnings.catch_warnings():
        # Los patrones usan grupos () para alternancias; str.contains solo mira si matchea.
        warnings.simplefilter("ignore", UserWarning)
        for name, rx in _COMPILED.items():
            out[name] = text.str.contains(rx).astype(np.int8)

    out["desc_cheap_cue_count"] = out[CHEAP_CUE_COLS].sum(axis=1).astype(np.int16)
    out["desc_premium_cue_count"] = out[PREMIUM_CUE_COLS].sum(axis=1).astype(np.int16)
    # Score neto: >0 tira a caro, <0 tira a barato. Feature densa y monotona.
    out["desc_price_band_score"] = (
        out["desc_premium_cue_count"] - out["desc_cheap_cue_count"]
    ).astype(np.int16)
    return out[FEATURE_COLUMNS]
