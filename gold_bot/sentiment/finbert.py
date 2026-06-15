"""FinBERT-based news sentiment scorer (optional, heavy dependency).

Requires the `transformers` and `torch` packages and downloads the
"ProsusAI/finbert" model weights on first use - both are optional and not
installed by default (see requirements.txt). Falls back gracefully with a
clear error if unavailable, so the rest of the pipeline (which uses
gold_bot.sentiment.lexicon by default) still works offline.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_pipeline():
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "FinBERT sentiment requires the optional 'transformers' and "
            "'torch' packages. Install them with: pip install transformers torch"
        ) from e
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


_LABEL_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def score_text(text: str) -> float:
    """Return a signed sentiment score in [-1, 1] for a single headline/text."""
    clf = _get_pipeline()
    result = clf(text, truncation=True)[0]
    return _LABEL_SIGN.get(result["label"].lower(), 0.0) * result["score"]


def score_headlines(headlines: list[str]) -> float:
    """Return the mean FinBERT sentiment score across a list of headlines, in [-1, 1]."""
    if not headlines:
        return 0.0
    clf = _get_pipeline()
    results = clf(headlines, truncation=True)
    scores = [_LABEL_SIGN.get(r["label"].lower(), 0.0) * r["score"] for r in results]
    return sum(scores) / len(scores)
