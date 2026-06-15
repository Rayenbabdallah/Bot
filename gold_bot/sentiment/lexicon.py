"""Lightweight lexicon-based news sentiment scorer.

This is a dependency-free fallback for the FinBERT sentiment service
(gold_bot.sentiment.finbert). It is intentionally simple - a small
finance-oriented word list - and should be treated as a placeholder /
offline-testable stand-in, not a production sentiment model.
"""
from __future__ import annotations

import re

POSITIVE_WORDS = {
    "surge", "rally", "gain", "gains", "rise", "rises", "rising", "bullish",
    "growth", "strong", "strength", "beat", "beats", "upbeat", "optimism",
    "haven", "safe-haven", "demand", "record", "rebound", "recovery", "soar",
    "soars", "boost", "boosts", "outperform", "upgrade",
}

NEGATIVE_WORDS = {
    "fall", "falls", "falling", "drop", "drops", "decline", "declines",
    "bearish", "weak", "weakness", "miss", "misses", "slump", "slumps",
    "plunge", "plunges", "recession", "selloff", "sell-off", "downgrade",
    "tighten", "tightening", "hawkish", "concern", "concerns", "risk-off",
    "crash", "losses", "loss",
}

_WORD_RE = re.compile(r"[a-z'-]+")


def score_text(text: str) -> float:
    """Return a sentiment score in [-1, 1] for a single piece of text.

    Score = (positive_hits - negative_hits) / total_words_matched, or 0.0
    if no lexicon words are found.
    """
    words = _WORD_RE.findall(text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def score_headlines(headlines: list[str]) -> float:
    """Return the mean sentiment score across a list of headlines, in [-1, 1].

    Returns 0.0 (neutral) for an empty list.
    """
    if not headlines:
        return 0.0
    scores = [score_text(h) for h in headlines]
    return sum(scores) / len(scores)
