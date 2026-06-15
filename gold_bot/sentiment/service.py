"""Sentiment scoring service: a single entry point that picks between the
lexicon fallback and the optional FinBERT model.

Per the research guide, the sentiment overlay should *agree* with the
technical/ML signal direction to size up a trade - it is a filter, not a
standalone signal generator (see gold_bot.strategy.ml_overlay).
"""
from __future__ import annotations

from gold_bot.sentiment import lexicon


def get_sentiment_score(headlines: list[str], method: str = "lexicon") -> float:
    """Return a sentiment score in [-1, 1] for a list of news headlines.

    method: "lexicon" (default, dependency-free) or "finbert" (requires
    transformers+torch, downloads model weights on first use).
    """
    if method == "lexicon":
        return lexicon.score_headlines(headlines)
    if method == "finbert":
        from gold_bot.sentiment import finbert
        return finbert.score_headlines(headlines)
    raise ValueError(f"Unknown sentiment method: {method!r}")


def sentiment_agrees(score: float, direction: int, threshold: float = 0.1) -> bool:
    """Return True if the sentiment score agrees with (or is neutral toward)
    the proposed trade `direction` (1 = long, -1 = short).

    A score with magnitude below `threshold` is treated as neutral and
    counts as agreement (sentiment doesn't veto on weak/no signal).
    """
    if abs(score) < threshold:
        return True
    return (score > 0 and direction == 1) or (score < 0 and direction == -1)
