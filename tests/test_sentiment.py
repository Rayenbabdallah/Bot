from gold_bot.sentiment.lexicon import score_headlines, score_text
from gold_bot.sentiment.service import get_sentiment_score, sentiment_agrees


def test_score_text_positive_and_negative():
    assert score_text("Gold rallies as demand surges") > 0
    assert score_text("Gold plunges amid selloff and recession fears") < 0
    assert score_text("The cat sat on the mat") == 0.0


def test_score_headlines_empty_is_neutral():
    assert score_headlines([]) == 0.0


def test_get_sentiment_score_lexicon():
    score = get_sentiment_score(["Gold rallies on safe-haven demand"], method="lexicon")
    assert score > 0


def test_sentiment_agrees():
    assert sentiment_agrees(0.5, 1) is True
    assert sentiment_agrees(0.5, -1) is False
    assert sentiment_agrees(-0.5, -1) is True
    assert sentiment_agrees(0.05, -1, threshold=0.1) is True  # below threshold -> neutral
