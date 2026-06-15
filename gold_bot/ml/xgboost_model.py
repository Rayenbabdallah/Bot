"""XGBoost directional classifier for the ML overlay.

Predicts P(up) / P(down) / P(flat) for the next `horizon` bars from the
technical+regime feature set in `gold_bot.features`. Intended to be used as
a *filter*: a technical entry signal is only sized up (or taken at all) if
this model agrees with its direction (see gold_bot.strategy.ml_overlay).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


@dataclass
class XGBDirectionalModel:
    model: XGBClassifier
    feature_columns: list[str]
    classes_: np.ndarray

    @classmethod
    def train(cls, X: pd.DataFrame, y: pd.Series, **xgb_params) -> "XGBDirectionalModel":
        params = dict(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
        params.update(xgb_params)

        # XGBoost multi-class needs labels in {0, 1, 2}; map {-1, 0, 1} -> {0, 1, 2}.
        y_mapped = y.map({-1: 0, 0: 1, 1: 2})

        model = XGBClassifier(**params)
        model.fit(X, y_mapped)
        return cls(model=model, feature_columns=list(X.columns), classes_=np.array([-1, 0, 1]))

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with columns [-1, 0, 1] = P(down), P(flat), P(up)."""
        proba = self.model.predict_proba(X[self.feature_columns])
        return pd.DataFrame(proba, index=X.index, columns=self.classes_.tolist())[[-1, 0, 1]]

    def predict_direction(self, X: pd.DataFrame, min_confidence: float = 0.0) -> pd.Series:
        """Return predicted direction in {-1, 0, 1}.

        If the top class's probability is below `min_confidence`, returns 0
        (no-edge / abstain) regardless of the argmax class.
        """
        proba = self.predict_proba(X)
        direction = proba.idxmax(axis=1)
        confidence = proba.max(axis=1)
        direction[confidence < min_confidence] = 0
        return direction

    def feature_importances(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.feature_columns) \
            .sort_values(ascending=False)
