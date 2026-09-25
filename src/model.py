"""
ML model training & evaluation module for Business Entity Resolution Challenge.
Supports threshold tuning and evaluation using F_0.5 metric.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
from typing import Tuple, Dict, Any


def calculate_f_05(precision: float, recall: float) -> float:
    """Calculate F_0.5 metric (beta=0.5, Precision weighted 2x recall)."""
    if precision + recall == 0:
        return 0.0
    beta_sq = 0.5 ** 2  # 0.25
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


class EntityResolutionModel:
    """Classifier model for predicting record matches between entity pairs."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1
        )
        self.best_threshold = 0.5

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit model on feature matrix X and binary labels y."""
        self.clf.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match probability."""
        return self.clf.predict_proba(X)[:, 1]

    def optimize_threshold(self, X_val: pd.DataFrame, y_val: np.ndarray) -> float:
        """Find optimal decision threshold maximizing F_0.5 score on validation data."""
        probas = self.predict_proba(X_val)
        best_f05 = -1.0
        best_thresh = 0.5

        for thresh in np.arange(0.1, 0.9, 0.02):
            preds = (probas >= thresh).astype(int)
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            f05 = calculate_f_05(prec, rec)
            
            if f05 > best_f05:
                best_f05 = f05
                best_thresh = thresh

        self.best_threshold = best_thresh
        print(f"Optimal Threshold: {best_thresh:.2f} with F_0.5 Score: {best_f05:.4f}")
        return best_thresh

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Predict binary match labels using specified or tuned threshold."""
        thresh = threshold if threshold is not None else self.best_threshold
        probas = self.predict_proba(X)
        return (probas >= thresh).astype(int)
