# =============================================================================
# 03_knn_rf_cv.py
# KNN and Random Forest with 5-fold stratified cross-validation
# Assumes X, y are prepared from 02_preprocessing.py
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ── Merge rare classes (<5 samples) ──────────────────────────────────────────
counts = pd.Series(y).value_counts()
rare   = counts[counts < 5].index
if len(rare) > 0:
    y = np.where(np.isin(y, rare), 'Rare_Other', y)
    print(f"Merged {len(rare)} rare classes → 'Rare_Other'")

# ── Cross-validation setup ────────────────────────────────────────────────────
skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

# ── Model definitions ─────────────────────────────────────────────────────────
knn_pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("clf",    KNeighborsClassifier(n_neighbors=15, weights='distance', metric='cosine'))
])

rf_pipe = Pipeline([
    ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                   random_state=42, n_jobs=-1))
])

# ── Run CV ────────────────────────────────────────────────────────────────────
print("Running 5-Fold CV for KNN and Random Forest...")
knn_scores = cross_validate(knn_pipe, X, y, cv=skf, scoring=scoring, n_jobs=2)
rf_scores  = cross_validate(rf_pipe,  X, y, cv=skf, scoring=scoring, n_jobs=2)

def summarize(name, s):
    print(f"\n=== {name} (5-Fold CV) ===")
    print(f"Accuracy        : {s['test_accuracy'].mean():.4f}")
    print(f"F1-macro        : {s['test_f1_macro'].mean():.4f}")
    print(f"Precision-macro : {s['test_precision_macro'].mean():.4f}")
    print(f"Recall-macro    : {s['test_recall_macro'].mean():.4f}")

summarize("KNN (cosine, distance-weighted, k=15)", knn_scores)
summarize("Random Forest (balanced, 400 trees)",   rf_scores)

# ── Per-class report and confusion matrix ─────────────────────────────────────
def evaluate_pretty(name, pipe):
    print(f"\n=== {name} ===")
    y_pred = cross_val_predict(pipe, X, y, cv=skf, n_jobs=2)
    print(classification_report(y, y_pred, zero_division=0))

    labels = sorted(list(set(y)))
    cm = confusion_matrix(y, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt="d", cmap="coolwarm",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} - Confusion Matrix (5-Fold CV)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

evaluate_pretty("KNN (cosine, distance-weighted, k=15)", knn_pipe)
evaluate_pretty("Random Forest (balanced, 400 trees)",   rf_pipe)
