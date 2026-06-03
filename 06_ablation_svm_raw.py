# =============================================================================
# 06_ablation_svm_raw.py
# Ablation baseline: Linear SVM applied directly to standardized HVG features
# without any Transformer pretraining.
# Used to isolate the contribution of Transformer-based representations
# from the benefit of the SVM classifier alone.
# Assumes X, y are prepared from 02_preprocessing.py
# =============================================================================

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import wilcoxon

skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

# ── Ablation model definition ─────────────────────────────────────────────────
svm_raw_pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("svm",    LinearSVC(class_weight="balanced", max_iter=5000))
])

# ── 5-Fold CV ─────────────────────────────────────────────────────────────────
print("Running 5-Fold CV (SVM on raw HVGs — ablation baseline)...")
svm_raw_scores = cross_validate(svm_raw_pipe, X, y, cv=skf,
                                scoring=scoring, n_jobs=2,
                                return_train_score=False)

print("\n=== SVM on raw HVGs (Ablation Baseline) — 5-Fold CV ===")
print(f"Accuracy        : {svm_raw_scores['test_accuracy'].mean():.4f} ± {svm_raw_scores['test_accuracy'].std():.4f}")
print(f"F1-macro        : {svm_raw_scores['test_f1_macro'].mean():.4f} ± {svm_raw_scores['test_f1_macro'].std():.4f}")
print(f"Precision-macro : {svm_raw_scores['test_precision_macro'].mean():.4f} ± {svm_raw_scores['test_precision_macro'].std():.4f}")
print(f"Recall-macro    : {svm_raw_scores['test_recall_macro'].mean():.4f} ± {svm_raw_scores['test_recall_macro'].std():.4f}")

# ── Per-class report and confusion matrix ─────────────────────────────────────
y_true     = np.asarray(y).ravel().astype(str)
y_pred_svm = cross_val_predict(svm_raw_pipe, X, y_true, cv=skf, n_jobs=2)

print("\n=== Per-class Report (SVM on raw HVGs) ===")
classes = sorted(np.unique(y_true))
print(classification_report(y_true, y_pred_svm, target_names=classes, digits=3))

cm_svm = confusion_matrix(y_true, y_pred_svm, labels=classes)
plt.figure(figsize=(max(8, 0.4*len(classes)), max(6, 0.35*len(classes))))
sns.heatmap(cm_svm, annot=False, cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix — SVM on raw HVGs (Ablation)", fontsize=12, weight="bold")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(rotation=75, fontsize=8); plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("confusion_matrix_svm_ablation.png", dpi=400, bbox_inches="tight")
plt.show()

# ── Standard deviations for all models ───────────────────────────────────────
# Requires knn_scores, rf_scores, tr_scores, svm_scores from previous scripts
print("\n=== Standard Deviations — All Models ===")
for name, scores in [
    ("KNN",               knn_scores),
    ("Random Forest",     rf_scores),
    ("Transformer",       tr_scores),
    ("Transformer+SVM",   svm_scores),
    ("SVM on raw HVGs",   svm_raw_scores)
]:
    print(f"\n--- {name} ---")
    print(f"Accuracy        : {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
    print(f"F1-macro        : {scores['test_f1_macro'].mean():.4f} ± {scores['test_f1_macro'].std():.4f}")
    print(f"Precision-macro : {scores['test_precision_macro'].mean():.4f} ± {scores['test_precision_macro'].std():.4f}")
    print(f"Recall-macro    : {scores['test_recall_macro'].mean():.4f} ± {scores['test_recall_macro'].std():.4f}")

# ── Wilcoxon signed-rank test ─────────────────────────────────────────────────
# Tests whether hybrid model improvements are statistically significant
print("\n=== Wilcoxon Signed-Rank Test (Hybrid vs Baselines) ===")
hybrid_f1 = svm_scores['test_f1_macro']
for name, scores in [
    ("KNN",             knn_scores),
    ("Random Forest",   rf_scores),
    ("Transformer",     tr_scores)
]:
    stat, p = wilcoxon(hybrid_f1, scores['test_f1_macro'])
    print(f"Hybrid vs {name}: W={stat:.4f}, p={p:.4f}")

print("\nNote: With 5-fold CV, p=0.0625 is the minimum achievable p-value.")
print("W=0.000 indicates the hybrid model won all 5 folds against that baseline.")
