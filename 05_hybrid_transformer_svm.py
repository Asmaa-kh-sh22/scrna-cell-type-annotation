# =============================================================================
# 05_hybrid_transformer_svm.py
# Hybrid model: frozen Transformer embeddings + Linear SVM classifier
# with 5-fold stratified CV and final comparison plot
# Assumes transformer_pipe and TransformerEmbedder from 04_transformer.py
# =============================================================================

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

# ── Hybrid model definition ───────────────────────────────────────────────────
transformer_svm_pipe = Pipeline([
    ("embed", TransformerEmbedder(batch_size=512)),
    ("svm",   LinearSVC(class_weight="balanced", max_iter=5000))
])

# ── 5-Fold CV ─────────────────────────────────────────────────────────────────
print("Running 5-Fold CV (Transformer + SVM)...")
svm_scores = cross_validate(transformer_svm_pipe, X_dense, y,
                            cv=skf, scoring=scoring, n_jobs=2)

print("\n=== Hybrid Model (Transformer + SVM) — 5-Fold CV ===")
print(f"Accuracy        : {svm_scores['test_accuracy'].mean():.4f}")
print(f"F1-macro        : {svm_scores['test_f1_macro'].mean():.4f}")
print(f"Precision-macro : {svm_scores['test_precision_macro'].mean():.4f}")
print(f"Recall-macro    : {svm_scores['test_recall_macro'].mean():.4f}")

# ── Per-class report and confusion matrix ─────────────────────────────────────
y_true    = np.asarray(y).ravel().astype(str)
y_pred_hyb = cross_val_predict(transformer_svm_pipe, X_dense, y_true, cv=skf, n_jobs=2)

print("\n=== Per-class Report (Transformer + SVM) ===")
classes = sorted(np.unique(y_true))
print(classification_report(y_true, y_pred_hyb, target_names=classes, digits=3))

cm_hyb = confusion_matrix(y_true, y_pred_hyb, labels=classes)
plt.figure(figsize=(max(8, 0.4*len(classes)), max(6, 0.35*len(classes))))
sns.heatmap(cm_hyb, annot=False, cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix — Transformer + SVM", fontsize=12, weight="bold")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(rotation=75, fontsize=8); plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("confusion_matrix_transformer_svm.png", dpi=400, bbox_inches="tight")
plt.show()

# ── Side-by-side confusion matrices (RF, Transformer, Hybrid) ─────────────────
from sklearn.model_selection import cross_val_predict as cvp

y_pred_rf  = cvp(rf_pipe,          X,       y_true, cv=skf, n_jobs=2)
y_pred_tr  = cvp(transformer_pipe, X_dense, y_true, cv=skf, n_jobs=2)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
(ax_rf, ax_tr), (ax_empty, ax_hyb) = axes

for ax, preds, title in [
    (ax_rf,  y_pred_rf,  "Random Forest"),
    (ax_tr,  y_pred_tr,  "Transformer"),
    (ax_hyb, y_pred_hyb, "Transformer + SVM (Hybrid)")
]:
    cm = confusion_matrix(y_true, preds, labels=classes)
    sns.heatmap(cm, annot=False, cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.tick_params(axis='x', rotation=75)

ax_empty.axis("off")
plt.tight_layout()
plt.savefig("confusion_matrices_grid.png", dpi=400, bbox_inches='tight')
plt.show()

# ── Macro F1-score comparison bar chart ───────────────────────────────────────
models   = ['KNN', 'Random Forest', 'Transformer', 'Transformer + SVM']
f1_scores = [0.673, 0.673, 0.675, 0.7029]

plt.figure(figsize=(6, 4))
bars = plt.bar(models, f1_scores, color=['#4C72B0', '#4C72B0', '#4C72B0', '#DD8452'])
plt.title('Macro F1-Score Comparison', fontsize=12, weight='bold')
plt.ylabel('Macro F1-Score')
plt.ylim(0.65, 0.72)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
             f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("f1_score_comparison.png", dpi=400, bbox_inches='tight')
plt.show()
