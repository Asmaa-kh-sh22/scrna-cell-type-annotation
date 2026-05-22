# =============================================================================
# 04_transformer.py
# Self-supervised Transformer encoder with masked gene modeling pretraining
# and linear probe (Logistic Regression) evaluation via 5-fold CV
# Assumes X, y are prepared from 02_preprocessing.py
# =============================================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch device:", device)

# ── Prepare dense float32 input ───────────────────────────────────────────────
X_dense = X if isinstance(X, np.ndarray) else X.toarray()
X_dense = X_dense.astype(np.float32)
n_cells, n_genes = X_dense.shape

ssl_scaler = StandardScaler(with_mean=True, with_std=True)
X_std = ssl_scaler.fit_transform(X_dense).astype(np.float32)

# ── Model architecture ────────────────────────────────────────────────────────
class Patchify(nn.Module):
    def __init__(self, n_genes, n_tokens):
        super().__init__()
        self.n_tokens  = n_tokens
        self.patch_size = math.ceil(n_genes / n_tokens)
        self.total      = self.patch_size * self.n_tokens

    def forward(self, x):
        B, G = x.shape
        if G < self.total:
            x = torch.nn.functional.pad(x, (0, self.total - G))
        return x.view(B, self.n_tokens, self.patch_size)


class Unpatchify(nn.Module):
    def __init__(self, n_genes, n_tokens):
        super().__init__()
        self.patch_size = math.ceil(n_genes / n_tokens)
        self.n_genes    = n_genes

    def forward(self, x_tokens):
        B, T, P = x_tokens.shape
        return x_tokens.reshape(B, T * P)[:, :self.n_genes]


class GeneTransformer(nn.Module):
    def __init__(self, n_genes, n_tokens=64, d_model=128, n_heads=4, n_layers=2, p_drop=0.1):
        super().__init__()
        self.patchify   = Patchify(n_genes, n_tokens)
        self.unpatchify = Unpatchify(n_genes, n_tokens)
        self.patch_emb  = nn.Linear(self.patchify.patch_size, d_model)
        self.pos        = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        enc_layer       = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                     batch_first=True, dropout=p_drop)
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head       = nn.Linear(d_model, self.patchify.patch_size)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x):
        patches     = self.patchify(x)
        h           = self.patch_emb(patches) + self.pos
        h           = self.encoder(h)
        rec_patches = self.head(h)
        rec         = self.unpatchify(rec_patches)
        emb         = self.norm(h.mean(dim=1))
        return rec, emb


# ── Self-supervised pretraining (masked gene modeling) ───────────────────────
mask_ratio  = 0.15
epochs      = 10
batch_size  = 256
lr          = 1e-3
weight_decay = 1e-4

model   = GeneTransformer(n_genes=n_genes).to(device)
opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.SmoothL1Loss()

idx_all = np.arange(n_cells)

def batch_iter(bs=256):
    np.random.shuffle(idx_all)
    for i in range(0, len(idx_all), bs):
        yield idx_all[i:i+bs]

print("Pretraining Transformer encoder (masked gene modeling)...")
for ep in range(1, epochs + 1):
    model.train()
    total = 0.0
    for idx in batch_iter(batch_size):
        xb   = torch.from_numpy(X_std[idx]).to(device)
        mask = (torch.rand_like(xb) < mask_ratio).float()
        rec, _ = model(xb * (1 - mask))
        loss = loss_fn(rec * mask, xb * mask)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    print(f"  epoch {ep:02d}/{epochs} | loss={total:.4f}")

torch.save(model.state_dict(), "/content/transformer_encoder.pt")
print("Saved: /content/transformer_encoder.pt")

# ── Frozen embedder for sklearn ───────────────────────────────────────────────
_frozen = GeneTransformer(n_genes=n_genes, p_drop=0.0).to(device)
_frozen.load_state_dict(torch.load("/content/transformer_encoder.pt", map_location=device))
_frozen.eval()
for p in _frozen.parameters():
    p.requires_grad = False


class TransformerEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=512):
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = (X if isinstance(X, np.ndarray) else X.toarray()).astype(np.float32)
        X_arr = ssl_scaler.transform(X_arr).astype(np.float32)
        embs  = []
        with torch.no_grad():
            for i in range(0, X_arr.shape[0], self.batch_size):
                xb = torch.from_numpy(X_arr[i:i+self.batch_size]).to(device)
                _, e = _frozen(xb)
                embs.append(e.cpu().numpy())
        return np.concatenate(embs, axis=0)


# ── 5-Fold CV with linear probe ───────────────────────────────────────────────
skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

transformer_pipe = Pipeline([
    ("embed", TransformerEmbedder(batch_size=512)),
    ("clf",   LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1))
])

print("\nRunning 5-Fold CV (Transformer linear probe)...")
tr_scores = cross_validate(transformer_pipe, X_dense, y, cv=skf,
                           scoring=scoring, n_jobs=2)

print("\n=== Transformer (self-pretrained, linear probe) — 5-Fold CV ===")
print(f"Accuracy        : {tr_scores['test_accuracy'].mean():.4f}")
print(f"F1-macro        : {tr_scores['test_f1_macro'].mean():.4f}")
print(f"Precision-macro : {tr_scores['test_precision_macro'].mean():.4f}")
print(f"Recall-macro    : {tr_scores['test_recall_macro'].mean():.4f}")

# ── Per-class report and confusion matrix ─────────────────────────────────────
y_pred_tr = cross_val_predict(transformer_pipe, X_dense, y, cv=skf, n_jobs=2)
print("\n=== Per-class Report (Transformer) ===")
print(classification_report(y, y_pred_tr, zero_division=0))

labels = sorted(list(set(y)))
cm = confusion_matrix(y, y_pred_tr, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, fmt="d", cmap="coolwarm",
            xticklabels=labels, yticklabels=labels)
plt.title("Transformer (self-pretrained) — Confusion Matrix (5-Fold CV)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.show()
