# =============================================================================
# 02_preprocessing.py
# Preprocessing: filtering, normalization, scaling, and HVG selection
# Assumes adata is already loaded from 01_load_explore.py
# =============================================================================

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 1. Quality filtering ──────────────────────────────────────────────────────
print("Filtering cells and genes...")
sc.pp.filter_cells(adata, min_genes=200)   # Remove cells with too few genes
sc.pp.filter_genes(adata, min_cells=3)     # Remove genes expressed in too few cells

# ── 2. Normalization ──────────────────────────────────────────────────────────
print("Normalizing...")
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize to 10,000 reads per cell
sc.pp.log1p(adata)                             # Log-transform: log(1 + x)

# ── 3. Scaling ────────────────────────────────────────────────────────────────
print("Scaling...")
sc.pp.scale(adata, max_value=10)               # Z-score scaling, clip at 10

# ── 4. Feature selection (HVGs) ──────────────────────────────────────────────
print("Selecting highly variable genes (HVGs)...")
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]

# ── 5. Prepare features and labels ───────────────────────────────────────────
X = adata.X.toarray() if not isinstance(adata.X, __import__('numpy').ndarray) else adata.X
y = adata.obs['cell_type'].values

print(f"\nFinal dataset shape : {X.shape}")
print(f"Number of cell types: {len(set(y))}")

# ── 6. Visualize split distribution ──────────────────────────────────────────
import pandas as pd
from pathlib import Path

counts = adata.obs['cell_type'].value_counts().sort_index()
temp   = (counts * 0.40).round().astype(int)
test   = (counts - temp).astype(int)
train  = (temp * 0.50).round().astype(int)
val    = (temp - train).astype(int)

tbl = pd.DataFrame({
    "Total": counts,
    "Test (60%)": test,
    "Train (20%)": train,
    "Val (20%)": val
}).astype(int)
tbl["Min across splits"]       = tbl[["Train (20%)", "Val (20%)", "Test (60%)"]].min(axis=1)
tbl["Violates stratify (min<2)"] = tbl["Min across splits"] < 2

print("\nPer-class counts and violations:")
print(tbl.sort_values('Total', ascending=False).head(20))
print("\nNote: Classes with <2 samples per split require Cross-Validation instead of a fixed split.")
