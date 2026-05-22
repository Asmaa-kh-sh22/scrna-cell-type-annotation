# =============================================================================
# 01_load_explore.py
# Load and explore the Tabula Sapiens pancreas dataset
# =============================================================================
# NOTE: Designed for Google Colab with T4 GPU runtime.
# Mount Google Drive and update file_path before running.
# =============================================================================

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Mount Google Drive (Colab only) ──────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Load dataset ─────────────────────────────────────────────────────────────
file_path = '/content/drive/MyDrive/Single-Cell Annotation Dataset/ts_pancreas.h5ad'
adata = sc.read_h5ad(file_path)
print("Dataset loaded successfully!")

# ── Basic structure ───────────────────────────────────────────────────────────
print(f"Number of cells : {adata.n_obs}")
print(f"Number of genes : {adata.n_vars}")
print("\nAvailable metadata columns:")
print(adata.obs.columns.tolist())

# ── Cell type distribution ────────────────────────────────────────────────────
print("\nUnique cell types:")
print(adata.obs['cell_type'].value_counts())

# ── Quality-control metrics ───────────────────────────────────────────────────
print("\nQuality metrics summary:")
print(adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe().T)

# ── Visualize cell type distribution ─────────────────────────────────────────
adata.obs['cell_type'].value_counts().plot(kind='bar', figsize=(15, 10))
plt.title('Cell Type Distribution in Tabula Sapiens Pancreas Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
