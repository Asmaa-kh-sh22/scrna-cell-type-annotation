# Benchmarking Classical and Transformer-Based Models for Cell Type Annotation in Single-Cell RNA Sequencing

[![Python 3.12](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org/)
![Status](https://img.shields.io/badge/Status-Under%20Review-orange)

## Overview

This repository contains the code for the paper:

> **Benchmarking Classical and Transformer-Based Models for Cell Type Annotation in Single-Cell RNA Sequencing**
> Afrah Sulaih AL-Malki, Asmaa Khalid AL-Shumrani, Walaa Salim AL-Fahmi, Tahani M. Alsubait
> *Submitted to IEEE Access, 2025 (under review)*

We benchmark four models across three categories for automated cell type annotation in scRNA-seq data:

| Model | Category |
|---|---|
| K-Nearest Neighbors (KNN) | Classical ML |
| Random Forest (RF) | Classical ML |
| Self-Supervised Transformer | Deep Learning |
| Transformer + SVM (Hybrid) | Hybrid |

## Dataset

We use the **Tabula Sapiens pancreas dataset** (`ts_pancreas.h5ad`), available from the [Tabula Sapiens Consortium](https://tabula-sapiens-portal.ds.czbiohub.org/).

- 14,140 cells × 61,759 genes
- 20 cell types with pronounced class imbalance

> The dataset file is not included in this repository due to its size. Download it and place it at:
> `data/ts_pancreas.h5ad`

## Results

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|---|---|---|---|---|
| KNN (cosine, k=15) | 0.954 | 0.714 | 0.666 | 0.673 |
| Random Forest (400 trees) | 0.967 | 0.676 | 0.674 | 0.673 |
| Transformer (self-supervised) | 0.915 | 0.649 | 0.769 | 0.675 |
| **Transformer + SVM (Hybrid)** | **0.9455** | **0.7020** | **0.7209** | **0.7029** |

## Repository Structure

```
scrna-cell-type-annotation/
│
├── README.md
├── requirements.txt
├── notebook/
│   └── cell_type_annotation.ipynb    # Full Colab notebook (recommended)
└── src/
    ├── 01_load_explore.py            # Data loading and exploration
    ├── 02_preprocessing.py           # Filtering, normalization, HVG selection
    ├── 03_knn_rf_cv.py               # KNN and Random Forest with 5-fold CV
    ├── 04_transformer.py             # Self-supervised Transformer pretraining
    └── 05_hybrid_transformer_svm.py  # Hybrid Transformer + SVM model
```

## How to Run

### Option 1: Google Colab (Recommended)
Open `notebook/cell_type_annotation.ipynb` directly in Google Colab with a T4 GPU runtime.

### Option 2: Local
```bash
pip install -r requirements.txt
python src/01_load_explore.py
python src/02_preprocessing.py
python src/03_knn_rf_cv.py
python src/04_transformer.py
python src/05_hybrid_transformer_svm.py
```

## Environment

- Platform: Google Colab (NVIDIA T4 GPU)
- Python 3.12

## Citation

This paper is currently under review. Once published, please cite as:

```bibtex
@article{almalki2025benchmarking,
  title={Benchmarking Classical and Transformer-Based Models for Cell Type Annotation in Single-Cell RNA Sequencing},
  author={AL-Malki, Afrah Sulaih and AL-Shumrani, Asmaa Khalid and AL-Fahmi, Walaa Salim and Alsubait, Tahani M.},
  journal={IEEE Access},
  year={2025},
  note={Under review}
}
```

## Acknowledgment

The authors acknowledge the use of ChatGPT (OpenAI) and QuillBot for language editing and grammatical refinement of portions of the manuscript. All scientific content, experimental design, analysis, results, and conclusions were conceived, conducted, and written by the authors.
