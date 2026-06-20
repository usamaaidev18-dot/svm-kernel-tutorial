# Choosing the Right Kernel: How SVM Decision Boundaries Change with Data Geometry

> A practical deep dive into Linear, RBF, and Polynomial kernels — visualising decision boundaries, understanding C and gamma sensitivity, and knowing when each kernel fails. Applied to the UCI Steel Plates Faults dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange?logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Steel%20Plates%20Faults-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [Dataset](#dataset)
- [Tutorial Contents](#tutorial-contents)
- [Figures](#figures)
- [Accessibility](#accessibility)
- [Ethical Considerations](#ethical-considerations)
- [References](#references)
- [Licence](#licence)

---

## Overview

Support Vector Machines (SVMs) are powerful classifiers — but the choice of **kernel function** fundamentally changes what the model learns. This tutorial investigates how three commonly used kernels (Linear, RBF, and Polynomial) produce different decision boundaries on the same real-world dataset, and what drives those differences.

**The central question:**
> *Given a real industrial dataset with overlapping, imbalanced classes, how does kernel choice change what the SVM learns — and when does each kernel fail?*

**Why the Steel Plates Faults dataset?**
With 27 geometric features, 7 fault types, severe class imbalance, and heavy feature overlap, this dataset is a genuinely challenging classification problem from a stainless steel manufacturing plant. It is far more representative of real-world conditions than the toy datasets typically used to demonstrate SVMs — and, to the best of our knowledge, has not previously been used as the basis for a kernel-comparison tutorial of this kind.

**What you will learn:**
- How Linear, RBF, and Polynomial kernels create fundamentally different decision boundaries
- How the `C` and `gamma` hyperparameters reshape those boundaries
- Why some kernels fail on certain data geometries
- A practical, evidence-based framework for kernel selection in your own work

---

## Key Results

| Kernel | Test Accuracy | CV Accuracy (5-fold) | Macro F1 |
|---|---|---|---|
| Linear (default, C=1) | 71.0% | 73.4% | 0.678 |
| RBF (default, C=1, γ=scale) | 76.4% | 74.0% | 0.759 |
| Polynomial (default, C=1, deg=3) | 70.4% | 70.7% | 0.677 |
| **Best (GridSearchCV: RBF, C=1, γ=0.1)** | **77.6%** | **75.3%** | — |

**Key finding:** The RBF kernel consistently outperforms Linear and Polynomial on this dataset. The performance gap is most pronounced on minority classes — RBF achieves F1=0.57 on the rare `Dirtiness` class versus Linear's F1=0.22, demonstrating that kernel choice has a disproportionate effect on underrepresented classes.

---

## Repository Structure

```
svm-kernel-tutorial/
│
├── svm_kernel_tutorial.ipynb     # Main Jupyter notebook (fully executable)
├── faults.csv                    # UCI Steel Plates Faults dataset (real data)
├── svm_kernel_tutorial.pdf       # PDF tutorial report (<2000 words)
│
├── figures/                      # All 12 generated figures (PNG, 150 DPI)
│   ├── fig1_class_distribution.png
│   ├── fig2_feature_scaling.png
│   ├── fig3_pca_projection.png
│   ├── fig4_synthetic_kernels.png
│   ├── fig5_decision_boundaries.png
│   ├── fig6_f1_by_class.png
│   ├── fig7_c_sensitivity.png
│   ├── fig8_rbf_heatmap.png
│   ├── fig9_poly_degree_heatmap.png
│   ├── fig10_failure_modes.png
│   ├── fig11_confusion_matrix.png
│   └── fig12_summary_comparison.png
│
├── requirements.txt              # Python dependencies with pinned versions
├── LICENSE                       # MIT Licence
└── README.md                     # This file
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional but recommended) a virtual environment tool: `venv` or `conda`

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/usamaaidev18-dot/svm-kernel-tutorial.git
cd svm-kernel-tutorial
```

**2. Create and activate a virtual environment** *(recommended)*

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# OR using conda
conda create -n svm-tutorial python=3.10
conda activate svm-tutorial
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook svm_kernel_tutorial.ipynb
```

Or, if you prefer JupyterLab:

```bash
pip install jupyterlab
jupyter lab svm_kernel_tutorial.ipynb
```

**The notebook is fully self-contained.** All figures are generated in-place. Expected total runtime on a standard laptop: **3–8 minutes** (the GridSearchCV cell is the longest step, approximately 2–4 minutes).

> **Dataset loading:** The notebook first attempts to fetch the dataset automatically via `ucimlrepo` (requires internet). If unavailable, it falls back to the `faults.csv` file included in this repository. Either path produces identical results.

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | Steel Plates Faults |
| **Source** | UCI Machine Learning Repository |
| **Also available** | [Kaggle — Faulty Steel Plates](https://www.kaggle.com/datasets/uciml/faulty-steel-plates) |
| **Creators** | Buscema, M., Terzi, S., & Tastle, W. (Semeion Research Center, Italy) |
| **Samples** | 1,941 |
| **Features** | 27 (geometric shape descriptors: area, perimeter, luminosity, edge indices, orientation) |
| **Classes** | 7 fault types |
| **Licence** | CC BY 4.0 |
| **Citation** | Buscema et al. (2010). Steel Plates Faults [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5J88N |

The `faults.csv` file included in this repository is the unmodified dataset downloaded directly from the UCI repository via Kaggle. No rows have been removed, no features engineered, and no labels altered.

---

## Tutorial Contents

The notebook is structured as a complete, self-contained tutorial across 12 sections:

| Section | Description |
|---|---|
| **1. Environment Setup** | Imports, random seeds, colourblind-safe palette definition |
| **2. Dataset Loading** | `ucimlrepo` primary loader with local CSV fallback; one-hot → integer label conversion |
| **3. Exploratory Data Analysis** | Class distribution, feature scaling motivation, PCA 2D projection |
| **4. SVM Theory** | Margin maximisation, kernel trick, comparison table of all three kernels |
| **5. Preprocessing** | Stratified train/test split, StandardScaler, Pipeline construction |
| **6. Kernel Comparison (Baseline)** | Decision boundaries on PCA projection; per-class F1 scores |
| **7. C Sensitivity** | Cross-validated accuracy sweep across C for all three kernels |
| **8. RBF C × γ Heatmap** | Joint hyperparameter sensitivity analysis for the RBF kernel |
| **9. Polynomial Degree Sensitivity** | Effect of polynomial degree on accuracy across C values |
| **10. Failure Mode Demonstration** | Deliberate under- and overfitting with bad hyperparameters |
| **11. GridSearchCV** | Full pipeline search across all kernels; best model confusion matrix |
| **12. Summary & Practical Guide** | Performance comparison; kernel selection decision framework |

---

## Figures

All figures use the **Okabe-Ito colourblind-safe palette** and are saved at 150 DPI. A summary of what each figure demonstrates:

| Figure | What It Shows |
|---|---|
| Fig 1 — Class Distribution | Severe imbalance: Other_Faults (34.7%) vs Dirtiness (2.8%) |
| Fig 2 — Feature Scaling | Raw features span 10⁷ range; standardisation collapses to unit variance |
| Fig 3 — PCA Projection | Heavy class overlap in 2D confirms non-linear kernel is needed |
| Fig 4 — Synthetic Kernels | Linear 58% vs RBF 100% on circular data — intuition builder |
| Fig 5 — Decision Boundaries | RBF carves curved K_Scatch region; Linear draws straight cuts |
| Fig 6 — Per-Class F1 | RBF macro-F1=0.759 vs Linear=0.678; Dirtiness F1 doubles with RBF |
| Fig 7 — C Sensitivity | All kernels collapse at C=0.001; RBF most stable across C range |
| Fig 8 — RBF Heatmap | Best RBF: C=1, γ=0.1 (CV=0.754); high γ always harmful |
| Fig 9 — Poly Degree Heatmap | Degree 2 outperforms higher degrees; complexity adds no benefit |
| Fig 10 — Failure Modes | RBF overfitting: 40% train-test gap at γ=10 |
| Fig 11 — Confusion Matrix | K_Scatch near-perfect; Bumps↔Other_Faults is main confusion |
| Fig 12 — Summary Chart | RBF (77.6%) > Linear (71.0%) > Polynomial (70.4%) |

---

## Accessibility

This tutorial was designed with accessibility in mind throughout:

- **Colourblind-safe palette:** All figures use the Okabe-Ito palette, which is distinguishable under the three most common forms of colour vision deficiency (deuteranopia, protanopia, tritanopia).
- **Text alternatives:** Every figure is accompanied by a detailed caption in both the notebook (markdown) and the PDF report.
- **Screen reader compatibility:** The PDF report uses standard heading structure and descriptive alt text conventions. The notebook uses semantic markdown headings (H1/H2) throughout.
- **No colour-only encoding:** All figures use both colour and shape/pattern/label to convey information, so no information is lost if printed in greyscale.
- **Font sizes:** All axis labels, tick marks, and annotations meet a minimum 7.5pt equivalent size at 150 DPI output.

---

## Ethical Considerations

Several ethical considerations are relevant to this work:

**Class imbalance and fairness:** The dataset has a 12:1 ratio between the largest and smallest classes. Optimising for overall accuracy alone would systematically ignore rare fault types. In a real deployment, every missed `Dirtiness` fault reaching a customer represents a product quality failure. Users adapting this work should use `class_weight='balanced'` and evaluate per-class recall, not aggregate accuracy.

**Model explainability:** RBF SVMs are not interpretable. In safety-critical manufacturing deployments, predictions should be accompanied by post-hoc explanations (e.g. SHAP values) or supplemented by an interpretable model.

**Dataset licence:** The UCI Steel Plates Faults dataset is published under CC BY 4.0, permitting academic and commercial use with attribution. The dataset contains no personally identifiable information.

**Distribution shift:** A model trained on historical fault data may degrade silently if the manufacturing process changes. Any production deployment requires ongoing monitoring and scheduled retraining.

**Computational cost:** GridSearchCV trains 180 models. For larger datasets, consider random search or Bayesian optimisation as more energy-efficient alternatives.

---

## References

1. Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5J88N
2. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
3. Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
4. Scikit-learn developers (2024). sklearn.svm.SVC documentation. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
5. Hsu, C.-W., Chang, C.-C., & Lin, C.-J. (2016). A Practical Guide to Support Vector Classification. https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
6. Müller, A. C., & Guido, S. (2016). *Introduction to Machine Learning with Python*. O'Reilly Media.
7. Scikit-learn: Plot classification boundaries with SVM Kernels. https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html
8. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.

---

## Licence

This project is licensed under the **MIT Licence** — see the [LICENSE](LICENSE) file for details.

The dataset (`faults.csv`) is sourced from the UCI ML Repository and is licensed separately under **CC BY 4.0**. Attribution: Buscema et al. (2010). https://doi.org/10.24432/C5J88N

---

*Tutorial submitted as part of an Advanced Machine Learning course assignment.*  
*All code, figures, and report text are original work. References are cited throughout the notebook and PDF report.*
