# Machine Learning from Scratch

> **Machine Learning from Scratch with NumPy** — a beginner-friendly notebook series implementing every algorithm from first principles, inspired by Andrew Ng's lectures and assignments.

---

## Repository Structure

```
ML-from-Scratch/
│
├── supervised_learning/
│   ├── regression/
│   │   ├── 01_Linear_Regression.ipynb
│   │   ├── 02_Polynomial_Regression.ipynb
│   │   └── 03_Ridge_Regression.ipynb
│   │
│   ├── classification/
│   │   ├── 01_Logistic_Regression.ipynb
│   │   ├── 02_KNN.ipynb
│   │   ├── 03_Gaussian_Naive_Bayes.ipynb
│   │   └── 04_Decision_Tree.ipynb
│   │
│   └── ensemble_methods/
│       └── 01_Random_Forest.ipynb
│
├── unsupervised_learning/
│   ├── clustering/
│   │   ├── 01_KMeans.ipynb
│   │   └── 02_DBSCAN.ipynb
│   │
│   └── dimensionality_reduction/
│       └── 01_PCA.ipynb
│
├── README.md
└── requirements.txt
```

---

## Project Goals

| Goal | Detail |
|------|--------|
| **Learning-first** | Every notebook prioritises intuition and readable code over performance |
| **Zero external ML** | No Scikit-Learn, SciPy, TensorFlow, PyTorch — only NumPy + Matplotlib |
| **Mathematically grounded** | All algorithms explained with LaTeX formulas and variable glossaries |
| **Self-contained** | Each notebook generates its own synthetic dataset with a fixed random seed |
| **Andrew Ng style** | Short cells, explicit loops, named variables (X, y, w, b, alpha, J_history) |

---

## Algorithm Index

### Supervised Learning — Regression

| # | Notebook | Key Concepts |
|---|----------|--------------|
| 1 | [01_Linear_Regression.ipynb](supervised_learning/regression/01_Linear_Regression.ipynb) | Gradient descent, MSE cost, R² |
| 2 | [02_Polynomial_Regression.ipynb](supervised_learning/regression/02_Polynomial_Regression.ipynb) | Feature expansion, bias-variance, overfitting |
| 3 | [03_Ridge_Regression.ipynb](supervised_learning/regression/03_Ridge_Regression.ipynb) | L2 regularisation, weight decay, λ tuning |

### Supervised Learning — Classification

| # | Notebook | Key Concepts |
|---|----------|--------------|
| 1 | [01_Logistic_Regression.ipynb](supervised_learning/classification/01_Logistic_Regression.ipynb) | Sigmoid, binary cross-entropy, decision boundary |
| 2 | [02_KNN.ipynb](supervised_learning/classification/02_KNN.ipynb) | Euclidean distance, majority vote, k-effect |
| 3 | [03_Gaussian_Naive_Bayes.ipynb](supervised_learning/classification/03_Gaussian_Naive_Bayes.ipynb) | Bayes theorem, Gaussian likelihood, log-space |
| 4 | [04_Decision_Tree.ipynb](supervised_learning/classification/04_Decision_Tree.ipynb) | Gini impurity, information gain, recursive splitting |

### Supervised Learning — Ensemble Methods

| # | Notebook | Key Concepts |
|---|----------|--------------|
| 1 | [01_Random_Forest.ipynb](supervised_learning/ensemble_methods/01_Random_Forest.ipynb) | Bagging, bootstrap sampling, feature subsampling, majority vote |

### Unsupervised Learning — Clustering

| # | Notebook | Key Concepts |
|---|----------|--------------|
| 1 | [01_KMeans.ipynb](unsupervised_learning/clustering/01_KMeans.ipynb) | K-Means++, WCSS, elbow method, purity |
| 2 | [02_DBSCAN.ipynb](unsupervised_learning/clustering/02_DBSCAN.ipynb) | Core/border/noise points, BFS expansion, eps, k-distance plot |

### Unsupervised Learning — Dimensionality Reduction

| # | Notebook | Key Concepts |
|---|----------|--------------|
| 1 | [01_PCA.ipynb](unsupervised_learning/dimensionality_reduction/01_PCA.ipynb) | Covariance matrix, eigendecomposition, explained variance ratio, reconstruction |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | All algorithm implementations, data generation, metrics |
| `matplotlib` | ≥ 3.5 | Visualisations only |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

### 3. Run any notebook

Open any `.ipynb` file and run **Kernel → Restart & Run All** to execute it from scratch. All notebooks are self-contained and reproduce exactly with fixed random seeds.

---

##  Notebook Structure (every notebook follows this pattern)

1. **Markdown header** — algorithm name, goal
2. **Algorithm intuition** — key idea in plain English
3. **Mathematical formulas** — with variable glossary table
4. **Imports** — `numpy`, `matplotlib` only
5. **Synthetic data generation** — NumPy, fixed seed
6. **Train/Test split & preprocessing** — fit on training data only
7. **Core implementation** — short functions, explicit loops, clear names
8. **Training loop** — gradient descent with J_history where applicable
9. **Visualisations** — cost curves, decision boundaries, cluster plots
10. **Evaluation metrics** — written in NumPy (accuracy, MSE, R², confusion matrix, purity...)
11. **Observations & Limitations** — what worked, what doesn't, when to use alternatives

---

## Style Guide

- Variables: `X`, `y`, `w`, `b`, `alpha`, `num_iters`, `J_history`, `lambda_`
- All function docstrings explain parameters and return values
- No classes except where recursion requires a node (`TreeNode`)
- No list comprehension tricks or one-liners that obscure meaning
- Comments explain the *why*, not the *what*

---

## Correctness Notes

- **Bias gradient for squared error:** `db = (1/m) * sum(y_hat - y)` — uses the correct sum divided by m.
- **Normalisation:** train-set mean and std are computed and applied to both train and test sets.
- **Log-space arithmetic** in Gaussian Naive Bayes prevents floating-point underflow.
- **K-Means++** initialisation for better centroid placement.
- **eigh (not eig)** in PCA for numerically stable eigendecomposition of the symmetric covariance matrix.

---

*Built for educational purposes. Every implementation prioritises clarity over performance.*
