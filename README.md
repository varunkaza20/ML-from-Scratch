# Machine Learning from Scratch

**A collection of classical machine learning algorithms implemented from first principles using NumPy.**

This project explores the mathematics and mechanics behind machine learning algorithms through simple, readable Python implementations. Inspired by Andrew Ng's machine learning lectures and programming assignments, each notebook connects mathematical intuition with practical implementation, visualization, and evaluation.

The focus is on **understanding how algorithms work, not hiding their mechanics behind machine learning libraries.**

## Table of Contents

- [Project Overview](#project-overview)
- [Project Goals](#project-goals)
- [Repository Structure](#repository-structure)
- [Algorithm Index](#algorithm-index)
- [Implementation Philosophy](#implementation-philosophy)
- [Getting Started](#getting-started)
- [Notebook Structure](#notebook-structure)
- [Mathematical and Numerical Correctness](#mathematical-and-numerical-correctness)
- [Scope and Limitations](#scope-and-limitations)

---

## Project Overview

Machine learning libraries make it easy to train models, but implementing algorithms from scratch provides a deeper understanding of the mathematics, optimization procedures, and assumptions behind them.

This repository is a collection of Jupyter notebooks covering fundamental algorithms across two major learning paradigms:

- **Supervised Learning:** Learning from labelled data to make predictions.
- **Unsupervised Learning:** Discovering patterns and structure in unlabelled data.

Each notebook presents the underlying intuition, derives or explains the relevant mathematical expressions, implements the algorithm using NumPy, and demonstrates its behaviour on a synthetic dataset.

## Project Goals

| Principle | Approach |
|---|---|
| **Learning first** | Prioritize mathematical understanding and readable code over computational performance. |
| **From scratch** | Implement the core algorithms directly instead of calling existing ML library implementations. |
| **Mathematical foundations** | Explain key equations, variables, objective functions, and optimization procedures. |
| **Self-contained notebooks** | Generate synthetic datasets within each notebook using NumPy and fixed random seeds where applicable. |
| **Readable implementation** | Prefer straightforward functions, explicit loops, descriptive names, and focused code cells. |
| **Visual understanding** | Use simple plots to illustrate convergence, predictions, decision boundaries, and learned structure. |
| **Numerical correctness** | Examine gradients, array dimensions, edge cases, and evaluation metrics carefully. |

## Repository Structure

```text
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

## Algorithm Index

### 1. Supervised Learning

Supervised learning uses labelled examples to learn a mapping from input features to target values.

#### Regression

Regression algorithms predict continuous numerical values.

| # | Algorithm | Key Concepts |
|---|---|---|
| 01 | [Linear Regression](supervised_learning/regression/01_Linear_Regression.ipynb) | Mean squared error, normal equation, gradient descent, learning curves, \(R^2\) |
| 02 | [Polynomial Regression](supervised_learning/regression/02_Polynomial_Regression.ipynb) | Polynomial feature expansion, nonlinear relationships, model complexity, overfitting |
| 03 | [Ridge Regression](supervised_learning/regression/03_Ridge_Regression.ipynb) | L2 regularization, penalized loss, coefficient shrinkage, regularization strength |

#### Classification

Classification algorithms predict discrete class labels.

| # | Algorithm | Key Concepts |
|---|---|---|
| 01 | [Logistic Regression](supervised_learning/classification/01_Logistic_Regression.ipynb) | Sigmoid function, binary cross-entropy, gradient descent, decision boundary |
| 02 | [K-Nearest Neighbors](supervised_learning/classification/02_KNN.ipynb) | Euclidean distance, nearest neighbors, majority voting, choice of \(k\) |
| 03 | [Gaussian Naive Bayes](supervised_learning/classification/03_Gaussian_Naive_Bayes.ipynb) | Bayes' theorem, class priors, Gaussian likelihood, log probabilities |
| 04 | [Decision Tree](supervised_learning/classification/04_Decision_Tree.ipynb) | Gini impurity, information gain, feature splits, recursive tree construction |

#### Ensemble Methods

Ensemble learning combines multiple models to produce predictions.

| # | Algorithm | Key Concepts |
|---|---|---|
| 01 | [Random Forest](supervised_learning/ensemble_methods/01_Random_Forest.ipynb) | Bootstrap sampling, bagging, random feature selection, majority voting |

### 2. Unsupervised Learning

Unsupervised learning discovers patterns or useful representations without relying on labelled target values.

#### Clustering

Clustering groups observations based on similarities in their features.

| # | Algorithm | Key Concepts |
|---|---|---|
| 01 | [K-Means](unsupervised_learning/clustering/01_KMeans.ipynb) | Centroid initialization, assignment and update steps, within-cluster sum of squares |
| 02 | [DBSCAN](unsupervised_learning/clustering/02_DBSCAN.ipynb) | Density-based clustering, core and border points, noise detection, epsilon and minimum points |

#### Dimensionality Reduction

Dimensionality reduction represents data using fewer features while attempting to retain important information.

| # | Algorithm | Key Concepts |
|---|---|---|
| 01 | [Principal Component Analysis (PCA)](unsupervised_learning/dimensionality_reduction/01_PCA.ipynb) | Data centering, covariance matrices, eigendecomposition, explained variance, projection and reconstruction |

---

## Implementation Philosophy

### Minimal dependencies

The core algorithm implementations use NumPy for numerical computation and Matplotlib for visualization.

| Library | Purpose |
|---|---|
| NumPy | Numerical operations, matrix calculations, synthetic data generation, and evaluation metrics |
| Matplotlib | Plots, learning curves, decision boundaries, and cluster visualizations |

Scikit-Learn, SciPy, TensorFlow, and PyTorch are not used to implement the algorithms.

### Readability over abstraction

The implementations follow a straightforward, assignment-style approach:

- Use familiar variable names such as `X`, `y`, `w`, `b`, `alpha`, `num_iters`, and `J_history`.
- Break mathematical operations into small, understandable functions.
- Use explicit loops when they clarify the algorithm.
- Include docstrings describing function parameters and return values.
- Explain why important operations are performed, rather than merely commenting on what each line does.
- Avoid unnecessary classes, advanced language features, and abstractions that hide the underlying mathematics.

### Mathematical transparency

Whenever applicable, notebooks explicitly show the objective or loss function, parameter initialization, parameter updates, prediction procedure, and evaluation.

For example, linear regression implemented using batch gradient descent follows the objective:

\[
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}
\left(\hat{y}^{(i)}-y^{(i)}\right)^2
\]

The updates are calculated using the gradients of this objective rather than delegated to an external estimator.

### Reproducible experimentation

Each notebook generates its own illustrative dataset using NumPy. Fixed random seeds are used where applicable so that experiments can be reproduced.

When a train/test split is used, preprocessing parameters such as feature means and standard deviations are estimated from the training set and then applied to the test set.

---

## Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Jupyter Notebook or JupyterLab

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ML-from-Scratch.git
cd ML-from-Scratch
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install jupyterlab
```

The additional JupyterLab installation provides the notebook environment if it is not already installed.

### 4. Launch JupyterLab

```bash
jupyter lab
```

Navigate to the algorithm you want to explore and open its notebook.

### 5. Execute a notebook

Run the notebook from its first cell to its last. Where supported, use **Restart Kernel and Run All Cells** to check that it executes from a fresh kernel without relying on previously defined variables.

---

## Notebook Structure

Each notebook follows a consistent learning-oriented sequence, adapted to the algorithm being implemented.

| Step | Section | Purpose |
|---|---|---|
| 01 | Algorithm Overview | Introduce the problem, intuition, and use cases. |
| 02 | Mathematical Foundations | Explain the important equations and define their variables. |
| 03 | Imports and Utilities | Set up NumPy, Matplotlib, and essential helper functions. |
| 04 | Data Generation | Create a small, illustrative synthetic dataset. |
| 05 | Preprocessing | Perform appropriate data preparation and train/test splitting. |
| 06 | Implementation | Translate the mathematical procedure into readable Python functions. |
| 07 | Training or Fitting | Learn the parameters, construct the model, or discover clusters. |
| 08 | Visualization | Plot useful intermediate or final results. |
| 09 | Evaluation | Measure performance using appropriate metrics implemented with NumPy. |
| 10 | Observations and Limitations | Discuss assumptions, observed behaviour, and situations where the algorithm may struggle. |

Not every algorithm requires every step in precisely this order. For example, K-Means does not use labelled targets, and K-Nearest Neighbors does not require gradient descent.

---

## Mathematical and Numerical Correctness

Readability alone is not sufficient; the implementations must also reflect the underlying mathematics.

The following are important implementation considerations throughout the repository:

- **Gradient calculations:** Derive parameter updates from the corresponding objective function and check their dimensions.
- **Feature preprocessing:** Estimate preprocessing parameters on training data only when evaluating supervised models on held-out data.
- **Numerical stability:** Handle logarithms, small probabilities, zero standard deviations, and division-by-zero cases where relevant.
- **Distance-based algorithms:** Use consistent distance calculations and clearly defined parameters.
- **Clustering:** Handle empty clusters, stopping conditions, and noise points where appropriate.
- **PCA:** Use a symmetric covariance matrix and `numpy.linalg.eigh` for eigendecomposition.
- **Evaluation:** Calculate metrics with NumPy and explain what they measure instead of relying on imported ML metrics.

For example, the bias gradient for the standard linear regression squared-error objective is:

```python
db = (1 / m) * np.sum(preds - y)
```

It uses prediction errors rather than the sum of predictions alone.

These checks support the project's central objective: implementations that are both understandable and mathematically sound.

---

## Scope and Limitations

This repository is intended for education and experimentation, not production deployment.

The implementations prioritize clarity over speed and may not provide the optimizations, extensive validation, numerical safeguards, or large-scale performance of mature machine learning libraries.

Synthetic datasets are useful for illustrating how algorithms behave, but they do not establish that a model will perform well on real-world data. Algorithm assumptions, parameter choices, dataset characteristics, and evaluation methods all affect the results.

The purpose of this project is to build a stronger foundation in machine learning by studying how the algorithms work, how their mathematics translates into code, and where their limitations arise.

---

*An ongoing learning focused on understanding machine learning from first principles.*
