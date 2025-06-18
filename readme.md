# HKL: Harmony Search Kullback-Leibler Feature Selection Algorithm

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

HKL (Harmony search Kullback-Leibler) is a novel feature selection algorithm specifically designed for high-dimensional imbalanced datasets. By integrating Kullback-Leibler divergence with the Harmony Search metaheuristic, HKL provides an information-theoretic approach to identify features that maximize the separability between minority and majority classes.

## 📄 Citation

If you use this algorithm in your research, please cite:

```bibtex
@article{moayedikia2024hkl,
  title={Novel Feature Selection Algorithm Using Harmony Search and Kullback-Leibler Divergence for High-Dimensional Imbalanced Class Datasets},
  author={Moayedikia, Alireza and Jensen, Richard},
  journal={Data Mining and Knowledge Discovery},
  year={2025},
  publisher={Elsevier}
}
```

## ✨ Key Features

- **Information-theoretic foundation**: Uses KL divergence to evaluate feature subsets based on their ability to separate minority and majority classes
- **Direct class imbalance awareness**: Explicitly incorporates class distribution disparities into the optimization process
- **Dual optimization approach**: Balances both classification performance (G-mean) and class distribution divergence
- **Enhanced minority class discrimination**: Prioritizes features that maximize divergence between class distributions
- **High-dimensional support**: Efficiently handles datasets with thousands of features

## 🛠️ Installation

### Requirements

```bash
numpy>=1.19.0
scikit-learn>=0.23.0
pandas>=1.0.0
matplotlib>=3.2.0
seaborn>=0.10.0
scipy>=1.5.0
```

### Install via pip

```bash
pip install -r requirements.txt
```

### Clone the repository

```bash
git clone https://github.com/yourusername/HKL-Feature-Selection.git
cd HKL-Feature-Selection
```

## 🚀 Quick Start

### Basic Usage

```python
from hkl import HKL
from sklearn.datasets import make_classification

# Generate an imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=200,
    n_informative=30,
    n_redundant=20,
    weights=[0.9, 0.1],  # 90-10 class imbalance
    random_state=42
)

# Initialize HKL with optimal parameters
hkl = HKL(HMS=30, HMCR=0.9, PAR=0.3, beta=0.7, NI=150)

# Fit and transform the data
X_selected = hkl.fit_transform(X, y)

# Get selected feature indices
selected_features = hkl.get_selected_features()
print(f"Selected {len(selected_features)} out of {X.shape[1]} features")
```

### Extended Usage with Evaluation

```python
from hkl_extended import HKLExtended

# Initialize with verbose output
hkl = HKLExtended(
    HMS=30,      # Harmony Memory Size
    HMCR=0.9,    # Harmony Memory Consideration Rate
    PAR=0.3,     # Pitch Adjustment Rate
    beta=0.7,    # Weight parameter for fitness function
    NI=150,      # Number of iterations
    verbose=True
)

# Fit the algorithm
hkl.fit(X, y)

# Evaluate performance with cross-validation
results = hkl.evaluate_performance(X, y, cv_folds=5)

print("\nPerformance Metrics:")
for metric in ['accuracy', 'sensitivity', 'specificity', 'gmean', 'auc']:
    mean_val = results[f'{metric}_mean']
    std_val = results[f'{metric}_std']
    print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

# Visualize convergence
hkl.plot_convergence()
hkl.plot_feature_importance(top_n=20)
```

## 📊 Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **HMS** | 30 | Harmony Memory Size - number of solution vectors maintained |
| **HMCR** | 0.9 | Harmony Memory Consideration Rate - probability of selecting from memory |
| **PAR** | 0.3 | Pitch Adjustment Rate - probability of adjusting selected features |
| **beta** | 0.7 | Weight parameter balancing G-mean (70%) and KL divergence (30%) |
| **alpha** | 1.0 | Scaling parameter for sigmoid function in KL-guided selection |
| **NI** | 150 | Number of Improvisations (iterations) |
| **n_bins** | 50 | Number of bins for discretizing continuous features |

## 📈 Performance Results

HKL has been tested on six high-dimensional imbalanced datasets:

| Dataset | Features | Samples | Classes | HKL G-mean | Best Competitor |
|---------|----------|---------|---------|------------|-----------------|
| Colon (CL) | 2,000 | 60 | 2 | 0.715 | 0.715 (SYMON) |
| CNS | 7,129 | 50 | 2 | 0.810 | 0.790 (SYMON) |
| Lung | 12,533 | 181 | 2 | 1.000 | 1.000 (Multiple) |
| Breast Cancer | 24,481 | 97 | 2 | 0.690 | 0.660 (SYMON) |
| Cardiovascular | 9,182 | 174 | 11 | 1.000 | 1.000 (SYMON) |
| Glioma | 4,433 | 50 | 4 | 0.841 | 0.800 (SYMON) |

## 🔧 Advanced Features

### Convergence Analysis

```python
# Track algorithm convergence
hkl = HKLExtended(verbose=True)
hkl.fit(X, y)

# Access convergence history
fitness_history = hkl.fitness_history
kl_history = hkl.kl_history
stability_history = hkl.feature_stability

# Plot convergence metrics
fig = hkl.plot_convergence(figsize=(15, 5))
```

### Feature Importance Analysis

```python
# Get KL divergence values for all features
importances = hkl.get_feature_importances()

# Plot top features
hkl.plot_feature_importance(top_n=30)
```

### Custom Fitness Function

You can modify the fitness function by adjusting the beta parameter:
- `beta = 1.0`: Pure classification-based selection (G-mean only)
- `beta = 0.7`: Balanced approach (recommended)
- `beta = 0.0`: Pure KL divergence-based selection

## 📁 Repository Structure

```
HKL-Feature-Selection/
│
├── hkl.py                 # Basic HKL implementation
├── hkl_extended.py        # Extended version with utilities
├── requirements.txt       # Package dependencies
├── README.md             # This file
├── LICENSE               # MIT license
│
├── examples/
│   ├── basic_usage.py    # Simple example
│   ├── advanced_usage.py # Advanced features demo
│   └── benchmark.py      # Comparison with other methods
│
├── datasets/
│   └── README.md         # Instructions for dataset preparation
│
└── results/
    ├── convergence/      # Convergence plots
    └── performance/      # Performance comparison results
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Dr. Alireza Moayedikia** - *Corresponding Author* - Department of Business Technology and Entrepreneurship, Swinburne University of Technology
  - Email: amoayedikia@swin.edu.au
  
- **Prof. Richard Jensen** - Department of Computer Science, Aberystwyth University

## 🙏 Acknowledgments

- This research was supported by [funding organization if applicable]
- Thanks to all contributors who have helped improve this algorithm

## 📚 Related Work

For more information on feature selection for imbalanced datasets, see:

- [SYMON](https://github.com/link-to-symon) - Symmetric Uncertainty-based Harmony Search
- [SVM-RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) - Recursive Feature Elimination
- [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) - Synthetic Minority Oversampling

## ❓ FAQ

**Q: Can HKL handle multi-class imbalanced problems?**  
A: The current implementation is designed for binary classification. Extension to multi-class problems is planned for future releases.

**Q: What is the computational complexity?**  
A: O(d·n·(HMS + NI)), where d is the number of features, n is the number of samples, HMS is harmony memory size, and NI is the number of iterations.

**Q: How do I choose the optimal parameters?**  
A: The default parameters (HMS=30, HMCR=0.9, PAR=0.3, β=0.7) were optimized through extensive experiments. For specific datasets, you may want to perform parameter tuning using the provided parameter ranges in the paper.

## 📞 Contact

For questions, issues, or collaborations, please:
- Open an issue on GitHub
- Email the corresponding author at amoayedikia@swin.edu.au

---

⭐ If you find this work useful, please consider starring the repository!
