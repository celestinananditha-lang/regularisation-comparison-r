# Regularisation Across Model Families: OLS, SVM and Neural Networks in R

How much does regularisation actually help, and does it help every model family the same way? This project fits **10 model variants** to **two regression datasets** and compares test MSE and run time, with and without L1, L2 and Elastic Net penalties.

Built for MIS41120 Statistical Learning (MSc Business Analytics, UCD Smurfit) as a team of three.

## Datasets

| Dataset | Rows | Predictors | Target |
|---|---|---|---|
| Boston Housing (`MASS::Boston`) | 506 | 13 | Per-capita crime rate (`crim`) |
| UCI Communities and Crime (`communities.data`) | 1,994 | 100 (after cleaning) | Violent crimes per population (normalised 0–1) |

## Methods

- **Linear:** OLS, Ridge (L2), Lasso (L1), Elastic Net (`glmnet`, lambda chosen by 10-fold CV)
- **SVM:** radial kernel, near-unregularised (C = 1000) vs L2-regularised with C tuned by 10-fold CV (`e1071`)
- **Neural networks (MLP, 5–3 hidden layers):** unregularised (`neuralnet`), L2 weight decay (`nnet`), L1 and Elastic Net (`ANN2`)
- 80/20 train/test split made **before** any fitting; min-max scaling uses training-set statistics only, so nothing leaks from the test set
- `set.seed(123)` throughout for reproducibility

## Key results (test MSE)

| Model | Boston | Communities & Crime |
|---|---|---|
| OLS | 20.07 | **0.0159** |
| Ridge (L2) | 19.12 | 0.0164 |
| Lasso (L1) | 19.72 | 0.0166 |
| Elastic Net | 19.24 | 0.0162 |
| SVM, C = 1000 | 28.20 | 0.0248 |
| SVM, CV-tuned C | 10.94 | 0.0178 |
| MLP, unregularised | **7.35** | 0.8183 |
| MLP, L2 decay | 12.37 | 0.0517 |
| MLP, L1 | 7.59 | 0.0288 |
| MLP, Elastic Net | 7.59 | 0.0277 |

**Takeaways**

- **Regularisation matters most where the model can overfit.** On Communities and Crime (100 predictors), the unregularised MLP blew up to MSE 0.82. L1/Elastic Net penalties cut that by about 97%.
- **For SVMs, tuning C is the regularisation.** CV-tuned C roughly halved Boston MSE versus C = 1000.
- **Linear models barely moved.** Ridge, Lasso and Elastic Net landed within a few percent of OLS on both datasets. The Lasso path did zero out over half of the 100 crime predictors with almost no loss in accuracy, which makes for a far more interpretable model.
- **Cost:** the `neuralnet` MLPs took ~150 s on the crime data versus ~4–5 s for the `ANN2` versions.

<p align="center">
  <img src="results/figures/stl_Ellora_HirudayaKumar_Le_mse_comparison.png" width="80%">
</p>

More plots (Lasso paths, SVM CV curves, residuals per model, run times) are in [`results/figures`](results/figures).

## Repo structure

```
├── stl_Ellora_HirudayaKumar_utils.R    # package setup + shared helpers (scaling, MSE, residual plots)
├── stl_Ellora_HirudayaKumar_boston.R   # all models on Boston
├── stl_Ellora_HirudayaKumar_crime.R    # all models on Communities and Crime
├── stl_Ellora_HirudayaKumar_plots.R    # runs both, writes summary CSV + comparison plots
├── communities.data                       # UCI Communities and Crime dataset
└── results/
    ├── stl_Ellora_HirudayaKumar_Le_results_summary.csv
    └── figures/
```

## How to run

Requires R 4.x. Missing packages (`MASS`, `glmnet`, `e1071`, `neuralnet`, `nnet`, `caret`, `dplyr`, `ANN2`, `sparseSVM`) install automatically.

```r
setwd("path/to/this/repo")
source("stl_Ellora_HirudayaKumar_Le_plots.R")   # runs everything; the crime MLPs take a few minutes
```

## Team

Sudarshan Ellora · Mario Celestina Nanditha Hirudaya Kumar 

## Data source

Redmond, M. (2002). *Communities and Crime* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53W3X
