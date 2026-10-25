# Generated results

Recreate every artifact with `python3 -m src.analysis` from the repository root.
The analysis is deterministic for the recorded dependency versions. No network or
model service is required. All CSV and JSON accuracies are **fractions** in [0, 1];
multiply by 100 for percentages. Multiply accuracy differences by 100 for percentage
points. Standard deviations are sample SDs over five folds (`ddof=1`), not confidence
intervals.

| Artifact | Contents |
| --- | --- |
| `analysis.json` | Complete dashboard data, metadata, methods, and results |
| `model_comparison.csv` | Primary fixed-k comparison (all features; 2, 5, 10 PCs) |
| `fold_scores.csv` | Per-fold accuracy and train/test sizes for the primary comparison |
| `cv_splits.json` | Exact 1-based sample identifiers in the five shared outer folds |
| `oof_predictions.csv` | Actual class and out-of-fold prediction for every sample/model |
| `class_metrics.csv` | Precision, recall, F1, and support from out-of-fold predictions |
| `knn_sensitivity.csv` | Descriptive fixed-k experiments for k = 1, 3, 5, 7, 9, 11 |
| `nested_cv.csv` | Outer-fold scores and independently selected inner-CV k values |
| `feature_summary.csv` | Descriptive chemical measurement statistics |
| `explained_variance.csv` | Full-data descriptive PCA variance ratios |
| `pca_loadings.csv` | Full-data PCA unit eigenvectors (rows); signs are arbitrary |
| `pca_projection.csv` | Full-data two-component projection and cultivar labels |
| `figures/` | Six charts, each in PNG and SVG format |

All classification scores use training-fold-only scaling and optional PCA. The
full-data PCA projection and loadings describe the observations and are not inputs
to model evaluation. Each out-of-fold prediction is made by a classifier that did
not train on that observation. The same five folds are reused across representations.

The baseline and five-component model both classify 173/178 out-of-fold observations
correctly. Their mean fold accuracies differ slightly because fold sizes are 36, 36,
36, 35, and 35. The primary comparison fixes k=5 in advance. The sensitivity graph
must not be used to choose k and then report that same graph's best score as an
unbiased estimate; the separate nested-CV extension tunes k inside training data.

Results concern one small historical dataset and one partition. Overlapping training
sets make fold scores dependent. The study makes no significance, equivalence, or
external-generalization claim.
