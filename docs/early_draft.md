# Feasibility checkpoint

**Question:** How does classification accuracy change when k-nearest neighbours (k-NN) uses principal components instead of all 13 original chemical features to classify wine cultivars?

This checkpoint follows [early_draft_guidlines](../early_draft_guidlines). The acquired dataset, a working baseline, and a section-by-section paper outline are complete. The full analysis and paper are also available.

## Data acquired and loaded

The UCI Wine data are loaded through scikit-learn and committed as [data/wine.csv](../data/wine.csv). The feature matrix has **178 rows and 13 columns**, with a separate target vector of **178 labels**. Class counts are **59, 71, and 48**. There are **no missing feature values**. Sample identifiers and the class column in the CSV are excluded from the model inputs.

The target describes three wine cultivars from the same Italian region; it is not a quality rating. Data attribution, the class-label mapping, and the CC BY 4.0 licence are documented in [data/README.md](../data/README.md). UCI's dataset record is [Wine](https://archive.ics.uci.edu/dataset/109/wine), DOI [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J).

## Working baseline

The baseline uses all **13 features**, standardised within each training fold, followed by **k-NN with five neighbours**, uniform voting, and Euclidean distance. Evaluation uses **five-fold stratified cross-validation**, shuffling, and **seed 42**. No ensemble or neural network is needed to produce this result.

| Validation fold | Accuracy |
| --- | ---: |
| 1 | 97.22% |
| 2 | 97.22% |
| 3 | 100.00% |
| 4 | 94.29% |
| 5 | 97.14% |
| **Mean** | **97.17%** |
| **Sample standard deviation** | **2.02 percentage points** |

The baseline correctly classifies **173 of 178** pooled out-of-fold observations (**97.19%**). This pooled value differs slightly from the mean above because validation folds have different sizes. The standard deviation describes variation across folds and is not a confidence interval.

The baseline establishes a working result using feature scaling, k-NN, and model validation. The PCA extension compares 2, 5, and 10 components on exactly the same splits, with every transformation fitted within its training fold. The five-component model reaches a mean of 97.19% and also classifies 173 of 178 observations correctly. Its tiny mean advantage over the baseline is not an increase in the total number of correct predictions.

## Paper outline

1. **Abstract:** the question, dataset, comparison, and measured result.
2. **Introduction:** feature scaling, distance-based classification, and the motivation for PCA.
3. **Data and exploration:** acquisition, shape, classes, features, missing values, and descriptive PCA.
4. **Methods:** matched pipelines, shared stratified folds, leakage prevention, and evaluation metrics.
5. **Results:** baseline versus 2, 5, and 10 components, variability, confusion matrices, and retained variance.
6. **Sensitivity and tuning:** exploratory neighbour counts and a separately nested selection experiment.
7. **Discussion:** dimensionality versus accuracy, non-monotonic results, and limits of the small dataset.
8. **Conclusion:** the supported answer and useful further validation.
9. **Reproducibility and references:** code, data, result exports, and source attribution.

See [paper_outline.md](paper_outline.md) for the detailed outline and [report.md](report.md) for the complete paper. Exact results are in [results/analysis.json](../results/analysis.json); commands to reproduce them are in the [README](../README.md).
