# Paper outline

**Research question:** How does classification accuracy change when k-nearest neighbours (k-NN) uses principal components instead of all 13 original chemical features to classify wine cultivars?

1. **Abstract.** State the question, UCI Wine dataset, matched cross-validation design, and the main accuracy-versus-dimension result. Distinguish a near tie from evidence of an improvement.
2. **Introduction.** Explain why feature scaling matters for distance-based classification and why correlated measurements make PCA worth investigating. State the comparison proposed in [project_proposal](../project_proposal).
3. **Data and exploration.** Document the acquired dataset, its 178 × 13 feature matrix, class counts, feature names, missing-value checks, provenance, and licence. Describe the full-data PCA plot as exploration rather than model evaluation.
4. **Methods.** Specify standardisation, PCA settings, uniform-weight Euclidean k-NN, and the fixed baseline of five neighbours. Explain the shared shuffled stratified five-fold split, seed 42, and why transformations must be fitted inside each training fold. Define mean accuracy, sample standard deviation, and pooled out-of-fold accuracy.
5. **Main results.** Compare the original 13 features with 2, 5, and 10 principal components in a table and accuracy graph. Report the paired differences from the baseline and out-of-fold confusion matrices. Explain any discrepancy between mean fold accuracy and pooled accuracy.
6. **Sensitivity and tuning.** Show how the comparison changes across the neighbour counts. Treat this sweep as exploratory. Present nested cross-validation as a separate evaluation of neighbour-count tuning, with selection restricted to the inner folds.
7. **Discussion and limitations.** Interpret accuracy alongside dimension reduction and retained variance. Explain that PCA preserves variance rather than class separation. Discuss the small dataset, dependent folds, one seeded partition, and lack of external validation.
8. **Conclusion.** Answer the original question using the measured results. Identify what the evidence supports and what would require repeated or external validation.
9. **Reproducibility and references.** Link the code, pinned environment, raw data, result files, dashboard, and figures. Credit UCI and the relevant scikit-learn documentation.

The complete paper is [report.md](report.md). The feasibility checkpoint is [early_draft.md](early_draft.md).
