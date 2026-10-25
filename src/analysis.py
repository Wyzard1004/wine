"""Run offline, leakage-free Wine classification experiments and export artifacts.

Run from the repository root: python3 -m src.analysis
The bundled scikit-learn Wine data are an offline copy of the UCI dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wine-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_SPLITS = 5
K = 5
K_VALUES = (1, 3, 5, 7, 9, 11)
REPRESENTATIONS = (("baseline", 13), ("pca_2", 2), ("pca_5", 5), ("pca_10", 10))
SOURCE_URL = "https://archive.ics.uci.edu/dataset/109/wine"
CLASS_LABELS = ("Cultivar 1", "Cultivar 2", "Cultivar 3")
FEATURE_LABELS = (
    "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline",
)


def acquire_data(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Acquire the installed, network-independent Wine copy and save raw observations."""
    wine = load_wine()
    X = np.asarray(wine.data, dtype=float)
    y = np.asarray(wine.target, dtype=int) + 1
    if X.shape != (178, 13) or not np.isfinite(X).all():
        raise ValueError("The Wine data must contain 178 complete rows and 13 features.")
    if not np.array_equal(np.bincount(y)[1:], [59, 71, 48]):
        raise ValueError("Unexpected Wine cultivar counts.")
    frame = pd.DataFrame(X, columns=wine.feature_names)
    frame.insert(0, "class_id", y)
    frame.insert(0, "sample_id", np.arange(1, len(y) + 1))
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "wine.csv"
    frame.to_csv(path, index=False, float_format="%.8g")
    metadata = {
        "title": "Wine",
        "creators": "Aeberhard, S. & Forina, M.",
        "year": 1992,
        "citation": "Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J.",
        "doi": "https://doi.org/10.24432/C5PC7J",
        "source_url": SOURCE_URL,
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "acquisition": "sklearn.datasets.load_wine (bundled UCI copy; no network request)",
        "transformations": "Added 1-based sample_id; converted target 0,1,2 to class_id 1,2,3; chemical measurements unchanged.",
        "samples": 178,
        "features": 13,
        "class_counts": {str(i): int((y == i).sum()) for i in (1, 2, 3)},
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    write_json(data_dir / "metadata.json", metadata)
    return frame, X, y


def make_pipeline(components: int, k: int = K) -> Pipeline:
    """Fit standardization and optional PCA on the training rows of each fold only."""
    steps = [("scale", StandardScaler())]
    if components != 13:
        steps.append(("pca", PCA(n_components=components, svd_solver="full")))
    steps.append(("knn", KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")))
    return Pipeline(steps)


def make_splits(X: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y))


def evaluate_fixed(
    X: np.ndarray, y: np.ndarray, splits: list, components: int, k: int = K
) -> dict:
    predictions = np.zeros_like(y)
    fold_ids = np.zeros_like(y)
    folds = []
    retained_variance = []
    for number, (train, test) in enumerate(splits, start=1):
        pipeline = make_pipeline(components, k).fit(X[train], y[train])
        prediction = pipeline.predict(X[test])
        predictions[test] = prediction
        fold_ids[test] = number
        folds.append({
            "fold": number, "train_size": len(train), "test_size": len(test),
            "accuracy": float(np.mean(prediction == y[test])),
        })
        if components != 13:
            retained_variance.append(float(pipeline.named_steps["pca"].explained_variance_ratio_.sum()))
    scores = np.asarray([f["accuracy"] for f in folds])
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=[1, 2, 3], zero_division=0
    )
    return {
        "k": k, "components": components, "folds": folds,
        "fold_scores": scores.tolist(), "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std(ddof=1)),
        "oof_accuracy": float(np.mean(y == predictions)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[1, 2, 3]).tolist(),
        "class_metrics": [
            {"class_id": i + 1, "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])} for i in range(3)
        ],
        "retained_variance_by_fold": retained_variance,
        "predictions": predictions.tolist(), "fold_ids": fold_ids.tolist(),
    }


def evaluate_nested(X: np.ndarray, y: np.ndarray, splits: list) -> dict:
    """Tune k inside outer training folds; the outer test rows never select k."""
    models = []
    for model_id, components in REPRESENTATIONS:
        scores = []
        selected_k = []
        for number, (train, test) in enumerate(splits, start=1):
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + number)
            search = GridSearchCV(
                make_pipeline(components), {"knn__n_neighbors": list(K_VALUES)},
                scoring="accuracy", cv=inner, n_jobs=1, refit=True,
            ).fit(X[train], y[train])
            selected_k.append(int(search.best_params_["knn__n_neighbors"]))
            scores.append(float(search.score(X[test], y[test])))
        models.append({
            "id": model_id, "components": components, "fold_scores": scores,
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores, ddof=1)), "selected_k": selected_k,
        })
    return {
        "description": "Exploratory nested CV: same five outer folds; three stratified inner folds select k from 1,3,5,7,9,11 independently for each representation. Ties select the first (smallest) k. Representation itself is not tuned.",
        "inner_splits": 3, "k_values": list(K_VALUES), "models": models,
    }


def descriptive_pca(X: np.ndarray, y: np.ndarray) -> dict:
    standardized = StandardScaler().fit_transform(X)
    fitted = PCA(svd_solver="full").fit(standardized)
    projection = fitted.transform(standardized)
    return {
        "scope": "Full-data descriptive fit; excluded from model evaluation",
        "explained_variance_ratio": fitted.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(fitted.explained_variance_ratio_).tolist(),
        "loadings": fitted.components_.tolist(),
        "loading_definition": "Rows are principal-component unit eigenvectors; columns follow dataset.feature_names. Signs are arbitrary.",
        "projection": [{"sample_id": i + 1, "class_id": int(y[i]),
                        "pc1": float(projection[i, 0]), "pc2": float(projection[i, 1])}
                       for i in range(len(y))],
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_figures(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    colors = ["#81364b", "#c3915f", "#648578"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.labelcolor": "#364039",
        "text.color": "#26332d", "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff", "svg.hashsalt": "wine-study",
    })

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=220, metadata={"Software": "Wine PCA study"})
        fig.savefig(output / f"{name}.svg", metadata={"Date": None})
        plt.close(fig)

    # Ordered by dimension so the curve answers the proposal's question directly.
    ordered = sorted(result["models"], key=lambda m: m["components"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.array([m["components"] for m in ordered])
    means = np.array([m["mean_accuracy"] for m in ordered]) * 100
    errors = np.array([m["std_accuracy"] for m in ordered]) * 100
    ax.errorbar(x, means, yerr=errors, marker="o", color=colors[0], linewidth=2,
                markersize=8, capsize=5, label="Mean ± sample SD across 5 folds")
    for xi, yi in zip(x, means):
        ax.annotate(f"{yi:.2f}%", (xi, yi), xytext=(0, 11), textcoords="offset points", ha="center")
    ax.set(xticks=x, xticklabels=["2 PCs", "5 PCs", "10 PCs", "13 original"],
           ylabel="Classification accuracy (%)", xlabel="Input representation",
           title="k-NN accuracy after dimensionality reduction (k = 5)")
    ax.grid(axis="y", alpha=.18)
    ax.set_ylim(max(0, min(means-errors)-4), min(102, max(means+errors)+4))
    ax.legend(loc="lower right", frameon=False)
    fig.text(.02, .01, "Same stratified folds; scaling and PCA fit inside each training fold. SD is not a confidence interval.", fontsize=8)
    fig.subplots_adjust(bottom=.19)
    save(fig, "accuracy_comparison")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ratio = np.asarray(result["pca"]["explained_variance_ratio"]) * 100
    cumulative = np.asarray(result["pca"]["cumulative_variance"]) * 100
    ax.bar(range(1, 14), ratio, color=colors[2], alpha=.75, label="Individual variance")
    ax.plot(range(1, 14), cumulative, marker="o", color=colors[0], label="Cumulative variance")
    ax.set(xticks=range(1, 14), xlabel="Principal component", ylabel="Explained variance (%)",
           title="How much chemical variation do the components retain?", ylim=(0, 105))
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False)
    fig.text(.02, .01, "Descriptive PCA fit to all standardized samples; not used for accuracy evaluation.", fontsize=8)
    save(fig, "explained_variance")

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    points = result["pca"]["projection"]
    for class_id, color in zip([1, 2, 3], colors):
        group = [p for p in points if p["class_id"] == class_id]
        ax.scatter([p["pc1"] for p in group], [p["pc2"] for p in group], c=color,
                   s=30, alpha=.8, linewidth=.4, edgecolor="white", label=CLASS_LABELS[class_id-1])
    ax.set(xlabel=f"PC1 ({ratio[0]:.1f}% variance)", ylabel=f"PC2 ({ratio[1]:.1f}% variance)",
           title="The Wine dataset in two principal components")
    ax.axhline(0, color="#d8ded9", linewidth=.6)
    ax.axvline(0, color="#d8ded9", linewidth=.6)
    ax.legend(frameon=False)
    fig.text(.02, .01, "Full-data descriptive projection; apparent separation is not held-out accuracy.", fontsize=8)
    save(fig, "pca_projection")

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6), sharex=True, sharey=True)
    for ax, model in zip(axes, result["models"]):
        matrix = np.asarray(model["confusion_matrix"])
        ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=71)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i,j]), ha="center", va="center",
                        color="white" if matrix[i,j] > 35 else "#26332d")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=[1, 2, 3], yticklabels=[1, 2, 3],
               xlabel="Predicted cultivar", title="13 features" if model["id"] == "baseline" else f"{model['components']} PCs")
    axes[0].set_ylabel("Actual cultivar")
    fig.suptitle("Out-of-fold confusion matrices (k = 5)", fontweight="bold")
    save(fig, "confusion_matrices")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for index, (model_id, components) in enumerate(REPRESENTATIONS):
        rows = [r for r in result["knn_sensitivity"] if r["model_id"] == model_id]
        ax.plot([r["k"] for r in rows], [r["mean_accuracy"]*100 for r in rows],
                marker="o", linewidth=1.8, label="13 original features" if components==13 else f"{components} PCs")
    ax.set(xticks=K_VALUES, xlabel="Number of neighbours (k)", ylabel="Mean CV accuracy (%)",
           title="Sensitivity to the number of neighbours")
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False)
    fig.text(.02, .01, "Exploratory same-fold comparison; selecting the maximum here would bias performance estimates.", fontsize=8)
    save(fig, "knn_sensitivity")

    fig, ax = plt.subplots(figsize=(11, 8.5))
    corr = np.asarray(result["dataset"]["correlation"])
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = result["dataset"]["feature_labels"]
    ax.set(xticks=range(13), yticks=range(13), xticklabels=labels, yticklabels=labels,
           title="Chemical-feature correlations (Pearson r)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.75, label="Pearson correlation")
    save(fig, "feature_correlations")


def run(output_dir: Path = ROOT / "results", data_dir: Path = ROOT / "data") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, X, y = acquire_data(data_dir)
    feature_names = frame.columns[2:].tolist()
    splits = make_splits(X, y)
    models = []
    for model_id, components in REPRESENTATIONS:
        model = evaluate_fixed(X, y, splits, components)
        model.update({"id": model_id, "label": "All 13 features" if components == 13 else f"{components} principal components"})
        models.append(model)
    baseline = models[0]
    for model in models:
        deltas = np.asarray(model["fold_scores"]) - np.asarray(baseline["fold_scores"])
        model["delta_vs_baseline"] = float(deltas.mean())
        model["paired_fold_deltas"] = deltas.tolist()
        model["std_paired_delta"] = float(deltas.std(ddof=1))

    sensitivity = []
    for k in K_VALUES:
        for model_id, components in REPRESENTATIONS:
            model = next(m for m in models if m["id"] == model_id) if k == K else evaluate_fixed(X, y, splits, components, k)
            sensitivity.append({"k": k, "model_id": model_id,
                                "mean_accuracy": model["mean_accuracy"], "std_accuracy": model["std_accuracy"]})
    dummy_scores = []
    for train, test in splits:
        dummy = DummyClassifier(strategy="most_frequent").fit(X[train], y[train])
        dummy_scores.append(float(dummy.score(X[test], y[test])))

    result = {
        "metadata": {
            "seed": SEED, "n_splits": N_SPLITS, "k": K, "source_url": SOURCE_URL,
            "license": "CC BY 4.0", "method": "Shuffled stratified 5-fold CV; fold-local StandardScaler, optional unwhitened PCA, Euclidean uniform-weight k-NN (k=5). Same folds for every representation.",
            "variability": "Sample standard deviation (ddof=1) across five fold accuracies; not a confidence interval.",
            "accuracy_units": "Fractions in [0,1]; multiply by 100 for percentages. Delta multiplied by 100 gives percentage points.",
            "selection": "k=5 fixed before comparison. k-sensitivity is descriptive; nested CV tunes k using inner training folds only.",
            "limitations": "One small historical dataset, one 5-fold partition, no external test set. Fold scores share training observations and are dependent. No statistical significance or equivalence claim is made.",
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        },
        "dataset": {
            "n_samples": len(y), "n_features": X.shape[1], "n_classes": len(np.unique(y)),
            "missing_values": int(np.isnan(X).sum()), "duplicate_feature_rows": int(pd.DataFrame(X).duplicated().sum()),
            "feature_names": feature_names, "feature_labels": list(FEATURE_LABELS),
            "class_counts": [{"class_id": i, "label": CLASS_LABELS[i-1], "count": int(np.sum(y==i))} for i in [1,2,3]],
            "records": frame.to_dict(orient="records"),
            "feature_summary": [
                {"name": name, "label": FEATURE_LABELS[i], "mean": float(X[:,i].mean()),
                 "std": float(X[:,i].std(ddof=1)), "min": float(X[:,i].min()), "max": float(X[:,i].max())}
                for i, name in enumerate(feature_names)
            ],
            "correlation": np.corrcoef(X, rowvar=False).tolist(),
        },
        "models": models, "pca": descriptive_pca(X, y), "knn_sensitivity": sensitivity,
        "nested_cv": evaluate_nested(X, y, splits),
        "majority_baseline": {"strategy": "most_frequent", "fold_scores": dummy_scores,
                              "mean_accuracy": float(np.mean(dummy_scores)),
                              "std_accuracy": float(np.std(dummy_scores, ddof=1))},
    }
    write_json(output_dir / "analysis.json", result)
    pd.DataFrame([{key: model[key] for key in ["id", "label", "components", "k", "mean_accuracy", "std_accuracy", "oof_accuracy", "delta_vs_baseline"]} for model in models]).to_csv(output_dir / "model_comparison.csv", index=False)
    pd.DataFrame([{"model_id": m["id"], **fold} for m in models for fold in m["folds"]]).to_csv(output_dir / "fold_scores.csv", index=False)
    pd.DataFrame([{"model_id": m["id"], **metrics} for m in models for metrics in m["class_metrics"]]).to_csv(output_dir / "class_metrics.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output_dir / "knn_sensitivity.csv", index=False)
    pd.DataFrame(result["nested_cv"]["models"]).to_csv(output_dir / "nested_cv.csv", index=False)
    pd.DataFrame(result["dataset"]["feature_summary"]).to_csv(output_dir / "feature_summary.csv", index=False)
    pd.DataFrame({"component": range(1,14), "explained_variance_ratio": result["pca"]["explained_variance_ratio"], "cumulative_variance": result["pca"]["cumulative_variance"]}).to_csv(output_dir / "explained_variance.csv", index=False)
    pd.DataFrame(result["pca"]["loadings"], columns=feature_names, index=[f"PC{i}" for i in range(1,14)]).rename_axis("component").to_csv(output_dir / "pca_loadings.csv")
    pd.DataFrame(result["pca"]["projection"]).to_csv(output_dir / "pca_projection.csv", index=False)
    predictions = pd.DataFrame({"sample_id": frame.sample_id, "class_id": y, "fold": baseline["fold_ids"]})
    for model in models:
        predictions[model["id"]] = model["predictions"]
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    split_records = [{"fold": fold, "train_sample_ids": (train+1).tolist(), "test_sample_ids": (test+1).tolist()} for fold, (train,test) in enumerate(splits,start=1)]
    write_json(output_dir / "cv_splits.json", {"seed": SEED, "n_splits": N_SPLITS, "splits": split_records})
    create_figures(result, output_dir / "figures")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    result = run(args.output_dir.resolve(), args.data_dir.resolve())
    print(f"Loaded {result['dataset']['n_samples']} samples × {result['dataset']['n_features']} features; 3 classes; no missing values.")
    print("Representation       Mean accuracy     Sample SD      Delta vs baseline")
    for model in result["models"]:
        print(f"{model['id']:20} {model['mean_accuracy']*100:8.3f}%       {model['std_accuracy']*100:6.3f} pp    {model['delta_vs_baseline']*100:+7.3f} pp")
    print(f"Artifacts saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
