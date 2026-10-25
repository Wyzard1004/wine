"""Checks aimed at the scientific validity of the experiment and saved results."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.analysis import K, ROOT, SEED, acquire_data, evaluate_fixed, make_pipeline, make_splits


@pytest.fixture(scope="module")
def observations():
    wine = load_wine()
    return wine.data, wine.target + 1


def test_acquired_csv_preserves_all_measurements_and_targets(tmp_path, observations):
    X, y = observations
    frame, actual_X, actual_y = acquire_data(tmp_path)
    loaded = pd.read_csv(tmp_path / "wine.csv")
    assert loaded.shape == (178, 15)
    np.testing.assert_array_equal(loaded.iloc[:, 2:].to_numpy(), X)
    np.testing.assert_array_equal(loaded.class_id, y)
    np.testing.assert_array_equal(loaded.sample_id, np.arange(1, 179))
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["csv_sha256"] == hashlib.sha256((tmp_path / "wine.csv").read_bytes()).hexdigest()
    assert loaded.isna().sum().sum() == 0


def test_split_partition_is_disjoint_complete_stratified_and_reproducible(observations):
    X, y = observations
    splits = make_splits(X, y)
    seen = []
    for (train, test), (other_train, other_test) in zip(splits, make_splits(X, y)):
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(range(178))
        np.testing.assert_array_equal(train, other_train)
        np.testing.assert_array_equal(test, other_test)
        np.testing.assert_allclose(np.bincount(y[test], minlength=4)[1:] / len(test),
                                   np.bincount(y, minlength=4)[1:] / len(y), atol=.035)
        seen.extend(test)
    assert sorted(seen) == list(range(178))


@pytest.mark.parametrize("components", [2, 5, 10, 13])
def test_preprocessors_fit_only_training_rows(observations, components):
    X, y = observations
    train, test = make_splits(X, y)[0]
    pipeline = make_pipeline(components).fit(X[train], y[train])
    scaler = pipeline.named_steps["scale"]
    assert scaler.n_samples_seen_ == len(train)
    np.testing.assert_allclose(scaler.mean_, X[train].mean(axis=0))
    assert not np.allclose(scaler.mean_, X.mean(axis=0))
    if components != 13:
        expected = PCA(n_components=components, svd_solver="full").fit(
            StandardScaler().fit_transform(X[train]))
        np.testing.assert_allclose(pipeline.named_steps["pca"].components_, expected.components_)


def test_held_out_values_do_not_change_training_transform(observations):
    X, y = observations
    train, test = make_splits(X, y)[0]
    original = make_pipeline(5).fit(X[train], y[train])
    changed = X.copy()
    changed[test] += 1e6
    refitted = make_pipeline(5).fit(changed[train], y[train])
    np.testing.assert_array_equal(original.named_steps["scale"].mean_, refitted.named_steps["scale"].mean_)
    np.testing.assert_array_equal(original.named_steps["pca"].components_, refitted.named_steps["pca"].components_)


def test_saved_results_match_predictions_and_shared_folds(observations):
    X, y = observations
    result = json.loads((ROOT / "results/analysis.json").read_text())
    predictions = pd.read_csv(ROOT / "results/oof_predictions.csv")
    reference_fold_ids = result["models"][0]["fold_ids"]
    for model in result["models"]:
        actual = evaluate_fixed(X, y, make_splits(X, y), model["components"], K)
        assert model["fold_ids"] == reference_fold_ids
        np.testing.assert_array_equal(model["predictions"], predictions[model["id"]])
        np.testing.assert_allclose(model["fold_scores"], actual["fold_scores"])
        assert model["std_accuracy"] == pytest.approx(np.std(model["fold_scores"], ddof=1))
        np.testing.assert_array_equal(model["confusion_matrix"], confusion_matrix(y, model["predictions"], labels=[1,2,3]))
        matrix = np.asarray(model["confusion_matrix"])
        assert matrix.sum() == 178
        assert matrix.trace()/178 == pytest.approx(model["oof_accuracy"])
        assert sum(m["support"] for m in model["class_metrics"]) == 178


def test_exported_pca_descriptive_scope_and_variance():
    result = json.loads((ROOT / "results/analysis.json").read_text())
    pca = result["pca"]
    assert "descriptive" in pca["scope"]
    assert len(pca["projection"]) == 178
    assert len(pca["explained_variance_ratio"]) == 13
    np.testing.assert_allclose(np.cumsum(pca["explained_variance_ratio"]), pca["cumulative_variance"])
    assert pca["cumulative_variance"][-1] == pytest.approx(1)
    loadings = np.asarray(pca["loadings"])
    np.testing.assert_allclose(loadings @ loadings.T, np.eye(13), atol=1e-12)


def test_nested_selection_and_sensitivity_have_valid_outputs():
    result = json.loads((ROOT / "results/analysis.json").read_text())
    sensitivity = result["knn_sensitivity"]
    assert len(sensitivity) == 24
    for row in sensitivity:
        assert 0 <= row["mean_accuracy"] <= 1
        if row["k"] == 5:
            model = next(m for m in result["models"] if m["id"] == row["model_id"])
            assert row["mean_accuracy"] == model["mean_accuracy"]
    for model in result["nested_cv"]["models"]:
        assert len(model["fold_scores"]) == len(model["selected_k"]) == 5
        assert set(model["selected_k"]) <= {1,3,5,7,9,11}
        assert model["mean_accuracy"] == pytest.approx(np.mean(model["fold_scores"]))


def test_nested_search_never_receives_outer_test_rows(monkeypatch, observations):
    import src.analysis as analysis

    X, y = observations
    splits = make_splits(X, y)
    real_search = analysis.GridSearchCV
    visited = []

    class AuditedSearch:
        def __init__(self, estimator, param_grid, **kwargs):
            self.search = real_search(estimator, param_grid, **kwargs)

        def fit(self, train_X, train_y):
            train, test = splits[len(visited) % 5]
            np.testing.assert_array_equal(train_X, X[train])
            np.testing.assert_array_equal(train_y, y[train])
            assert not hasattr(self.search.estimator.named_steps["scale"], "mean_")
            visited.append((train, test))
            self.search.fit(train_X, train_y)
            assert self.search.best_estimator_.named_steps["scale"].n_samples_seen_ == len(train)
            self.best_params_ = self.search.best_params_
            return self

        def score(self, test_X, test_y):
            _, test = visited[-1]
            np.testing.assert_array_equal(test_X, X[test])
            np.testing.assert_array_equal(test_y, y[test])
            return self.search.score(test_X, test_y)

    monkeypatch.setattr(analysis, "GridSearchCV", AuditedSearch)
    nested = analysis.evaluate_nested(X, y, splits)
    assert len(visited) == 20
    saved = json.loads((ROOT / "results/analysis.json").read_text())["nested_cv"]
    assert nested == saved
