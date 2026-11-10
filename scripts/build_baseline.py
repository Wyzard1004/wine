"""Reproduce the revised checkpoint's repeated nested-CV study and PNG.

Run: python3 scripts/build_baseline.py
Only baseline.png is written in the checkout. The complete numerical audit is
written to --audit-json (default /tmp/wine-repeated-baseline.json). baseline.txt
is the short accompanying assignment response and is maintained separately.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wine-matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SEED, REPEATS, FOLDS = 42, 10, 5
K_GRID = list(range(1, 26, 2))
REPRESENTATIONS = [0, *range(1, 14)]  # 0 means all original standardized inputs.


def pipeline(r: int) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if r:
        steps.append(("pca", PCA(n_components=r, svd_solver="full", whiten=False)))
    steps.append(("knn", KNeighborsClassifier(weights="uniform", metric="euclidean")))
    return Pipeline(steps)


def evaluate_outer(index, train, test, X, y):
    assert not np.intersect1d(train, test).size
    inner = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+index+1).split(X[train], y[train]))
    rows = []
    for r in REPRESENTATIONS:
        search = GridSearchCV(pipeline(r), {"knn__n_neighbors": K_GRID},
                              scoring="accuracy", cv=inner, refit=True, n_jobs=1)
        search.fit(X[train], y[train])
        scaler = search.best_estimator_.named_steps["scale"]
        assert scaler.n_samples_seen_ == len(train)
        np.testing.assert_allclose(scaler.mean_, X[train].mean(axis=0))
        rows.append({"r": r, "selected_k": int(search.best_params_["knn__n_neighbors"]),
                     "predictions": search.predict(X[test]).tolist()})
    # With all PCs retained, an unwhitened orthogonal rotation preserves distances.
    assert rows[0]["predictions"] == rows[-1]["predictions"]
    assert rows[0]["selected_k"] == rows[-1]["selected_k"]
    return {"index": index, "train_ids": (train+1).tolist(), "test_ids": (test+1).tolist(), "models": rows}


def run(workers: int) -> dict:
    wine = load_wine()
    X, y = wine.data, wine.target+1
    assert X.shape == (178, 13) and np.isfinite(X).all()
    np.testing.assert_array_equal(np.bincount(y)[1:], [59, 71, 48])
    outer = list(RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED).split(X, y))
    completed = []
    # Shared data are read-only; every search owns its pipeline and CV fits.
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(evaluate_outer, i, train, test, X, y) for i, (train, test) in enumerate(outer)]
        for future in as_completed(pending):
            completed.append(future.result())
            if len(completed) % 5 == 0:
                print(f"Completed {len(completed)}/50 matched outer folds.", flush=True)
    completed.sort(key=lambda row: row["index"])
    predictions = np.full((14, REPEATS, len(y)), -1, dtype=int)
    selected = np.zeros((14, REPEATS, FOLDS), dtype=int)
    coverage = np.zeros((REPEATS, len(y)), dtype=int)
    for fold in completed:
        repeat, fold_number = divmod(fold["index"], FOLDS)
        test = np.asarray(fold["test_ids"])-1
        coverage[repeat, test] += 1
        for model in fold["models"]:
            r = model["r"]
            predictions[r, repeat, test] = model["predictions"]
            selected[r, repeat, fold_number] = model["selected_k"]
    assert np.all(coverage == 1) and np.all(predictions > 0)
    accuracies = np.mean(predictions == y, axis=2)
    baseline = accuracies[0]
    summaries = []
    for r in REPRESENTATIONS:
        scores = accuracies[r]
        paired = scores-baseline  # Error_baseline - error_r; positive is improvement.
        summaries.append({
            "r": r, "label": "All 13 original features" if r == 0 else f"{r} PCs",
            "repeat_accuracy": scores.tolist(), "repeat_correct": np.sum(predictions[r] == y, axis=1).tolist(),
            "mean_accuracy": float(scores.mean()), "mean_error": float(1-scores.mean()),
            "repeat_sd": float(scores.std(ddof=1)), "mc_se": float(scores.std(ddof=1)/np.sqrt(REPEATS)),
            "paired_error_reduction": paired.tolist(), "mean_paired_error_reduction": float(paired.mean()),
            "paired_repeat_sd": float(paired.std(ddof=1)), "paired_mc_se": float(paired.std(ddof=1)/np.sqrt(REPEATS)),
            "selected_k": selected[r].tolist(),
            "selected_k_counts": dict(sorted(Counter(selected[r].ravel().tolist()).items())),
            "predictions": predictions[r].tolist(),
        })
    return {"metadata": {"seed": SEED, "repeats": REPEATS, "outer_folds": FOLDS, "inner_folds": 5,
                         "inner_seed_range": [43, 92], "k_grid": K_GRID,
                         "repeat_summary": "Pool all 178 held-out predictions within each complete repeat; summarize ten repeat accuracies.",
                         "uncertainty": "Sample SD across repeats describes partition/tuning-split sensitivity. MC SE=SD/sqrt(10) describes the estimated partition average conditional on these wines, not population uncertainty.",
                         "source": "https://archive.ics.uci.edu/dataset/109/wine"},
            "models": summaries, "folds": completed}


def plot(result: dict, output: Path):
    baseline, *models = result["models"]
    bg, ink, muted, wine, sage = "#f7f6f2", "#292b29", "#626d5b", "#78384b", "#506e5d"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": ink, "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted})
    fig = plt.figure(figsize=(11.8, 9.2), facecolor=bg)
    fig.text(.08, .95, "WINE / REPEATED VALIDATION", color=wine, fontsize=10, weight="bold", va="top")
    fig.text(.08, .9, "Do fewer dimensions help?", family="DejaVu Serif", fontsize=29, va="top")
    fig.text(.08, .838, "10 repeats × 5 outer folds · k tuned separately for every representation", color=muted, fontsize=11)
    fig.text(.08, .787, f"All-feature baseline  {baseline['mean_accuracy']*100:.2f}%", color=sage, fontsize=14, weight="bold")
    fig.text(.59, .79, f"Repeat SD {baseline['repeat_sd']*100:.2f} pp  /  MC SE {baseline['mc_se']*100:.2f} pp", color=muted, fontsize=10)
    ax = fig.add_axes([.095, .405, .82, .315], facecolor=bg)
    x = np.arange(1, 14)
    scores = np.asarray([m["mean_accuracy"] for m in models])*100
    spread = np.asarray([m["repeat_sd"] for m in models])*100
    base, base_sd = baseline["mean_accuracy"]*100, baseline["repeat_sd"]*100
    ax.axhspan(base-base_sd, base+base_sd, color=sage, alpha=.13, label="All features: mean ±1 repeat SD")
    ax.axhline(base, color=sage, linestyle="--", lw=1.5)
    ax.errorbar(x, scores, yerr=spread, fmt="o-", color=wine, lw=1.7, capsize=3,
                markersize=5.5, label="PCA + k-NN: mean ±1 repeat SD")
    ax.set(xlim=(.65, 13.35), xticks=x, ylabel="Held-out accuracy (%)", xlabel="Number of principal components (r)")
    ax.set_ylim(max(0, np.floor(min(scores-spread)-2)), min(102, max(100, max(scores+spread)+1)))
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    best = max(models, key=lambda model: model["mean_accuracy"])
    best_rs = [model["r"] for model in models if np.isclose(model["mean_accuracy"], best["mean_accuracy"])]
    best_label = "–".join(map(str, best_rs))
    ax.annotate(f"{best_label} PCs: {best['mean_accuracy']*100:.2f}%",
                xy=(np.mean(best_rs), best["mean_accuracy"]*100), xytext=(5.1, 99.5),
                color=wine, fontsize=9)
    ax.grid(axis="y", alpha=.17)
    ax.spines[["top", "right"]].set_visible(False)
    lower = fig.add_axes([.095, .16, .82, .15], facecolor=bg)
    # Show the near-baseline comparisons in detail; r=1 remains in the main panel.
    detail_x = x[1:]
    delta = np.asarray([m["mean_paired_error_reduction"] for m in models[1:]])*100
    delta_sd = np.asarray([m["paired_repeat_sd"] for m in models[1:]])*100
    lower.axhline(0, color=sage, linestyle="--", lw=1.3)
    lower.errorbar(detail_x, delta, yerr=delta_sd, fmt="o-", color=wine, lw=1.4, capsize=3, markersize=4)
    lower.set(xlim=(1.65, 13.35), xticks=detail_x, xlabel="Number of principal components (r)", ylabel="Paired error reduction\n(percentage points)")
    lower.set_title("Paired detail, r = 2–13 · positive values favour PCA · whiskers ±1 repeat SD", loc="left", fontsize=10, color=muted, pad=12)
    lower.grid(axis="y", alpha=.17)
    lower.spines[["top", "right"]].set_visible(False)
    fig.text(.08, .07, "Inner 5-fold CV: odd k = 1–25. Scaling, PCA and tuning remain inside training folds.", color=muted, fontsize=9)
    fig.text(.08, .045, "Repeat SD / MC SE describe partition sensitivity on the same 178 wines; neither is a population confidence interval.", color=muted, fontsize=9)
    fig.text(.08, .022, f"1 PC reduces accuracy by {-models[0]['mean_paired_error_reduction']*100:.2f} pp; 13 PCs match the baseline. One wine = 0.56 pp per repeat.", color=muted, fontsize=9)
    fig.savefig(output, dpi=220, facecolor=bg, metadata={"Software": "Wine repeated nested-CV checkpoint"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit-json", type=Path, default=Path("/tmp/wine-repeated-baseline.json"))
    parser.add_argument("--from-json", type=Path, help="Replot a previously generated audit without rerunning CV.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = json.loads(args.from_json.read_text()) if args.from_json else run(args.workers)
    args.audit_json.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plot(result, ROOT/"baseline.png")
    for model in result["models"]:
        print(f"{model['label']:25} accuracy={model['mean_accuracy']*100:.4f}% SD={model['repeat_sd']*100:.4f}pp SE={model['mc_se']*100:.4f}pp paired={model['mean_paired_error_reduction']*100:+.4f}pp ±SD{model['paired_repeat_sd']*100:.4f}")
    print(f"Audit: {args.audit_json}; figure: {ROOT/'baseline.png'}")


if __name__ == "__main__":
    main()
