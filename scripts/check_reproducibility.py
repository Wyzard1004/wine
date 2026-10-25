"""Compare regenerated scientific outputs to the committed reference.

Allows floating point differences between numerical libraries on different hosts,
and a different Python patch version. Requires identical observations, predictions,
fold assignments, selected neighbours, and all numerical results within 1e-10.
"""

import csv
import io
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def compare(expected, actual, location):
    if location.endswith("metadata.versions.python"):
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), location
        for key, value in expected.items():
            compare(value, actual[key], f"{location}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), location
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{location}[{index}]")
    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)), location
        assert math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10), (
            f"{location}: expected {expected}, got {actual}"
        )
    else:
        assert expected == actual, f"{location}: expected {expected!r}, got {actual!r}"


def cell(value):
    try:
        return float(value)
    except ValueError:
        return value


def main():
    tracked = subprocess.check_output(
        ["git", "ls-files", "data", "results"], cwd=ROOT, text=True
    ).splitlines()
    paths = [p for p in tracked if p.endswith((".json", ".csv"))]
    if not paths:
        raise SystemExit("Commit the reference data and results before running this check.")
    for path in paths:
        expected_text = subprocess.check_output(
            ["git", "show", f"HEAD:{path}"], cwd=ROOT, text=True
        )
        actual_text = (ROOT / path).read_text(encoding="utf-8")
        if path.endswith(".json"):
            compare(json.loads(expected_text), json.loads(actual_text), path)
        else:
            expected = [[cell(c) for c in row] for row in csv.reader(io.StringIO(expected_text))]
            actual = [[cell(c) for c in row] for row in csv.reader(io.StringIO(actual_text))]
            compare(expected, actual, path)
    print(f"Reproduced {len(paths)} data and result files (numerical tolerance 1e-10).")


if __name__ == "__main__":
    main()
