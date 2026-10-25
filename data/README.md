# UCI Wine data

`wine.csv` contains **178 samples × 13 chemical features**, with three cultivar classes
(59, 71, and 48 samples). It has no missing measurements. The CSV also includes
`sample_id` (1–178, an identifier only) and `class_id` (1, 2, or 3, the target).
Neither identifier nor target is supplied to the classifier.

The observations are the UCI Wine dataset, acquired through scikit-learn's bundled
`sklearn.datasets.load_wine` copy. Acquisition and every analysis step work offline
after dependencies are installed. Chemical measurements are unchanged; scikit-learn's
0-based target is relabelled 1–3, and an explicit sample identifier is added.
`metadata.json` records provenance and a SHA-256 checksum of this CSV.
The original UCI `wine.data` archive was independently checked on 24 October 2026:
all 178 × 13 chemical values and all class labels match this bundled copy exactly.

The target denotes **cultivars** of wines from the same region in Italy, rather than
wine quality ratings, red/white labels, or named grape varieties. No cultivar names
are provided; the dashboard uses Cultivar 1, Cultivar 2, and Cultivar 3.

Source: [UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine).

Citation: Aeberhard, S. & Forina, M. (1992). *Wine* [Dataset]. UCI Machine Learning
Repository. [https://doi.org/10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J).

License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Attribution follows the UCI repository metadata; some scikit-learn releases contain
an unrelated citation in the human-readable Wine description.

Recreate the data and results from the repository root with `python3 -m src.analysis`.
