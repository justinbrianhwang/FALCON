"""Terminal-only localization baseline (Plan §19.1, Task T8).

Features are derived from the final recorded OutcomeState (plus the last
rounds' evaluation states for the slope and the last aggregation state for
the update norm). The nearest-centroid classifier is the weakest §19.1
method; stronger sklearn models are deliberately NOT added.
"""
from pathlib import Path

import numpy as np

from falcon.recorder import Recorder


def _round_ids(root: Path, run_id: str) -> list[int]:
    """Sorted round ids recorded under ``root/runs/<run_id>`` (metadata-independent)."""
    run_dir = Path(root) / "runs" / run_id
    ids: list[int] = []
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.startswith("round_"):
            try:
                ids.append(int(child.name[len("round_"):]))
            except ValueError:
                continue
    return sorted(ids)


def _sorted_class_keys(per_class: dict) -> list[str]:
    """Class keys in numeric order when every key parses as int, else lexicographic."""
    keys = list(per_class.keys())
    try:
        return sorted(keys, key=int)
    except ValueError:
        return sorted(keys)


def terminal_features(run_root: Path, run_id: str) -> np.ndarray:
    """Fixed-order feature vector from a run's recorded terminal state.

    Feature order (documented contract for downstream classifiers):

    1. final accuracy — ``metrics["accuracy"]`` of the last recorded round;
    2. final loss — ``metrics["loss"]`` of the last recorded round;
    3. per-class accuracies — ``per_class[k]["accuracy"]`` of the last round,
       classes ordered by :func:`_sorted_class_keys` (numeric when all keys
       are integer-like, else lexicographic);
    4. accuracy slope over the last 3 rounds — least-squares slope of
       ``metrics["accuracy"]`` against round index 0..n-1 over the last
       ``min(3, rounds)`` rounds; 0.0 when fewer than 2 rounds exist;
    5. global-update norm — L2 norm of the last round's ``aggregate`` vector.

    Returns a float64 array of shape ``(4 + num_classes,)``.
    """
    recorder = Recorder(Path(run_root), run_id)
    rounds = _round_ids(run_root, run_id)
    if not rounds:
        raise ValueError(f"no recorded rounds for run {run_id!r} under {run_root}")
    last = rounds[-1]
    final = recorder.load(last, "evaluation")

    window = rounds[-3:]
    accuracies = np.asarray(
        [recorder.load(r, "evaluation").metrics["accuracy"] for r in window],
        dtype=np.float64,
    )
    if len(accuracies) >= 2:
        slope = float(
            np.polyfit(np.arange(len(accuracies), dtype=np.float64), accuracies, 1)[0]
        )
    else:
        slope = 0.0

    aggregate = recorder.load(last, "aggregation").aggregate
    features = [
        float(final.metrics["accuracy"]),
        float(final.metrics["loss"]),
        *(
            float(final.per_class[key]["accuracy"])
            for key in _sorted_class_keys(final.per_class)
        ),
        slope,
        float(np.linalg.norm(aggregate)),
    ]
    return np.asarray(features, dtype=np.float64)


class NearestCentroidStageClassifier:
    """Nearest-centroid classifier over terminal feature vectors.

    Weakest §19.1 baseline by design; numpy only, no sklearn.

    - ``fit`` z-normalizes per feature using the training mean/std
      (population std, ddof=0); zero-variance features get scale 1.0 so they
      cannot produce NaN/inf or dominate distances. Non-finite training
      features are rejected — they would poison every centroid silently.
    - Centroids are the per-class means in normalized space; classes keep
      first-appearance order from ``y``.
    - ``predict`` returns the label of the euclidean-nearest centroid; ties
      break to the earliest class in first-appearance order (``np.argmin``).
      A non-finite feature vector is REJECTED: ``np.argmin`` over NaN
      distances returns index 0, a silent first-class bias (T8-F finding 6).
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._classes: list[str] = []
        self._centroids: np.ndarray | None = None

    def fit(
        self, X: list[np.ndarray], y: list[str]
    ) -> "NearestCentroidStageClassifier":
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} != {len(y)}")
        if not X:
            raise ValueError("cannot fit on an empty training set")
        data = np.stack([np.asarray(x, dtype=np.float64) for x in X])
        if not np.all(np.isfinite(data)):
            raise ValueError("non-finite training features")
        self._mean = data.mean(axis=0)
        std = data.std(axis=0)
        self._scale = np.where(std == 0.0, 1.0, std)  # zero-variance guard
        normalized = (data - self._mean) / self._scale
        self._classes = list(dict.fromkeys(y))  # first-appearance order
        self._centroids = np.stack(
            [
                normalized[[i for i, label in enumerate(y) if label == cls]].mean(axis=0)
                for cls in self._classes
            ]
        )
        return self

    def predict(self, x: np.ndarray) -> str:
        if self._centroids is None or self._mean is None or self._scale is None:
            raise RuntimeError("predict called before fit")
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite feature vector")
        z = (x - self._mean) / self._scale
        distances = np.linalg.norm(self._centroids - z, axis=1)
        return self._classes[int(np.argmin(distances))]
