# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Permutation feature importance helpers for ``JointFMClient.feature_importance``.

Each feature column is shuffled across history rows while its marginal is
preserved, and the permuted forecast is compared against one shared baseline
forecast. A forecast reads mostly a feature's last few values, and one shuffle
can leave similar values there by chance, so each feature is shuffled
``n_repeats`` times and its scores are averaged. Two scores come out of each
comparison: the absolute shift in forecast mean, and the centered squared
2-Wasserstein distance between the baseline and permuted sample sets (which
also catches shape changes a mean shift alone would miss).
"""

from __future__ import annotations

from collections.abc import Sequence
import random
from typing import Any, Final

import numpy as np

DEFAULT_IMPORTANCE_REPEATS: Final[int] = 5
# Distinct primes keep (feature, repeat) seeds apart for fewer than 7919
# features and 1009 repeats.
FEATURE_SEED_STRIDE: Final[int] = 1009
REPEAT_SEED_STRIDE: Final[int] = 7919


def sample_w2_distance(
    baseline: Sequence[float] | np.ndarray,
    permuted: Sequence[float] | np.ndarray,
    *,
    location_invariant: bool = True,
) -> float:
    """Centered squared 2-Wasserstein distance between two 1D sample sets.

    Both sample vectors are one-dimensional draws at the same target and
    horizon, so the distance has a closed form: sort both vectors and take the
    mean squared gap (halved to match the usual W2 cost convention). When
    ``location_invariant`` is true, both vectors are centered first so the
    score reflects spread, skew, and tail movement rather than repeating the
    mean-shift readout.
    """
    if len(baseline) == 0 or len(permuted) == 0:
        return 0.0
    if len(baseline) != len(permuted):
        raise ValueError(
            f"sample vectors must have equal length; got {len(baseline)} and {len(permuted)}"
        )

    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(permuted, dtype=np.float64)
    if location_invariant:
        left = left - left.mean()
        right = right - right.mean()
    squared_gaps = np.square(np.sort(left) - np.sort(right))
    return float(squared_gaps.mean()) / 2.0


def _is_history_row_sequence(history: Any) -> bool:
    """Return whether ``history`` is a sequence of row mappings, not a DataFrame."""
    return isinstance(history, Sequence) and not isinstance(
        history, str | bytes | bytearray
    )


def permute_history_column(history: Any, feature: str, *, seed: int) -> Any:
    """Return a copy of ``history`` with ``feature`` shuffled across rows."""
    rng = random.Random(seed)
    if _is_history_row_sequence(history):
        rows = [dict(row) for row in history]
        present_indices = [index for index, row in enumerate(rows) if feature in row]
        if not present_indices:
            raise ValueError(f"history rows are missing column {feature!r}")
        values = [rows[index][feature] for index in present_indices]
        rng.shuffle(values)
        for index, value in zip(present_indices, values, strict=True):
            rows[index][feature] = value
        return rows

    pandas_module = _require_pandas_module()
    if not isinstance(history, pandas_module.DataFrame):
        raise ValueError(
            "history must be a pandas DataFrame or a sequence of row mappings"
        )
    if feature not in history.columns:
        raise ValueError(f"history frame is missing column {feature!r}")
    permuted = history.copy()
    values = list(history[feature])
    rng.shuffle(values)
    permuted[feature] = values
    return permuted


def permutation_seed(seed: int, *, feature_index: int, repeat: int) -> int:
    """Return the shuffle seed for one repeat of one feature column."""
    return (
        seed + FEATURE_SEED_STRIDE * (feature_index + 1) + REPEAT_SEED_STRIDE * repeat
    )


def average_importance_entries(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average repeated entries of one feature, score by score.

    Every entry must come from ``feature_importance_entry`` for the same
    feature, targets, and horizons; the result has the same layout.
    """
    first = entries[0]
    averaged: dict[str, Any] = {"feature": first["feature"]}
    for score in ("mean", "distance"):
        averaged[score] = {
            target: {
                horizon: float(
                    np.mean([entry[score][target][horizon] for entry in entries])
                )
                for horizon in by_horizon
            }
            for target, by_horizon in first[score].items()
        }
    return averaged


def feature_importance_entry(
    *,
    feature: str,
    horizons: Sequence[int],
    target_columns: Sequence[str],
    baseline_samples: np.ndarray,
    permuted_samples: np.ndarray,
    baseline_columns: Sequence[str],
) -> dict[str, Any]:
    """Score one permuted feature against the shared baseline forecast.

    Both sample arrays have axis order ``(sample, horizon, column)``. Returns ``{"feature": ..., "mean": {target: {horizon: value}}, "distance":
    {target: {horizon: value}}}``, where ``mean`` is the absolute shift in
    forecast mean and ``distance`` is ``sample_w2_distance``, both indexed by
    every requested target and horizon.
    """
    mean_scores: dict[str, dict[int, float]] = {}
    distance_scores: dict[str, dict[int, float]] = {}
    for target in target_columns:
        target_index = list(baseline_columns).index(target)
        mean_by_horizon: dict[int, float] = {}
        distance_by_horizon: dict[int, float] = {}
        for horizon_index, horizon in enumerate(horizons):
            baseline_values = baseline_samples[:, horizon_index, target_index]
            permuted_values = permuted_samples[:, horizon_index, target_index]
            mean_by_horizon[horizon] = abs(
                float(permuted_values.mean()) - float(baseline_values.mean())
            )
            distance_by_horizon[horizon] = sample_w2_distance(
                baseline_values, permuted_values
            )
        mean_scores[target] = mean_by_horizon
        distance_scores[target] = distance_by_horizon
    return {"feature": feature, "mean": mean_scores, "distance": distance_scores}


def _require_pandas_module() -> Any:
    """Return pandas or raise with the SDK extra needed for DataFrame history."""
    try:
        import pandas as pandas_module
    except ImportError as error:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "pandas history support requires installing jointfm-client[notebooks]"
        ) from error
    return pandas_module


__all__ = [
    "DEFAULT_IMPORTANCE_REPEATS",
    "average_importance_entries",
    "feature_importance_entry",
    "permutation_seed",
    "permute_history_column",
    "sample_w2_distance",
]
