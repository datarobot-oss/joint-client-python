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

"""pandas conversions of forecast values; pandas is imported only when a caller asks for a frame."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _require_pandas_module() -> Any:
    try:
        import pandas as pandas_module
    except ImportError as error:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "pandas result conversion requires installing jointfm-client[notebooks]"
        ) from error
    return pandas_module


def _tidy_frame(
    values: np.ndarray,
    *,
    query_times: Sequence[Any],
    requested_columns: Sequence[str],
    leading_name: str | None = None,
    leading_labels: Sequence[Any] = (),
) -> Any:
    """Return one row per ``(leading, query_time, requested_column)`` cell.

    ``values`` has axis order ``(leading, horizon, column)``. Without a
    ``leading_name`` the leading axis must have length one and no leading
    column is emitted, which is the mean forecast's shape. Label columns take
    pandas' own type inference over the labels, as rows built from Python
    values do.
    """
    pandas_module = _require_pandas_module()
    leading_count, horizon_count, column_count = values.shape
    cells_per_leading = horizon_count * column_count
    flat_cells = np.arange(leading_count * cells_per_leading)
    columns: dict[str, Any] = {}
    if leading_name is not None:
        columns[leading_name] = _label_column(
            pandas_module, leading_labels, flat_cells // cells_per_leading
        )
    columns["query_time"] = _label_column(
        pandas_module, query_times, (flat_cells // column_count) % horizon_count
    )
    columns["requested_column"] = _label_column(
        pandas_module, requested_columns, flat_cells % column_count
    )
    columns["value"] = values.reshape(-1)
    return pandas_module.DataFrame(columns)


def _wide_frame(
    values: np.ndarray,
    *,
    query_times: Sequence[Any],
    requested_columns: Sequence[str],
    leading_name: str | None = None,
    leading_labels: Sequence[Any] = (),
) -> Any:
    """Return one row per ``(leading, query_time)`` with one column per requested column.

    Axis order and the meaning of ``leading_name`` match :func:`_tidy_frame`.
    Raises ``ValueError`` when a requested column is named like an index
    column, which would otherwise overwrite it.
    """
    pandas_module = _require_pandas_module()
    leading_count, horizon_count, column_count = values.shape
    flat_rows = np.arange(leading_count * horizon_count)
    columns: dict[str, Any] = {}
    if leading_name is not None:
        columns[leading_name] = _label_column(
            pandas_module, leading_labels, flat_rows // horizon_count
        )
    columns["query_time"] = _label_column(
        pandas_module, query_times, flat_rows % horizon_count
    )
    clashing = sorted(set(columns) & set(requested_columns))
    if clashing:
        raise ValueError(
            f"requested columns {clashing!r} clash with the wide frame's index columns"
        )
    table = values.reshape(leading_count * horizon_count, column_count)
    for column_index, column_name in enumerate(requested_columns):
        columns[column_name] = table[:, column_index]
    return pandas_module.DataFrame(columns)


def _label_column(
    pandas_module: Any, labels: Sequence[Any], positions: np.ndarray
) -> Any:
    """Return ``labels`` taken at ``positions``, typed as pandas infers ``labels``."""
    return pandas_module.Series(list(labels)).take(positions).reset_index(drop=True)
