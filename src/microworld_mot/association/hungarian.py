from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _greedy_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates: list[tuple[float, int, int]] = []
    for row in range(cost.shape[0]):
        for column in range(cost.shape[1]):
            if np.isfinite(cost[row, column]):
                candidates.append((float(cost[row, column]), row, column))
    candidates.sort()
    used_rows: set[int] = set()
    used_columns: set[int] = set()
    selected_rows: list[int] = []
    selected_columns: list[int] = []
    for _, row, column in candidates:
        if row not in used_rows and column not in used_columns:
            used_rows.add(row)
            used_columns.add(column)
            selected_rows.append(row)
            selected_columns.append(column)
    return np.asarray(selected_rows), np.asarray(selected_columns)


def linear_assignment(
    cost: np.ndarray,
    threshold: float = float("inf"),
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost.ndim != 2:
        raise ValueError(f"cost must be two-dimensional, got shape={cost.shape}")
    rows, columns = cost.shape
    if rows == 0 or columns == 0:
        return [], list(range(rows)), list(range(columns))
    safe_cost = np.where(np.isfinite(cost), cost, 1e9)
    try:
        from scipy.optimize import linear_sum_assignment

        row_indices, column_indices = linear_sum_assignment(safe_cost)
    except ImportError:
        row_indices, column_indices = _greedy_assignment(safe_cost)

    matches = [
        (int(row), int(column))
        for row, column in zip(row_indices, column_indices, strict=False)
        if np.isfinite(cost[row, column]) and cost[row, column] <= threshold
    ]
    matched_rows = {row for row, _ in matches}
    matched_columns = {column for _, column in matches}
    unmatched_rows = [row for row in range(rows) if row not in matched_rows]
    unmatched_columns = [column for column in range(columns) if column not in matched_columns]
    return matches, unmatched_rows, unmatched_columns


def assignment_cost(cost: np.ndarray, matches: Iterable[tuple[int, int]]) -> float:
    values = [float(cost[row, column]) for row, column in matches]
    return float(np.mean(values)) if values else 0.0

