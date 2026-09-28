from __future__ import annotations

import numpy as np

from .hungarian import assignment_cost, linear_assignment


def candidate_assignments(
    cost: np.ndarray,
    max_hypotheses: int,
    threshold: float,
) -> list[list[tuple[int, int]]]:
    """Produce a small deterministic set of near-optimal association hypotheses.

    The first hypothesis is Hungarian. Additional hypotheses forbid one edge from
    the best assignment. This is deliberately bounded; the world model is only
    invoked for ambiguous components rather than enumerating all permutations.
    """
    base, _, _ = linear_assignment(cost, threshold)
    hypotheses: list[list[tuple[int, int]]] = [base]
    seen = {tuple(sorted(base))}
    alternatives: list[tuple[float, list[tuple[int, int]]]] = []
    for row, column in base:
        modified = cost.copy()
        modified[row, column] = np.inf
        candidate, _, _ = linear_assignment(modified, threshold)
        key = tuple(sorted(candidate))
        if key not in seen:
            alternatives.append((assignment_cost(cost, candidate), candidate))
            seen.add(key)
    alternatives.sort(key=lambda item: item[0])
    hypotheses.extend(candidate for _, candidate in alternatives[: max(0, max_hypotheses - 1)])
    return hypotheses

