"""Levenshtein sequence metrics for per-video gesture event streams."""

from __future__ import annotations

from typing import Sequence


def edit_distance(first: Sequence[int], second: Sequence[int]) -> int:
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, start=1):
        current = [row]
        for column, right in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (int(left) != int(right)),
                )
            )
        previous = current
    return int(previous[-1])


__all__ = ["edit_distance"]
