"""Strictly causal completed-event targets for scheduled Event Queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CompletedEventTargets:
    """Fixed-capacity targets; class 0 is reserved for no-event predictions."""

    query_times: np.ndarray
    previous_query_times: np.ndarray
    classes: np.ndarray
    boundaries: np.ndarray
    left_censored: np.ndarray
    valid: np.ndarray
    boundary_mask: np.ndarray
    tiou_valid: np.ndarray
    event_ids: np.ndarray
    global_intervals: np.ndarray

    @property
    def target_counts(self) -> np.ndarray:
        return self.valid.sum(axis=1, dtype=np.int64)


def query_times_for_sequence(length: int, stride: int) -> np.ndarray:
    """Return causal prefix lengths, including one final partial-bin flush."""

    frames = int(length)
    step = int(stride)
    if frames <= 0 or step <= 0:
        raise ValueError("length and stride must be positive")
    values = list(range(step, frames + 1, step))
    if not values or values[-1] != frames:
        values.append(frames)
    return np.asarray(values, dtype=np.int64)


def _field(annotation: Any, name: str) -> int:
    if isinstance(annotation, Mapping):
        return int(annotation[name])
    return int(getattr(annotation, name))


def build_completed_event_targets(
    annotations: Sequence[Any],
    query_times: Sequence[int],
    *,
    memory_length: int,
    max_targets: int,
    previous_query_time: int = 0,
    stream_start: int = 0,
) -> CompletedEventTargets:
    """Assign each event to the first query whose observed prefix contains its end.

    For consecutive query prefix lengths ``previous < current``, a target is owned
    exactly once when ``previous < end_frame_exclusive <= current``.  This also
    handles a final query whose interval is shorter than the nominal stride.
    """

    times = np.asarray(query_times, dtype=np.int64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("query_times must be a non-empty one-dimensional sequence")
    if np.any(times <= 0) or np.any(times[1:] <= times[:-1]):
        raise ValueError("query_times must be strictly increasing positive prefixes")
    capacity = int(max_targets)
    window = int(memory_length)
    if capacity <= 0 or window <= 0:
        raise ValueError("max_targets and memory_length must be positive")
    first_previous = int(previous_query_time)
    reset_start = int(stream_start)
    if first_previous < 0 or first_previous >= int(times[0]):
        raise ValueError("previous_query_time must precede the first query")
    if reset_start < 0 or reset_start > first_previous:
        raise ValueError("stream_start must be in [0, previous_query_time]")

    classes = np.full((times.size, capacity), -1, dtype=np.int64)
    boundaries = np.zeros((times.size, capacity, 2), dtype=np.float32)
    censored = np.zeros((times.size, capacity), dtype=np.bool_)
    valid = np.zeros((times.size, capacity), dtype=np.bool_)
    boundary_mask = np.zeros((times.size, capacity, 2), dtype=np.bool_)
    tiou_valid = np.zeros((times.size, capacity), dtype=np.bool_)
    event_ids = np.full((times.size, capacity), -1, dtype=np.int64)
    global_intervals = np.full((times.size, capacity, 2), -1, dtype=np.int64)
    previous_times = np.empty(times.size, dtype=np.int64)
    normalized = []
    for event_id, annotation in enumerate(annotations):
        start = _field(annotation, "start_frame")
        end = _field(annotation, "end_frame_exclusive")
        class_id = _field(annotation, "class_id")
        if not 0 <= class_id < 13 or not 0 <= start < end:
            raise ValueError(f"invalid annotation class={class_id}, interval=[{start},{end})")
        # An event crossing a long-gap reset is cancelled, not converted into a
        # finite-memory left-censored target.
        if start < reset_start < end:
            continue
        normalized.append((start, end, class_id, event_id))

    previous = first_previous
    assigned_ends = set()
    for query_index, current in enumerate(times.tolist()):
        previous_times[query_index] = previous
        owned = [item for item in normalized if previous < item[1] <= current]
        if len(owned) > capacity:
            raise ValueError(
                f"query t={current} owns {len(owned)} events but capacity is {capacity}"
            )
        conceptual_left = current - window
        visible_left = max(reset_start, 0, conceptual_left)
        scale = float(window)
        for slot, (start, end, class_id, event_id) in enumerate(owned):
            identity = (start, end, class_id)
            if identity in assigned_ends:
                raise AssertionError("completed event assigned to more than one query")
            assigned_ends.add(identity)
            is_censored = start < visible_left
            classes[query_index, slot] = class_id + 1
            boundaries[query_index, slot, 0] = (
                0.0 if is_censored else float(start - conceptual_left) / scale
            )
            boundaries[query_index, slot, 1] = float(end - conceptual_left) / scale
            censored[query_index, slot] = is_censored
            valid[query_index, slot] = True
            boundary_mask[query_index, slot] = (not is_censored, True)
            tiou_valid[query_index, slot] = not is_censored
            event_ids[query_index, slot] = event_id
            global_intervals[query_index, slot] = (start, end)
        previous = current
    return CompletedEventTargets(
        query_times=times,
        previous_query_times=previous_times,
        classes=classes,
        boundaries=boundaries,
        left_censored=censored,
        valid=valid,
        boundary_mask=boundary_mask,
        tiou_valid=tiou_valid,
        event_ids=event_ids,
        global_intervals=global_intervals,
    )


__all__ = [
    "CompletedEventTargets",
    "build_completed_event_targets",
    "query_times_for_sequence",
]
