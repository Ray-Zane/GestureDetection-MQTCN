"""Causal Background/Candidate/Active/Ending continuous-event decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np

from streaming.event_tracker import ContinuousEvent, EventTracker


@dataclass(frozen=True)
class DecoderConfig:
    ema_alpha: float = 0.6
    activation_threshold: float = 0.45
    sustain_threshold: float = 0.25
    background_threshold: float = 0.60
    boundary_threshold: float = 0.60
    candidate_frames: int = 3
    end_confirm_frames: int = 3
    switch_confirm_frames: int = 3
    min_event_frames: int = 8
    cooldown_frames: int = 6
    dedup_gap_frames: int = 8
    dedup_tiou: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "ema_alpha",
            "activation_threshold",
            "sustain_threshold",
            "background_threshold",
            "boundary_threshold",
            "dedup_tiou",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if min(
            self.candidate_frames,
            self.end_confirm_frames,
            self.switch_confirm_frames,
            self.min_event_frames,
        ) <= 0:
            raise ValueError("decoder frame counts must be positive")


class StreamingDecoder:
    """Consume one frame at a time; gesture probability columns are 1..13."""

    def __init__(self, config: Optional[DecoderConfig] = None) -> None:
        self.config = config or DecoderConfig()
        self.tracker = EventTracker(
            min_event_frames=self.config.min_event_frames,
            cooldown_frames=self.config.cooldown_frames,
            dedup_gap_frames=self.config.dedup_gap_frames,
            dedup_tiou=self.config.dedup_tiou,
        )
        self.reset()

    def reset(self, *, start_frame: int = 0) -> None:
        start = int(start_frame)
        if start < 0:
            raise ValueError("start_frame must be non-negative")
        self.state = "Background"
        self.ema: Optional[np.ndarray] = None
        self.candidate_class: Optional[int] = None
        self.candidate_start = 0
        self.candidate_count = 0
        self.active_class: Optional[int] = None
        self.active_start = 0
        self.active_scores: List[float] = []
        self.ending_start = 0
        self.ending_count = 0
        self.switch_class: Optional[int] = None
        self.switch_start = 0
        self.switch_count = 0
        self.last_frame = start - 1
        self.tracker.reset()

    def _activate(self, class_id: int, start: int, score: float) -> None:
        self.state = "Active"
        self.active_class = int(class_id)
        self.active_start = int(start)
        self.active_scores = [float(score)]
        self.candidate_class = None
        self.candidate_count = 0
        self.ending_count = 0
        self.switch_class = None
        self.switch_count = 0

    def _finalize(self, end_exclusive: int, emitted_at: int) -> List[ContinuousEvent]:
        emitted: List[ContinuousEvent] = []
        if self.active_class is not None and end_exclusive > self.active_start:
            event = ContinuousEvent(
                class_id=self.active_class,
                start_frame=self.active_start,
                end_frame_exclusive=int(end_exclusive),
                score=float(np.mean(self.active_scores)) if self.active_scores else 0.0,
                emitted_at_frame=max(int(emitted_at), int(end_exclusive) - 1),
            )
            accepted = self.tracker.add(event)
            if accepted is not None:
                emitted.append(accepted)
        self.state = "Background"
        self.active_class = None
        self.active_scores = []
        self.ending_count = 0
        self.switch_class = None
        self.switch_count = 0
        return emitted

    def step(
        self,
        frame_probabilities: Sequence[float],
        start_probability: float,
        end_probability: float,
        frame_index: int,
    ) -> List[ContinuousEvent]:
        frame = int(frame_index)
        if frame != self.last_frame + 1:
            raise ValueError("decoder frames must be contiguous from zero")
        self.last_frame = frame
        probability = np.asarray(frame_probabilities, dtype=np.float64)
        if probability.shape != (14,) or not np.isfinite(probability).all():
            raise ValueError("frame probabilities must be finite shape [14]")
        if self.ema is None:
            self.ema = probability.copy()
        else:
            alpha = self.config.ema_alpha
            self.ema = alpha * probability + (1.0 - alpha) * self.ema
        gesture_column = int(np.argmax(self.ema[1:])) + 1
        gesture_class = gesture_column - 1
        gesture_score = float(self.ema[gesture_column])
        background_score = float(self.ema[0])
        confident_gesture = (
            gesture_score >= self.config.activation_threshold
            and gesture_score > background_score
        )
        emitted: List[ContinuousEvent] = []

        if self.state == "Background":
            if confident_gesture:
                self.state = "Candidate"
                self.candidate_class = gesture_class
                self.candidate_start = frame
                self.candidate_count = 1
            return emitted

        if self.state == "Candidate":
            if confident_gesture and gesture_class == self.candidate_class:
                self.candidate_count += 1
                if self.candidate_count >= self.config.candidate_frames:
                    self._activate(
                        int(self.candidate_class), self.candidate_start, gesture_score
                    )
            elif confident_gesture:
                self.candidate_class = gesture_class
                self.candidate_start = frame
                self.candidate_count = 1
            else:
                self.state = "Background"
                self.candidate_class = None
                self.candidate_count = 0
            return emitted

        if self.active_class is None:
            raise AssertionError("Active/Ending state without active class")
        current_score = float(self.ema[self.active_class + 1])
        self.active_scores.append(current_score)

        different = confident_gesture and gesture_class != self.active_class
        if different:
            if self.switch_class == gesture_class:
                self.switch_count += 1
            else:
                self.switch_class = gesture_class
                self.switch_start = frame
                self.switch_count = 1
            if self.switch_count >= self.config.switch_confirm_frames:
                switch_start = self.switch_start
                emitted.extend(self._finalize(switch_start, frame))
                self._activate(gesture_class, switch_start, gesture_score)
                return emitted
        else:
            self.switch_class = None
            self.switch_count = 0

        active_duration = frame + 1 - self.active_start
        should_end = active_duration >= self.config.min_event_frames and (
            float(end_probability) >= self.config.boundary_threshold
            or background_score >= self.config.background_threshold
            or current_score < self.config.sustain_threshold
        )
        if self.state == "Active" and should_end:
            self.state = "Ending"
            self.ending_start = frame
            self.ending_count = 1
        elif self.state == "Ending":
            recovered = (
                current_score >= self.config.activation_threshold
                and background_score < self.config.background_threshold
                and float(end_probability) < self.config.boundary_threshold
            )
            if recovered:
                self.state = "Active"
                self.ending_count = 0
            else:
                self.ending_count += 1
                if self.ending_count >= self.config.end_confirm_frames:
                    emitted.extend(self._finalize(self.ending_start + 1, frame))
        return emitted

    def finish(self) -> List[ContinuousEvent]:
        if self.last_frame < 0:
            return []
        if self.state in {"Active", "Ending"}:
            end = self.ending_start + 1 if self.state == "Ending" else self.last_frame + 1
            return self._finalize(end, self.last_frame)
        return []

    def decode(
        self,
        frame_probabilities: np.ndarray,
        start_probabilities: np.ndarray,
        end_probabilities: np.ndarray,
    ) -> List[Mapping[str, object]]:
        frames = np.asarray(frame_probabilities, dtype=np.float64)
        starts = np.asarray(start_probabilities, dtype=np.float64)
        ends = np.asarray(end_probabilities, dtype=np.float64)
        if frames.ndim != 2 or frames.shape[1] != 14:
            raise ValueError("frame_probabilities must be [T,14]")
        if starts.shape != (frames.shape[0],) or ends.shape != (frames.shape[0],):
            raise ValueError("boundary probabilities must be [T]")
        self.reset(start_frame=0)
        for frame in range(frames.shape[0]):
            self.step(frames[frame], starts[frame], ends[frame], frame)
        self.finish()
        return [event.as_dict() for event in self.tracker.events]


def decoder_config_from_mapping(payload: Mapping[str, object]) -> DecoderConfig:
    fields = DecoderConfig.__dataclass_fields__
    return DecoderConfig(**{key: payload[key] for key in fields if key in payload})


__all__ = ["DecoderConfig", "StreamingDecoder", "decoder_config_from_mapping"]
