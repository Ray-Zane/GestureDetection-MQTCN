#!/usr/bin/env python3
"""Extract frame-aligned MediaPipe hand skeletons from IPN Hand.

Input videos and frame counts come exclusively from the generated manifest.
MediaPipe Hand Landmarker runs in VIDEO mode and every decoded RGB frame maps
to exactly one row in the output artifact.  Frames without a detected hand
remain present with zero landmarks and ``valid_mask=False``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
except ImportError as error:  # Keep --help failures understandable in a wrong environment.
    print(
        "ERROR: Skeleton extraction requires numpy, opencv-python and mediapipe in the active "
        f"Python environment ({error}).",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "skeleton_raw"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "preprocessing"
MODEL_CANDIDATES = (
    PROJECT_ROOT / "assets" / "hand_landmarker.task",
    PROJECT_ROOT / "weights" / "hand_landmarker.task",
    PROJECT_ROOT / "pretrained" / "hand_landmarker.task",
    PROJECT_ROOT / "legend" / "weights" / "hand_landmarker.task",
)
LANDMARK_COUNT = 21
COORDINATE_COUNT = 3
SCHEMA_VERSION = 1
REQUIRED_ARRAYS = {
    "image_landmarks",
    "world_landmarks",
    "valid_mask",
    "handedness",
    "handedness_score",
    "frame_index",
    "timestamp_ms",
    "fps",
    "width",
    "height",
    "video_id",
    "split",
    "schema_version",
    "model_sha256",
    "extraction_config_json",
    "elapsed_seconds",
    "processing_fps",
}


class ExtractionError(RuntimeError):
    """Raised when an artifact cannot be produced without losing alignment."""


@dataclass(frozen=True)
class ManifestRecord:
    video_id: str
    split: str
    video_path: Path
    fps: float
    num_frames: int


@dataclass(frozen=True)
class ArtifactSummary:
    total_frames: int
    valid_frames: int
    fps: float
    width: int
    height: int
    elapsed_seconds: float
    processing_fps: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scalar(data: Mapping[str, Any], key: str) -> Any:
    value = data[key]
    return value.item() if getattr(value, "shape", None) == () else value


def validate_probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ExtractionError(f"--{name.replace('_', '-')} must be in [0, 1]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
        np.savez_compressed(str(temporary), **arrays)
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def resolve_model_path(configured: Optional[Path]) -> Path:
    if configured is not None:
        model_path = configured.expanduser().resolve()
        if not model_path.is_file():
            raise ExtractionError(f"MediaPipe model does not exist: {model_path}")
        return model_path
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in MODEL_CANDIDATES)
    raise ExtractionError(
        "MediaPipe hand_landmarker.task was not found. Pass --model-path explicitly. "
        f"Searched: {searched}"
    )


def resolve_video_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.is_file():
        return project_candidate
    return (manifest_path.parent / path).resolve()


def load_manifest(path: Path, split: str) -> List[ManifestRecord]:
    if not path.is_file():
        raise ExtractionError(f"Manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExtractionError(f"Cannot read manifest {path}: {error}") from error
    if not isinstance(payload, list):
        raise ExtractionError(f"Manifest must contain a JSON list: {path}")

    records: List[ManifestRecord] = []
    seen: set[str] = set()
    required = {"video_id", "split", "video_path", "fps", "num_frames", "segments"}
    for index, raw_record in enumerate(payload):
        if not isinstance(raw_record, dict):
            raise ExtractionError(f"Manifest record {index} is not an object")
        missing = sorted(required - set(raw_record))
        if missing:
            raise ExtractionError(f"Manifest record {index} is missing fields: {missing}")
        video_id = str(raw_record["video_id"])
        record_split = str(raw_record["split"])
        if not video_id:
            raise ExtractionError(f"Manifest record {index} has an empty video_id")
        if video_id in seen:
            raise ExtractionError(f"Duplicate manifest video_id: {video_id}")
        if record_split != split:
            raise ExtractionError(
                f"Manifest record {video_id} has split={record_split!r}, expected {split!r}"
            )
        try:
            fps = float(raw_record["fps"])
            num_frames = int(raw_record["num_frames"])
        except (TypeError, ValueError) as error:
            raise ExtractionError(f"Invalid fps/num_frames for {video_id}") from error
        if not math.isfinite(fps) or fps <= 0:
            raise ExtractionError(f"Invalid manifest FPS for {video_id}: {fps}")
        if num_frames <= 0:
            raise ExtractionError(f"Invalid manifest frame count for {video_id}: {num_frames}")
        video_path = resolve_video_path(str(raw_record["video_path"]), path)
        if not video_path.is_file():
            raise ExtractionError(f"Video for {video_id} does not exist: {video_path}")
        records.append(ManifestRecord(video_id, split, video_path, fps, num_frames))
        seen.add(video_id)
    if not records:
        raise ExtractionError(f"Manifest contains no records: {path}")
    return records


def select_records(
    records: Sequence[ManifestRecord], requested_video_ids: Sequence[str]
) -> List[ManifestRecord]:
    if not requested_video_ids:
        return list(records)
    by_id = {record.video_id: record for record in records}
    selected: List[ManifestRecord] = []
    seen: set[str] = set()
    for requested in requested_video_ids:
        video_id = Path(requested).stem if Path(requested).suffix else requested
        if video_id in seen:
            continue
        if video_id not in by_id:
            raise ExtractionError(f"Requested video_id is not in the manifest: {video_id}")
        selected.append(by_id[video_id])
        seen.add(video_id)
    return selected


def build_extraction_config(args: argparse.Namespace, model_hash: str) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "running_mode": "VIDEO",
        "landmark_count": LANDMARK_COUNT,
        "coordinate_count": COORDINATE_COUNT,
        "image_coordinate_space": "MediaPipe normalized image coordinates",
        "world_coordinate_space": "MediaPipe world coordinates in meters",
        "missing_value": 0.0,
        "num_hands": args.num_hands,
        "hand_selection": (
            "highest handedness score" if args.num_hands > 1 else "first/only hand"
        ),
        "min_hand_detection_confidence": args.min_detection_confidence,
        "min_hand_presence_confidence": args.min_presence_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "model_sha256": model_hash,
        "mediapipe_version": str(mp.__version__),
        "opencv_version": str(cv2.__version__),
    }


def config_json(config: Mapping[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def create_landmarker(args: argparse.Namespace, model_path: Path) -> Any:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def select_hand(result: Any) -> Optional[int]:
    if not result.hand_landmarks:
        return None
    if len(result.hand_landmarks) == 1 or not result.handedness:
        return 0
    scores: List[float] = []
    for index in range(len(result.hand_landmarks)):
        categories = result.handedness[index] if index < len(result.handedness) else []
        scores.append(float(categories[0].score) if categories else -1.0)
    return max(range(len(result.hand_landmarks)), key=lambda index: scores[index])


def handedness_value(result: Any, hand_index: int) -> Tuple[str, float]:
    if hand_index >= len(result.handedness) or not result.handedness[hand_index]:
        return "", 0.0
    category = result.handedness[hand_index][0]
    label = str(category.category_name or category.display_name or "")
    return label, float(category.score)


def landmarks_to_array(landmarks: Sequence[Any], description: str) -> np.ndarray:
    if len(landmarks) != LANDMARK_COUNT:
        raise ExtractionError(
            f"MediaPipe returned {len(landmarks)} {description} landmarks; "
            f"expected {LANDMARK_COUNT}"
        )
    values = np.asarray(
        [(landmark.x, landmark.y, landmark.z) for landmark in landmarks], dtype=np.float32
    )
    if values.shape != (LANDMARK_COUNT, COORDINATE_COUNT) or not np.isfinite(values).all():
        raise ExtractionError(f"MediaPipe returned invalid {description} landmarks")
    return values


def make_timestamps(num_frames: int, fps: float) -> np.ndarray:
    timestamps = np.empty(num_frames, dtype=np.int64)
    previous = -1
    for frame_index in range(num_frames):
        timestamp = int(round(frame_index * 1000.0 / fps))
        timestamp = max(previous + 1, timestamp)
        timestamps[frame_index] = timestamp
        previous = timestamp
    return timestamps


def inspect_video_container(record: ManifestRecord, fps_tolerance: float) -> Tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(record.video_path))
    if not capture.isOpened():
        capture.release()
        raise ExtractionError(f"Cannot open video: {record.video_path}")
    try:
        container_fps = float(capture.get(cv2.CAP_PROP_FPS))
        container_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not math.isfinite(container_fps) or container_fps <= 0:
        raise ExtractionError(f"Video reports invalid FPS for {record.video_id}: {container_fps}")
    if abs(container_fps - record.fps) > fps_tolerance:
        raise ExtractionError(
            f"FPS mismatch for {record.video_id}: manifest={record.fps}, video={container_fps}"
        )
    if container_frames != record.num_frames:
        raise ExtractionError(
            f"Frame-count mismatch for {record.video_id}: "
            f"manifest={record.num_frames}, video={container_frames}"
        )
    if width <= 0 or height <= 0:
        raise ExtractionError(
            f"Video reports invalid resolution for {record.video_id}: {width}x{height}"
        )
    return container_fps, container_frames, width, height


def extract_video(
    record: ManifestRecord,
    output_path: Path,
    model_path: Path,
    extraction_config_json: str,
    model_hash: str,
    args: argparse.Namespace,
) -> ArtifactSummary:
    _, _, width, height = inspect_video_container(record, args.fps_tolerance)
    image_landmarks = np.zeros(
        (record.num_frames, LANDMARK_COUNT, COORDINATE_COUNT), dtype=np.float32
    )
    world_landmarks = np.zeros_like(image_landmarks)
    valid_mask = np.zeros(record.num_frames, dtype=np.bool_)
    handedness = np.full(record.num_frames, "", dtype="<U5")
    handedness_score = np.zeros(record.num_frames, dtype=np.float32)
    frame_index = np.arange(record.num_frames, dtype=np.int64)
    timestamp_ms = make_timestamps(record.num_frames, record.fps)

    capture = cv2.VideoCapture(str(record.video_path))
    if not capture.isOpened():
        capture.release()
        raise ExtractionError(f"Cannot open video: {record.video_path}")
    landmarker = None
    decoded_frames = 0
    started = time.perf_counter()
    try:
        landmarker = create_landmarker(args, model_path)
        while True:
            decoded, bgr_frame = capture.read()
            if not decoded:
                break
            if decoded_frames >= record.num_frames:
                raise ExtractionError(
                    f"Decoded more frames than the manifest declares for {record.video_id}"
                )
            if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
                raise ExtractionError(
                    f"Unexpected frame shape at {record.video_id}[{decoded_frames}]: "
                    f"{bgr_frame.shape}"
                )
            frame_height, frame_width = bgr_frame.shape[:2]
            if frame_width != width or frame_height != height:
                raise ExtractionError(
                    f"Resolution changed at {record.video_id}[{decoded_frames}]: "
                    f"expected {width}x{height}, got {frame_width}x{frame_height}"
                )

            rgb_frame = np.ascontiguousarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, int(timestamp_ms[decoded_frames]))
            hand_index = select_hand(result)
            if hand_index is not None:
                if hand_index >= len(result.hand_world_landmarks):
                    raise ExtractionError(
                        f"MediaPipe omitted world landmarks for {record.video_id}[{decoded_frames}]"
                    )
                image_landmarks[decoded_frames] = landmarks_to_array(
                    result.hand_landmarks[hand_index], "image"
                )
                world_landmarks[decoded_frames] = landmarks_to_array(
                    result.hand_world_landmarks[hand_index], "world"
                )
                label, score = handedness_value(result, hand_index)
                handedness[decoded_frames] = label
                handedness_score[decoded_frames] = score
                valid_mask[decoded_frames] = True

            decoded_frames += 1
            if args.progress_interval > 0 and (
                decoded_frames % args.progress_interval == 0
                or decoded_frames == record.num_frames
            ):
                elapsed = max(time.perf_counter() - started, 1e-9)
                speed = decoded_frames / elapsed
                print(
                    f"    {decoded_frames}/{record.num_frames} frames "
                    f"({decoded_frames / record.num_frames:.1%}), {speed:.1f} frame/s",
                    flush=True,
                )
    finally:
        if landmarker is not None:
            landmarker.close()
        capture.release()

    if decoded_frames != record.num_frames:
        raise ExtractionError(
            f"Decoded-frame mismatch for {record.video_id}: "
            f"expected {record.num_frames}, got {decoded_frames}"
        )

    elapsed_seconds = time.perf_counter() - started
    processing_fps = decoded_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0
    arrays: Dict[str, Any] = {
        "image_landmarks": image_landmarks,
        "world_landmarks": world_landmarks,
        "valid_mask": valid_mask,
        "handedness": handedness,
        "handedness_score": handedness_score,
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "fps": np.asarray(record.fps, dtype=np.float64),
        "width": np.asarray(width, dtype=np.int32),
        "height": np.asarray(height, dtype=np.int32),
        "video_id": np.asarray(record.video_id),
        "split": np.asarray(record.split),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "model_sha256": np.asarray(model_hash),
        "extraction_config_json": np.asarray(extraction_config_json),
        "elapsed_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
        "processing_fps": np.asarray(processing_fps, dtype=np.float64),
    }
    atomic_save_npz(output_path, arrays)
    return ArtifactSummary(
        record.num_frames,
        int(valid_mask.sum()),
        record.fps,
        width,
        height,
        elapsed_seconds,
        processing_fps,
    )


def validate_artifact(
    path: Path,
    record: ManifestRecord,
    expected_config_json: str,
    expected_model_hash: str,
) -> Tuple[bool, str, Optional[ArtifactSummary]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(REQUIRED_ARRAYS - set(data.files))
            if missing:
                return False, f"missing arrays: {missing}", None
            expected_landmark_shape = (record.num_frames, LANDMARK_COUNT, COORDINATE_COUNT)
            if data["image_landmarks"].shape != expected_landmark_shape:
                return False, "image_landmarks shape mismatch", None
            if data["world_landmarks"].shape != expected_landmark_shape:
                return False, "world_landmarks shape mismatch", None
            for key in ("valid_mask", "handedness", "handedness_score", "frame_index", "timestamp_ms"):
                if data[key].shape != (record.num_frames,):
                    return False, f"{key} shape mismatch", None
            if str(scalar(data, "video_id")) != record.video_id:
                return False, "video_id mismatch", None
            if str(scalar(data, "split")) != record.split:
                return False, "split mismatch", None
            if int(scalar(data, "schema_version")) != SCHEMA_VERSION:
                return False, "schema_version mismatch", None
            if str(scalar(data, "model_sha256")) != expected_model_hash:
                return False, "model hash mismatch", None
            if str(scalar(data, "extraction_config_json")) != expected_config_json:
                return False, "extraction configuration mismatch", None
            if not math.isclose(float(scalar(data, "fps")), record.fps, abs_tol=1e-6):
                return False, "fps mismatch", None
            if not np.array_equal(data["frame_index"], np.arange(record.num_frames)):
                return False, "frame_index is not contiguous", None
            timestamps = data["timestamp_ms"]
            if record.num_frames and int(timestamps[0]) != 0:
                return False, "timestamp_ms does not start at zero", None
            if record.num_frames > 1 and not np.all(np.diff(timestamps) > 0):
                return False, "timestamp_ms is not strictly increasing", None
            valid_mask = data["valid_mask"].astype(np.bool_, copy=False)
            image_landmarks = data["image_landmarks"]
            world_landmarks = data["world_landmarks"]
            if not np.isfinite(image_landmarks).all() or not np.isfinite(world_landmarks).all():
                return False, "landmarks contain NaN/Inf", None
            if np.any(image_landmarks[~valid_mask]) or np.any(world_landmarks[~valid_mask]):
                return False, "invalid frames contain non-zero landmarks", None
            width = int(scalar(data, "width"))
            height = int(scalar(data, "height"))
            if width <= 0 or height <= 0:
                return False, "invalid resolution metadata", None
            summary = ArtifactSummary(
                record.num_frames,
                int(valid_mask.sum()),
                float(scalar(data, "fps")),
                width,
                height,
                float(scalar(data, "elapsed_seconds")),
                float(scalar(data, "processing_fps")),
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, f"cannot read artifact: {error}", None
    return True, "valid", summary


def result_row(
    record: ManifestRecord,
    output_path: Path,
    status: str,
    summary: Optional[ArtifactSummary],
    error: str = "",
) -> Dict[str, Any]:
    return {
        "video_id": record.video_id,
        "split": record.split,
        "status": status,
        "num_frames": record.num_frames,
        "valid_frames": summary.valid_frames if summary is not None else None,
        "valid_rate": (
            summary.valid_frames / summary.total_frames
            if summary is not None and summary.total_frames
            else None
        ),
        "fps": summary.fps if summary is not None else record.fps,
        "width": summary.width if summary is not None else None,
        "height": summary.height if summary is not None else None,
        "elapsed_seconds": summary.elapsed_seconds if summary is not None else None,
        "processing_fps": summary.processing_fps if summary is not None else None,
        "output_path": str(output_path),
        "error": error,
    }


def write_report(
    path: Path,
    split: str,
    selected_count: int,
    rows: Sequence[Mapping[str, Any]],
    model_path: Path,
    extraction_config: Mapping[str, Any],
) -> None:
    successful = [row for row in rows if row["status"] in {"complete", "resumed"}]
    failed = [row for row in rows if row["status"] == "failed"]
    total_frames = sum(int(row["num_frames"]) for row in successful)
    valid_frames = sum(int(row["valid_frames"]) for row in successful)
    payload = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "split": split,
        "selected_videos": selected_count,
        "reported_videos": len(rows),
        "successful_videos": len(successful),
        "failed_videos": len(failed),
        "complete": len(successful) == selected_count and not failed,
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "valid_rate": valid_frames / total_frames if total_frames else None,
        "model_path": str(model_path),
        "extraction_config": dict(extraction_config),
        "videos": list(rows),
    }
    atomic_write_json(path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path (default: data/manifests/<split>.json).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Skeleton output root."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Extraction report path (default: outputs/skeleton_extraction_<split>.json).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="MediaPipe hand_landmarker.task; known project locations are searched by default.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Extract one manifest video; repeat the option to select multiple videos.",
    )
    parser.add_argument("--num-hands", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--fps-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500,
        help="Print progress every N decoded frames; use 0 to disable frame progress.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-extract even when a valid artifact exists."
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop immediately after the first video failure."
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_hands < 1:
        raise ExtractionError("--num-hands must be at least 1")
    if not math.isfinite(args.fps_tolerance) or args.fps_tolerance < 0:
        raise ExtractionError("--fps-tolerance must be non-negative")
    if args.progress_interval < 0:
        raise ExtractionError("--progress-interval must be non-negative")
    for name in (
        "min_detection_confidence",
        "min_presence_confidence",
        "min_tracking_confidence",
    ):
        validate_probability(name, float(getattr(args, name)))


def run(args: argparse.Namespace) -> Tuple[int, int, Path]:
    validate_args(args)
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else (DEFAULT_MANIFEST_DIR / f"{args.split}.json").resolve()
    )
    report_path = (
        args.report.resolve()
        if args.report is not None
        else (DEFAULT_REPORT_DIR / f"skeleton_extraction_{args.split}.json").resolve()
    )
    output_dir = (args.output_root.resolve() / args.split)
    model_path = resolve_model_path(args.model_path)
    model_hash = sha256_file(model_path)
    extraction_config = build_extraction_config(args, model_hash)
    expected_config_json = config_json(extraction_config)
    records = select_records(load_manifest(manifest_path, args.split), args.video_id)

    # Fail before a long run if the model and installed MediaPipe are incompatible.
    preflight_landmarker = create_landmarker(args, model_path)
    preflight_landmarker.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    failures = 0
    print(f"Skeleton split: {args.split}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Model: {model_path}", flush=True)
    print(f"Selected videos: {len(records)}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    for index, record in enumerate(records, start=1):
        output_path = output_dir / f"{record.video_id}.npz"
        print(f"[{index:03d}/{len(records):03d}] {record.video_id}", flush=True)
        if output_path.is_file() and not args.overwrite:
            valid, reason, summary = validate_artifact(
                output_path, record, expected_config_json, model_hash
            )
            if valid:
                rows.append(result_row(record, output_path, "resumed", summary))
                print(
                    f"    resumed: valid={summary.valid_frames}/{summary.total_frames} "
                    f"({summary.valid_frames / summary.total_frames:.1%})",
                    flush=True,
                )
                write_report(
                    report_path, args.split, len(records), rows, model_path, extraction_config
                )
                continue
            print(f"    existing artifact is stale/invalid; re-extracting ({reason})", flush=True)

        try:
            summary = extract_video(
                record,
                output_path,
                model_path,
                expected_config_json,
                model_hash,
                args,
            )
            rows.append(result_row(record, output_path, "complete", summary))
            print(
                f"    complete: valid={summary.valid_frames}/{summary.total_frames} "
                f"({summary.valid_frames / summary.total_frames:.1%}), "
                f"speed={summary.processing_fps:.1f} frame/s",
                flush=True,
            )
        except (ExtractionError, OSError, RuntimeError, ValueError, cv2.error) as error:
            failures += 1
            rows.append(result_row(record, output_path, "failed", None, str(error)))
            print(f"    FAILED: {error}", file=sys.stderr, flush=True)
            write_report(
                report_path, args.split, len(records), rows, model_path, extraction_config
            )
            if args.fail_fast:
                break
            continue
        write_report(report_path, args.split, len(records), rows, model_path, extraction_config)

    successful = sum(row["status"] in {"complete", "resumed"} for row in rows)
    return successful, failures, report_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        successful, failures, report_path = run(args)
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Completed .npz files are safe; run the same command to resume.",
            file=sys.stderr,
        )
        return 130
    except (ExtractionError, OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print("\nSkeleton extraction summary")
    print(f"  Successful videos: {successful}")
    print(f"  Failed videos:     {failures}")
    print(f"  Report:            {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
