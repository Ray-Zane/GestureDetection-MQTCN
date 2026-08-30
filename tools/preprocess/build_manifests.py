#!/usr/bin/env python3
"""Build strict IPN Hand train/test manifests for the aligned videos.

The official IPN Hand annotation files use one-based, inclusive frame
boundaries.  This project uses zero-based, inclusive boundaries so that a
manifest segment can index a skeleton array directly.  Only ``D0X`` is
background; ``B0A``, ``B0B`` and ``G01`` through ``G11`` are the 13 target
gesture classes.

The script intentionally depends only on the Python standard library.  If
OpenCV is installed it is used to read video metadata; otherwise the bundled
ISO-BMFF reader handles the MP4 files in ``data/raw/aligned_videos``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import struct
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "raw" / "annotations"
DEFAULT_VIDEOS_DIR = PROJECT_ROOT / "data" / "raw" / "aligned_videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "outputs" / "preprocessing" / "dataset_audit.txt"
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg"}
BACKGROUND_CODE = "D0X"


@dataclass(frozen=True)
class GestureClass:
    class_id: int
    code: str
    official_class_id: int
    name: str


GESTURE_CLASSES: Tuple[GestureClass, ...] = (
    GestureClass(0, "B0A", 2, "Pointing with one finger"),
    GestureClass(1, "B0B", 3, "Pointing with two fingers"),
    GestureClass(2, "G01", 4, "Click with one finger"),
    GestureClass(3, "G02", 5, "Click with two fingers"),
    GestureClass(4, "G03", 6, "Throw up"),
    GestureClass(5, "G04", 7, "Throw down"),
    GestureClass(6, "G05", 8, "Throw left"),
    GestureClass(7, "G06", 9, "Throw right"),
    GestureClass(8, "G07", 10, "Open twice"),
    GestureClass(9, "G08", 11, "Double click with one finger"),
    GestureClass(10, "G09", 12, "Double click with two fingers"),
    GestureClass(11, "G10", 13, "Zoom in"),
    GestureClass(12, "G11", 14, "Zoom out"),
)
CLASS_BY_CODE: Mapping[str, GestureClass] = {item.code: item for item in GESTURE_CLASSES}
EXPECTED_OFFICIAL_CLASSES: Mapping[str, int] = {
    BACKGROUND_CODE: 1,
    **{item.code: item.official_class_id for item in GESTURE_CLASSES},
}


class ManifestBuildError(RuntimeError):
    """Raised when the dataset cannot produce an unambiguous manifest."""


@dataclass(frozen=True)
class Annotation:
    video_id: str
    label: str
    official_class_id: int
    start: int
    end: int
    frames: int

    @property
    def key(self) -> Tuple[str, str, int, int, int, int]:
        return (
            self.video_id,
            self.label,
            self.official_class_id,
            self.start,
            self.end,
            self.frames,
        )


@dataclass(frozen=True)
class VideoMetadata:
    num_frames: int
    fps: float
    source: str


@dataclass(frozen=True)
class Box:
    box_type: bytes
    payload_start: int
    end: int


def clean_video_id(value: str) -> str:
    value = value.strip()
    suffix = Path(value).suffix.lower()
    return Path(value).stem if suffix in VIDEO_SUFFIXES else value


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise ManifestBuildError(f"Missing {description}: {path}")
    return path


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, content)


def read_class_index(path: Path) -> Dict[str, int]:
    classes: Dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        for line_number, row in enumerate(reader, start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if line_number == 1 and row[0].strip().lower() == "id":
                continue
            if len(row) != 2:
                raise ManifestBuildError(f"{path}:{line_number}: expected 2 columns, got {len(row)}")
            try:
                official_id = int(row[0].strip())
            except ValueError as error:
                raise ManifestBuildError(
                    f"{path}:{line_number}: invalid class id {row[0]!r}"
                ) from error
            label = row[1].strip()
            if label in classes:
                raise ManifestBuildError(f"{path}:{line_number}: duplicate label {label}")
            classes[label] = official_id
    if classes != dict(EXPECTED_OFFICIAL_CLASSES):
        raise ManifestBuildError(
            "classIdx.txt does not match the expected IPN Hand label mapping: "
            f"expected {dict(EXPECTED_OFFICIAL_CLASSES)}, got {classes}"
        )
    return classes


def read_video_list(path: Path) -> Tuple[List[str], Dict[str, int]]:
    order: List[str] = []
    declared_frames: Dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            columns = line.split()
            if len(columns) != 2:
                raise ManifestBuildError(f"{path}:{line_number}: expected '<video> <frames>'")
            video_id = clean_video_id(columns[0])
            try:
                frames = int(columns[1])
            except ValueError as error:
                raise ManifestBuildError(
                    f"{path}:{line_number}: invalid frame count {columns[1]!r}"
                ) from error
            if frames <= 0:
                raise ManifestBuildError(f"{path}:{line_number}: frame count must be positive")
            if video_id in declared_frames:
                raise ManifestBuildError(f"{path}:{line_number}: duplicate video {video_id}")
            order.append(video_id)
            declared_frames[video_id] = frames
    if not order:
        raise ManifestBuildError(f"No videos found in {path}")
    return order, declared_frames


def read_annotations(path: Path, *, has_header: bool = False) -> List[Annotation]:
    annotations: List[Annotation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        for line_number, row in enumerate(reader, start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if has_header and line_number == 1:
                expected = ["video", "label", "id", "t_start", "t_end", "frames"]
                if [cell.strip().lower() for cell in row] != expected:
                    raise ManifestBuildError(f"{path}: unexpected header {row}")
                continue
            if len(row) != 6:
                raise ManifestBuildError(f"{path}:{line_number}: expected 6 columns, got {len(row)}")
            video_id = clean_video_id(row[0])
            label = row[1].strip()
            try:
                official_class_id, start, end, frames = (int(cell.strip()) for cell in row[2:])
            except ValueError as error:
                raise ManifestBuildError(f"{path}:{line_number}: invalid integer field") from error
            if label not in EXPECTED_OFFICIAL_CLASSES:
                raise ManifestBuildError(f"{path}:{line_number}: unknown label {label}")
            if official_class_id != EXPECTED_OFFICIAL_CLASSES[label]:
                raise ManifestBuildError(
                    f"{path}:{line_number}: label {label} expects official id "
                    f"{EXPECTED_OFFICIAL_CLASSES[label]}, got {official_class_id}"
                )
            if start < 1 or end < start:
                raise ManifestBuildError(
                    f"{path}:{line_number}: invalid one-based inclusive range [{start}, {end}]"
                )
            if frames != end - start + 1:
                raise ManifestBuildError(
                    f"{path}:{line_number}: frames={frames} but range [{start}, {end}] "
                    f"contains {end - start + 1} frames"
                )
            annotations.append(
                Annotation(video_id, label, official_class_id, start, end, frames)
            )
    if not annotations:
        raise ManifestBuildError(f"No annotations found in {path}")
    return annotations


def group_annotations(rows: Iterable[Annotation]) -> Dict[str, List[Annotation]]:
    grouped: Dict[str, List[Annotation]] = defaultdict(list)
    for row in rows:
        grouped[row.video_id].append(row)
    for video_rows in grouped.values():
        video_rows.sort(key=lambda item: (item.start, item.end, item.label))
    return dict(grouped)


def discover_videos(directory: Path) -> Tuple[Dict[str, Path], List[str]]:
    if not directory.is_dir():
        raise ManifestBuildError(f"Aligned video directory does not exist: {directory}")
    videos: Dict[str, Path] = {}
    duplicate_ids: List[str] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        video_id = clean_video_id(path.name)
        if video_id in videos:
            duplicate_ids.append(video_id)
        else:
            videos[video_id] = path
    return videos, sorted(set(duplicate_ids))


def iter_boxes(stream: BinaryIO, start: int, end: int) -> Iterator[Box]:
    position = start
    while position + 8 <= end:
        stream.seek(position)
        header = stream.read(8)
        if len(header) != 8:
            break
        size, box_type = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) != 8:
                raise ManifestBuildError("Truncated extended MP4 box header")
            size = struct.unpack(">Q", extended)[0]
            header_size = 16
        elif size == 0:
            size = end - position
        if size < header_size or position + size > end:
            raise ManifestBuildError(
                f"Invalid MP4 box {box_type!r} at byte {position}: size={size}, parent_end={end}"
            )
        yield Box(box_type, position + header_size, position + size)
        position += size


def child_box(stream: BinaryIO, parent: Box, box_type: bytes) -> Optional[Box]:
    for box in iter_boxes(stream, parent.payload_start, parent.end):
        if box.box_type == box_type:
            return box
    return None


def read_exact_at(stream: BinaryIO, position: int, size: int, description: str) -> bytes:
    stream.seek(position)
    payload = stream.read(size)
    if len(payload) != size:
        raise ManifestBuildError(f"Truncated MP4 {description}")
    return payload


def parse_mp4_video_metadata(path: Path) -> VideoMetadata:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        moov = next(
            (box for box in iter_boxes(stream, 0, file_size) if box.box_type == b"moov"),
            None,
        )
        if moov is None:
            raise ManifestBuildError(f"MP4 has no moov box: {path}")
        for trak in iter_boxes(stream, moov.payload_start, moov.end):
            if trak.box_type != b"trak":
                continue
            mdia = child_box(stream, trak, b"mdia")
            if mdia is None:
                continue
            hdlr = child_box(stream, mdia, b"hdlr")
            if hdlr is None:
                continue
            handler_type = read_exact_at(stream, hdlr.payload_start + 8, 4, "handler type")
            if handler_type != b"vide":
                continue

            mdhd = child_box(stream, mdia, b"mdhd")
            minf = child_box(stream, mdia, b"minf")
            stbl = child_box(stream, minf, b"stbl") if minf is not None else None
            stsz = child_box(stream, stbl, b"stsz") if stbl is not None else None
            stts = child_box(stream, stbl, b"stts") if stbl is not None else None
            if mdhd is None or stsz is None:
                raise ManifestBuildError(f"Video track lacks mdhd/stsz metadata: {path}")

            version = read_exact_at(stream, mdhd.payload_start, 1, "mdhd version")[0]
            if version == 0:
                timing = read_exact_at(stream, mdhd.payload_start + 12, 8, "mdhd timing")
                timescale, duration = struct.unpack(">II", timing)
            elif version == 1:
                timing = read_exact_at(stream, mdhd.payload_start + 20, 12, "mdhd timing")
                timescale, duration = struct.unpack(">IQ", timing)
            else:
                raise ManifestBuildError(f"Unsupported mdhd version {version}: {path}")

            stsz_header = read_exact_at(stream, stsz.payload_start, 12, "stsz header")
            _, _, num_frames = struct.unpack(">III", stsz_header)
            sample_count = 0
            sample_duration = 0
            if stts is not None:
                entry_count = struct.unpack(
                    ">I", read_exact_at(stream, stts.payload_start + 4, 4, "stts entry count")
                )[0]
                entries = read_exact_at(
                    stream, stts.payload_start + 8, entry_count * 8, "stts entries"
                )
                for offset in range(0, len(entries), 8):
                    count, delta = struct.unpack_from(">II", entries, offset)
                    sample_count += count
                    sample_duration += count * delta
            if num_frames <= 0:
                num_frames = sample_count
            if num_frames <= 0:
                raise ManifestBuildError(f"MP4 reports no video frames: {path}")
            if sample_count not in (0, num_frames):
                raise ManifestBuildError(
                    f"MP4 stsz/stts sample count mismatch for {path}: {num_frames} vs {sample_count}"
                )
            timing_duration = sample_duration or duration
            if timescale <= 0 or timing_duration <= 0:
                raise ManifestBuildError(f"MP4 has invalid video timing metadata: {path}")
            fps = num_frames * float(timescale) / float(timing_duration)
            if not math.isfinite(fps) or fps <= 0:
                raise ManifestBuildError(f"MP4 has invalid FPS {fps}: {path}")
            return VideoMetadata(num_frames, fps, "mp4")
    raise ManifestBuildError(f"MP4 has no video track: {path}")


def read_video_metadata(path: Path) -> VideoMetadata:
    try:
        import cv2  # type: ignore
    except (ImportError, ModuleNotFoundError):
        cv2 = None

    if cv2 is not None:
        capture = cv2.VideoCapture(str(path))
        if capture.isOpened():
            num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
            if num_frames > 0 and math.isfinite(fps) and fps > 0:
                return VideoMetadata(num_frames, fps, "opencv")
        else:
            capture.release()

    if path.suffix.lower() in {".mp4", ".mov"}:
        return parse_mp4_video_metadata(path)
    raise ManifestBuildError(
        f"Cannot inspect {path}; install opencv-python or provide MP4 aligned videos"
    )


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def describe(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def validate_split_inputs(
    split: str,
    video_order: Sequence[str],
    annotations: Mapping[str, Sequence[Annotation]],
) -> None:
    listed = set(video_order)
    annotated = set(annotations)
    missing_annotations = sorted(listed - annotated)
    unlisted_annotations = sorted(annotated - listed)
    if missing_annotations or unlisted_annotations:
        raise ManifestBuildError(
            f"{split} split/annotation mismatch; missing annotations={missing_annotations}, "
            f"unlisted annotations={unlisted_annotations}"
        )


def verify_master_annotations(
    master_rows: Sequence[Annotation], split_rows: Sequence[Annotation]
) -> None:
    master_counter = Counter(row.key for row in master_rows)
    split_counter = Counter(row.key for row in split_rows)
    if master_counter != split_counter:
        missing = list((master_counter - split_counter).elements())[:5]
        extra = list((split_counter - master_counter).elements())[:5]
        raise ManifestBuildError(
            "Annot_TrainList.txt + Annot_TestList.txt do not reproduce Annot_List.txt; "
            f"missing sample={missing}, extra sample={extra}"
        )


def build_audit_text(audit: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "IPN Hand Dataset Audit",
        "================================",
        f"Status: {audit['status']}",
        "Frame convention: manifest uses zero-based, inclusive [start, end] segments.",
        "Official source convention: one-based, inclusive; conversion is start-1/end-1.",
        "Background: D0X only. B0A, B0B and G01-G11 are 13 gesture classes.",
        "",
        "Dataset coverage",
        "----------------",
        f"Train videos: {audit['video_counts']['train']}",
        f"Test videos: {audit['video_counts']['test']}",
        f"Total videos: {audit['video_counts']['total']}",
        f"Aligned videos matched: {audit['video_counts']['matched']}",
        f"Unmatched/missing/duplicate aligned videos: {audit['video_counts']['unmatched']}",
        f"Train/Test overlap: {audit['video_counts']['split_overlap']}",
        f"Master annotation rows matched: {audit['master_annotations_matched']}",
        "",
        "Gesture counts",
        "--------------",
        "class_id  code  class_name                         train  test  total",
    ]
    for row in audit["class_counts"]:
        lines.append(
            f"{row['class_id']:>8}  {row['class_code']:<4}  "
            f"{row['class_name']:<33}  {row['train']:>5}  {row['test']:>4}  {row['total']:>5}"
        )

    durations = audit["gesture_duration_frames"]
    gaps = audit["adjacent_gesture_gap_frames"]
    lines.extend(
        [
            "",
            "Gesture duration (frames)",
            "-------------------------",
            f"Count: {int(durations['count'])}",
            f"Min / mean / median / max: {format_number(durations['min'])} / "
            f"{format_number(durations['mean'])} / {format_number(durations['median'])} / "
            f"{format_number(durations['max'])}",
            "",
            "Adjacent gesture gap (frames)",
            "-----------------------------",
            f"Pair count: {int(gaps['count'])}",
            f"Min / mean / median / max: {format_number(gaps['min'])} / "
            f"{format_number(gaps['mean'])} / {format_number(gaps['median'])} / "
            f"{format_number(gaps['max'])}",
            f"No-background consecutive gesture pairs (gap=0): "
            f"{audit['no_background_consecutive_gestures']}",
            "",
            "Integrity checks",
            "----------------",
            f"Segment out-of-bounds count: {audit['segment_out_of_bounds']}",
            f"Official annotation timeline gap count: {audit['timeline_gaps']}",
            f"Official annotation timeline overlap count: {audit['timeline_overlaps']}",
            f"Official list vs aligned frame-count mismatches: "
            f"{audit['official_list_frame_mismatches']}",
            f"  Known +1 differences (official list is one frame longer): "
            f"{audit['official_list_plus_one_mismatches']}",
            f"  Unexpected differences: {audit['unexpected_official_list_frame_mismatches']}",
            f"Aligned video vs annotation-last-frame mismatches: "
            f"{audit['aligned_annotation_frame_mismatches']}",
            f"FPS values: {', '.join(audit['fps_values'])}",
            f"Video metadata backends: {', '.join(audit['metadata_sources'])}",
            "",
            "Completion criteria",
            "-------------------",
            f"All videos matched official split and annotation: "
            f"{'YES' if audit['all_videos_matched'] else 'NO'}",
            f"Segment out-of-bounds count = 0: "
            f"{'YES' if audit['segment_out_of_bounds'] == 0 else 'NO'}",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifests(args: argparse.Namespace) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    annotations_dir = args.annotations_dir.resolve()
    videos_dir = args.videos_dir.resolve()
    class_index_path = require_file(annotations_dir / "classIdx.txt", "class index")
    train_list_path = require_file(annotations_dir / "Video_TrainList.txt", "train video list")
    test_list_path = require_file(annotations_dir / "Video_TestList.txt", "test video list")
    train_annotations_path = require_file(
        annotations_dir / "Annot_TrainList.txt", "train annotations"
    )
    test_annotations_path = require_file(
        annotations_dir / "Annot_TestList.txt", "test annotations"
    )
    master_annotations_path = require_file(annotations_dir / "Annot_List.txt", "master annotations")

    read_class_index(class_index_path)
    train_order, train_declared = read_video_list(train_list_path)
    test_order, test_declared = read_video_list(test_list_path)
    overlap = sorted(set(train_order) & set(test_order))
    if overlap:
        raise ManifestBuildError(f"Train/test split overlap: {overlap}")

    train_rows = read_annotations(train_annotations_path)
    test_rows = read_annotations(test_annotations_path)
    master_rows = read_annotations(master_annotations_path, has_header=True)
    train_grouped = group_annotations(train_rows)
    test_grouped = group_annotations(test_rows)
    validate_split_inputs("train", train_order, train_grouped)
    validate_split_inputs("test", test_order, test_grouped)
    verify_master_annotations(master_rows, train_rows + test_rows)

    videos, duplicate_video_ids = discover_videos(videos_dir)
    expected_ids = set(train_order) | set(test_order)
    missing_videos = sorted(expected_ids - set(videos))
    extra_videos = sorted(set(videos) - expected_ids)
    if duplicate_video_ids or missing_videos or extra_videos:
        raise ManifestBuildError(
            "Aligned video matching failed; "
            f"missing={missing_videos}, extra={extra_videos}, duplicates={duplicate_video_ids}"
        )

    manifests: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": []}
    class_counts: Dict[str, Counter[str]] = {"train": Counter(), "test": Counter()}
    gesture_durations: List[int] = []
    gesture_gaps: List[int] = []
    segment_out_of_bounds = 0
    timeline_gaps = 0
    timeline_overlaps = 0
    official_list_frame_mismatches = 0
    official_list_plus_one_mismatches = 0
    unexpected_official_list_frame_mismatches = 0
    aligned_annotation_frame_mismatches = 0
    fps_values: List[float] = []
    metadata_sources: Counter[str] = Counter()

    for split, order, declared, grouped in (
        ("train", train_order, train_declared, train_grouped),
        ("test", test_order, test_declared, test_grouped),
    ):
        for video_id in order:
            video_path = videos[video_id]
            metadata = read_video_metadata(video_path)
            fps_values.append(metadata.fps)
            metadata_sources[metadata.source] += 1
            if abs(metadata.fps - args.expected_fps) > args.fps_tolerance:
                raise ManifestBuildError(
                    f"Unexpected FPS for {video_id}: {metadata.fps} "
                    f"(expected {args.expected_fps} +/- {args.fps_tolerance})"
                )

            rows = list(grouped[video_id])
            if rows[0].start != 1:
                timeline_gaps += 1
            for previous, current in zip(rows, rows[1:]):
                delta = current.start - previous.end - 1
                if delta > 0:
                    timeline_gaps += 1
                elif delta < 0:
                    timeline_overlaps += 1
            if declared[video_id] != metadata.num_frames:
                official_list_frame_mismatches += 1
                if declared[video_id] - metadata.num_frames == 1:
                    official_list_plus_one_mismatches += 1
                else:
                    unexpected_official_list_frame_mismatches += 1
            if rows[-1].end != metadata.num_frames:
                aligned_annotation_frame_mismatches += 1

            segments: List[Dict[str, Any]] = []
            for annotation in rows:
                if annotation.label == BACKGROUND_CODE:
                    continue
                gesture_class = CLASS_BY_CODE[annotation.label]
                start = annotation.start - 1
                end = annotation.end - 1
                if start < 0 or end < start or end >= metadata.num_frames:
                    segment_out_of_bounds += 1
                segment = {
                    "start": start,
                    "end": end,
                    "class_id": gesture_class.class_id,
                    "class_name": gesture_class.name,
                    "class_code": gesture_class.code,
                    "official_class_id": gesture_class.official_class_id,
                }
                segments.append(segment)
                class_counts[split][gesture_class.code] += 1
                gesture_durations.append(end - start + 1)
            for previous, current in zip(segments, segments[1:]):
                gesture_gaps.append(current["start"] - previous["end"] - 1)

            manifests[split].append(
                {
                    "video_id": video_id,
                    "split": split,
                    "video_path": portable_path(video_path),
                    "fps": metadata.fps,
                    "num_frames": metadata.num_frames,
                    "segments": segments,
                }
            )

    no_background_count = sum(gap == 0 for gap in gesture_gaps)
    unmatched_count = len(missing_videos) + len(extra_videos) + len(duplicate_video_ids)
    all_videos_matched = (
        unmatched_count == 0
        and not overlap
        and aligned_annotation_frame_mismatches == 0
        and unexpected_official_list_frame_mismatches == 0
        and timeline_gaps == 0
        and timeline_overlaps == 0
    )
    status = "PASS" if all_videos_matched and segment_out_of_bounds == 0 else "FAIL"
    audit: Dict[str, Any] = {
        "status": status,
        "video_counts": {
            "train": len(train_order),
            "test": len(test_order),
            "total": len(expected_ids),
            "matched": len(videos),
            "unmatched": unmatched_count,
            "split_overlap": len(overlap),
        },
        "master_annotations_matched": len(master_rows) == len(train_rows) + len(test_rows),
        "class_counts": [
            {
                "class_id": item.class_id,
                "class_code": item.code,
                "class_name": item.name,
                "train": class_counts["train"][item.code],
                "test": class_counts["test"][item.code],
                "total": class_counts["train"][item.code] + class_counts["test"][item.code],
            }
            for item in GESTURE_CLASSES
        ],
        "gesture_duration_frames": describe(gesture_durations),
        "adjacent_gesture_gap_frames": describe(gesture_gaps),
        "no_background_consecutive_gestures": no_background_count,
        "segment_out_of_bounds": segment_out_of_bounds,
        "timeline_gaps": timeline_gaps,
        "timeline_overlaps": timeline_overlaps,
        "official_list_frame_mismatches": official_list_frame_mismatches,
        "official_list_plus_one_mismatches": official_list_plus_one_mismatches,
        "unexpected_official_list_frame_mismatches": (
            unexpected_official_list_frame_mismatches
        ),
        "aligned_annotation_frame_mismatches": aligned_annotation_frame_mismatches,
        "fps_values": [format_number(value) for value in sorted(set(fps_values))],
        "metadata_sources": [
            f"{source}={count}" for source, count in sorted(metadata_sources.items())
        ],
        "all_videos_matched": all_videos_matched,
    }
    return manifests, audit


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--fps-tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    if not math.isfinite(args.expected_fps) or args.expected_fps <= 0:
        parser.error("--expected-fps must be positive")
    if not math.isfinite(args.fps_tolerance) or args.fps_tolerance < 0:
        parser.error("--fps-tolerance must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        manifests, audit = build_manifests(args)
        audit_text = build_audit_text(audit)
        if audit["status"] != "PASS":
            atomic_write_text(args.audit_output.resolve(), audit_text)
            raise ManifestBuildError(
                f"Dataset audit failed; see {args.audit_output.resolve()}"
            )
        output_dir = args.output_dir.resolve()
        atomic_write_json(output_dir / "train.json", manifests["train"])
        atomic_write_json(output_dir / "test.json", manifests["test"])
        atomic_write_text(args.audit_output.resolve(), audit_text)
    except (ManifestBuildError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        f"Dataset PASS: train={len(manifests['train'])}, test={len(manifests['test'])}, "
        f"segments={sum(len(row['segments']) for split in manifests.values() for row in split)}"
    )
    print(f"Manifests: {output_dir / 'train.json'}, {output_dir / 'test.json'}")
    print(f"Audit: {args.audit_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
