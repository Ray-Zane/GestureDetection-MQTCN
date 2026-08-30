#!/usr/bin/env python3
"""Build frame-aligned MP4 streams from the official IPN Hand JPG frames.

The local AVI conversions may contain duplicated frames and therefore do not
necessarily share the annotation timeline.  This tool writes each numbered JPG
exactly once, in numeric order, producing an MP4 that can still be consumed as
a real-time video stream while retaining frame-to-annotation alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tools.preprocess.ipn_source import (
    IPNSourceError,
    clean_video_id,
    direct_file_map,
    find_annotations_dir,
    find_named_file,
    load_official_frame_counts,
    read_annotations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "aligned_videos"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "preprocessing" / "aligned_video_report.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class AlignedVideoError(RuntimeError):
    """Raised when a frame sequence cannot be encoded without ambiguity."""


class ProgressBar:
    """Small dependency-free terminal progress bar with speed and ETA."""

    def __init__(self, total: int, label: str, width: int = 28) -> None:
        self.total = max(1, total)
        self.label = label
        self.width = width
        self.started = time.perf_counter()
        self.last_rendered = -1
        self.interactive = sys.stdout.isatty()

    def update(self, current: int) -> None:
        current = min(max(0, current), self.total)
        percent = int(current * 100 / self.total)
        # Interactive terminals redraw smoothly in place. Captured logs emit a
        # compact line every 10% so CI output does not contain 100 bar states.
        render_step = 1 if self.interactive else 10
        render_bucket = percent if self.interactive else percent // render_step * render_step
        if current not in {0, self.total} and render_bucket == self.last_rendered:
            return
        self.last_rendered = render_bucket
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        speed = current / elapsed
        remaining = (self.total - current) / speed if speed > 0 else 0.0
        filled = round(self.width * current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        line = (
            f"  {self.label:<10} [{bar}] {percent:3d}% "
            f"{current:>5}/{self.total:<5} {speed:6.1f} frame/s ETA {remaining:6.1f}s"
        )
        prefix = "\r" if self.interactive else ""
        end = "\n" if current == self.total or not self.interactive else ""
        print(prefix + line, end=end, flush=True)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
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


def frame_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise AlignedVideoError(f"Frame filename lacks a numeric suffix: {path.name}") from error


def discover_frame_sequences(frames_dir: Path) -> Dict[str, Path]:
    if not frames_dir.is_dir():
        raise AlignedVideoError(f"Frame root does not exist: {frames_dir}")
    result = {path.name: path for path in frames_dir.iterdir() if path.is_dir()}
    if not result:
        raise AlignedVideoError(f"No per-video frame directories found in: {frames_dir}")
    return result


def ordered_frames(directory: Path) -> Tuple[List[Path], List[int]]:
    numbered: Dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        number = frame_number(path)
        if number in numbered:
            raise AlignedVideoError(
                f"Duplicate frame number {number} in {directory}: "
                f"{numbered[number].name}, {path.name}"
            )
        numbered[number] = path
    if not numbered:
        raise AlignedVideoError(f"No numbered images found in: {directory}")
    numbers = sorted(numbered)
    expected = list(range(numbers[0], numbers[-1] + 1))
    missing = sorted(set(expected) - set(numbers))
    if numbers[0] != 1 or missing:
        preview = missing[:10]
        raise AlignedVideoError(
            f"Frame sequence must start at 1 and be consecutive in {directory}; "
            f"first={numbers[0]}, missing={preview}"
        )
    return [numbered[number] for number in numbers], numbers


def choose_video_ids(
    sequences: Dict[str, Path], requested: Sequence[str]
) -> List[str]:
    if requested:
        selected: List[str] = []
        seen = set()
        for value in requested:
            video_id = clean_video_id(value)
            if video_id in seen:
                continue
            if video_id not in sequences:
                raise AlignedVideoError(f"Frame directory not found for: {video_id}")
            selected.append(video_id)
            seen.add(video_id)
        return selected
    return sorted(sequences)


def sample_indices(frame_count: int, count: int) -> List[int]:
    count = min(max(1, count), frame_count)
    return sorted(
        {
            min(frame_count - 1, max(0, round(index * (frame_count - 1) / max(1, count - 1))))
            for index in range(count)
        }
    )


def validate_encoded_video(
    path: Path,
    source_frames: Sequence[Path],
    expected_fps: float,
    validation_samples: int,
    show_progress: bool = False,
) -> Dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise AlignedVideoError(f"Encoded video cannot be opened: {path}")
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    selected = set(sample_indices(len(source_frames), validation_samples))
    content_errors: List[Dict[str, Any]] = []
    decoded_frames = 0
    progress = ProgressBar(len(source_frames), "validate") if show_progress else None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index = decoded_frames
        decoded_frames += 1
        if progress is not None:
            progress.update(decoded_frames)
        if index not in selected:
            continue
        source = cv2.imread(str(source_frames[index]), cv2.IMREAD_COLOR)
        if source is None:
            raise AlignedVideoError(f"Unable to read validation source frame: {source_frames[index]}")
        if source.shape != frame.shape:
            raise AlignedVideoError(
                f"Decoded/source shape mismatch at frame {index + 1}: {frame.shape} vs {source.shape}"
            )
        mae = float(np.abs(frame.astype(np.int16) - source.astype(np.int16)).mean())
        content_errors.append({"frame": index + 1, "mean_absolute_error": mae})
    capture.release()
    errors = [item["mean_absolute_error"] for item in content_errors]
    expected_frames = len(source_frames)
    return {
        "opened": True,
        "reported_frames": reported_frames,
        "decoded_frames": decoded_frames,
        "expected_frames": expected_frames,
        "fps": decoded_fps,
        "expected_fps": expected_fps,
        "resolution": [width, height],
        "sampled_content_frames": len(content_errors),
        "sample_mae_mean": float(np.mean(errors)) if errors else None,
        "sample_mae_max": max(errors) if errors else None,
        "sample_content_errors": content_errors,
        "frame_count_match": reported_frames == expected_frames and decoded_frames == expected_frames,
        "fps_match": math.isclose(decoded_fps, expected_fps, abs_tol=0.01),
        # The threshold is deliberately generous for lossy mp4v encoding; the
        # observed error is normally far lower.
        "content_match": bool(errors) and max(errors) < 20.0,
        "accepted": (
            reported_frames == expected_frames
            and decoded_frames == expected_frames
            and math.isclose(decoded_fps, expected_fps, abs_tol=0.01)
            and bool(errors)
            and max(errors) < 20.0
        ),
    }


def encode_video(
    video_id: str,
    frames: Sequence[Path],
    output_path: Path,
    fps: float,
    codec: str,
    validation_samples: int,
) -> Dict[str, Any]:
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise AlignedVideoError(f"Unable to read first frame: {frames[0]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    writer = None
    progress = ProgressBar(len(frames), "encode")
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".mp4", delete=False) as stream:
            temporary = Path(stream.name)
        writer = cv2.VideoWriter(
            str(temporary), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if not writer.isOpened():
            raise AlignedVideoError(
                f"Unable to create MP4 with codec {codec!r}; try --codec mp4v"
            )
        for index, path in enumerate(frames, start=1):
            image = first if index == 1 else cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise AlignedVideoError(f"Unable to read frame: {path}")
            if image.shape[:2] != (height, width):
                raise AlignedVideoError(
                    f"Frame resolution changed at {path}: {image.shape[1]}x{image.shape[0]}, "
                    f"expected {width}x{height}"
                )
            writer.write(image)
            progress.update(index)
        writer.release()
        writer = None
        validation = validate_encoded_video(
            temporary, frames, fps, validation_samples, show_progress=True
        )
        if not validation["accepted"]:
            raise AlignedVideoError(f"Encoded video validation failed: {validation}")
        os.replace(str(temporary), str(output_path))
        temporary = None
    finally:
        if writer is not None:
            writer.release()
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        "video_id": video_id,
        "status": "encoded",
        "output_file": str(output_path),
        "source_frames": len(frames),
        "first_source_frame": frames[0].name,
        "last_source_frame": frames[-1].name,
        "resolution": [width, height],
        "fps": fps,
        "codec": codec,
        "validation": validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build annotation-aligned MP4s from IPN JPG frames.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--annotations-dir", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--video-id", action="append", default=[], help="Build one video; default: all frame directories."
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--validation-samples", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit 1 unless every output validates.")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.fps <= 0:
        raise AlignedVideoError("--fps must be positive")
    if len(args.codec) != 4:
        raise AlignedVideoError("--codec must contain exactly four characters")
    if args.validation_samples < 1:
        raise AlignedVideoError("--validation-samples must be at least 1")
    data_root = args.data_root.resolve()
    annotations_dir = find_annotations_dir(data_root, args.annotations_dir)
    frames_dir = (args.frames_dir or data_root / "frames").resolve()
    output_dir = args.output_dir.resolve()
    sequences = discover_frame_sequences(frames_dir)
    selected = choose_video_ids(sequences, args.video_id)
    official_counts = load_official_frame_counts(annotations_dir)
    files = direct_file_map(annotations_dir)
    annotation_path = find_named_file(files, "Annot_List.txt")
    if annotation_path is None:
        raise AlignedVideoError(f"Annot_List.txt not found in: {annotations_dir}")
    annotation_last = {}
    for annotation in read_annotations(annotation_path):
        annotation_last[annotation.video] = max(annotation_last.get(annotation.video, 0), annotation.end)

    rows: List[Dict[str, Any]] = []
    print(f"Building {len(selected)} aligned MP4 video(s) at {args.fps:g} FPS", flush=True)
    for index, video_id in enumerate(selected, start=1):
        frame_paths, numbers = ordered_frames(sequences[video_id])
        output_path = output_dir / f"{video_id}.mp4"
        print(f"[{index:03d}/{len(selected):03d}] {video_id}", flush=True)
        if output_path.is_file() and not args.overwrite:
            validation = validate_encoded_video(
                output_path, frame_paths, args.fps, args.validation_samples, show_progress=True
            )
            if validation["accepted"]:
                row = {
                    "video_id": video_id,
                    "status": "resumed",
                    "output_file": str(output_path),
                    "source_frames": len(frame_paths),
                    "first_source_frame": frame_paths[0].name,
                    "last_source_frame": frame_paths[-1].name,
                    "resolution": validation["resolution"],
                    "fps": args.fps,
                    "codec": args.codec,
                    "validation": validation,
                }
            else:
                raise AlignedVideoError(
                    f"Existing output failed validation; rerun with --overwrite: {output_path}"
                )
        else:
            row = encode_video(
                video_id,
                frame_paths,
                output_path,
                args.fps,
                args.codec,
                args.validation_samples,
            )
        row["official_list_frames"] = official_counts.get(video_id)
        row["annotation_last_frame"] = annotation_last.get(video_id)
        row["source_vs_official_difference"] = (
            len(frame_paths) - official_counts[video_id] if video_id in official_counts else None
        )
        row["annotation_covered_by_source"] = annotation_last.get(video_id, 0) <= len(frame_paths)
        row["source_indices_consecutive"] = numbers == list(range(1, len(numbers) + 1))
        row["accepted"] = bool(
            row["validation"]["accepted"]
            and row["annotation_covered_by_source"]
            and row["source_indices_consecutive"]
        )
        rows.append(row)
        atomic_write_json(
            args.report.resolve(),
            {
                "schema_version": 1,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "accepted": all(item["accepted"] for item in rows) and len(rows) == len(selected),
                "selected_videos": len(selected),
                "completed_videos": len(rows),
                "videos": rows,
            },
        )
        print(
            f"  frames={len(frame_paths)}, resolution={row['resolution'][0]}x{row['resolution'][1]}, "
            f"sample_mae_max={row['validation']['sample_mae_max']:.3f}, accepted={row['accepted']}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": all(row["accepted"] for row in rows) and len(rows) == len(selected),
        "selected_videos": len(selected),
        "completed_videos": len(rows),
        "output_dir": str(output_dir),
        "fps": args.fps,
        "codec": args.codec,
        "videos": rows,
    }
    atomic_write_json(args.report.resolve(), payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (AlignedVideoError, IPNSourceError, OSError, ValueError, cv2.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print("\nAligned MP4 summary")
    print(f"  Videos:       {report['completed_videos']}/{report['selected_videos']}")
    print(f"  FPS:          {report['fps']}")
    print(f"  Accepted:     {report['accepted']}")
    print(f"  Output dir:   {report['output_dir']}")
    print(f"  Report:       {args.report.resolve()}")
    return 1 if args.strict and not report["accepted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
