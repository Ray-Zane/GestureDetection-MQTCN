"""Run GestureDetection-MQTCN on an IPN/local video or a live camera."""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Support both documented module execution and the common direct invocation:
# ``python tools/realtime_demo.py ...``.  In the latter case Python otherwise
# exposes only ``tools/`` on sys.path and cannot resolve sibling packages such
# as ``datasets`` and ``streaming``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from datasets.ipn_manifest import IPN_CLASS_NAMES, VideoManifest, load_manifest
from models.gesture_detection_mqtcn import GestureDetectionMQTCN
from models.baseline import ContinuousBaseline
from preprocessing.p3_features import P3FeatureConfig
from streaming.realtime_pipeline import RealtimePipeline
from streaming.decoder import decoder_config_from_mapping
from streaming.query_tracker import query_decoder_config_from_mapping
from utils.io import atomic_json, file_sha256


DEFAULT_CONFIG = Path("configs/runtime.yaml")
STREAM_ID = "demo"

# MediaPipe's 21-point hand graph.  Kept local so the drawing path is version
# independent and does not affect the model input.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Local video path")
    source.add_argument(
        "--video-id", type=str, help="Video ID from the selected manifest split"
    )
    source.add_argument("--camera", type=int, help="OpenCV camera index, usually 0")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="Override config checkpoint")
    parser.add_argument("--hand-model", type=Path, help="Override config MediaPipe model")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--save-video", type=Path, help="Optional annotated video")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--display-width", type=int, default=1280, help="Preview/canvas width"
    )
    parser.add_argument(
        "--display-height", type=int, default=720, help="Preview/canvas height"
    )
    parser.add_argument(
        "--mirror-display",
        action="store_true",
        help="Mirror only the preview/saved image; inference remains unmirrored",
    )
    parser.add_argument(
        "--realtime-playback",
        action="store_true",
        help="Pace a local video at its nominal FPS",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_config(root: Path, path: Path) -> Tuple[Mapping[str, Any], Path]:
    resolved = resolve(root, path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("runtime config must be a mapping")
    return payload, resolved


def _choose_device(value: str) -> torch.device:
    requested = str(value).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _feature_config(payload: Mapping[str, Any]) -> P3FeatureConfig:
    raw = payload["feature_builder"]
    return P3FeatureConfig(
        coordinate_source=str(raw["coordinate_source"]),
        motion_lags=tuple(int(value) for value in raw["motion_lags"]),
        max_hold_frames=int(raw["max_hold_frames"]),
        missing_clip_frames=int(raw["missing_clip_frames"]),
        include_handedness=bool(raw["include_handedness"]),
    )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.#-]+", "_", value).strip("._")
    return cleaned or "source"


def _resolve_source(
    root: Path, args: argparse.Namespace
) -> Tuple[Mapping[str, Any], Optional[VideoManifest]]:
    if args.video_id is not None:
        entries = load_manifest(root, args.split)
        entry = next(
            (item for item in entries if item.video_id == str(args.video_id)), None
        )
        if entry is None:
            raise ValueError(
                f"video_id {args.video_id!r} is not in data/manifests/{args.split}.json"
            )
        if not entry.video_path.is_file():
            raise FileNotFoundError(entry.video_path)
        return {
            "kind": "manifest_video",
            "source_id": entry.video_id,
            "path": str(entry.video_path),
            "split": entry.split,
        }, entry
    if args.video is not None:
        path = resolve(root, args.video)
        if not path.is_file():
            raise FileNotFoundError(path)
        return {
            "kind": "local_video",
            "source_id": path.stem,
            "path": str(path),
        }, None
    if int(args.camera) < 0:
        raise ValueError("--camera must be non-negative")
    return {
        "kind": "camera",
        "source_id": f"camera_{int(args.camera)}",
        "camera_index": int(args.camera),
    }, None


def _load_model(
    root: Path,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    *,
    verify_expected_hash: bool = True,
) -> Tuple[Any, Mapping[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    expected_hash = str(config["artifact"].get("checkpoint_sha256", ""))
    actual_hash = file_sha256(checkpoint_path)
    if verify_expected_hash and expected_hash and actual_hash != expected_hash:
        raise RuntimeError(f"model checkpoint SHA-256 mismatch: {checkpoint_path}")
    raw = config["model"]
    baseline = ContinuousBaseline(
        int(raw["input_dim"]),
        architecture="b1",
        hidden_dim=int(raw["hidden_dim"]),
        num_classes=int(raw["num_frame_classes"]),
        kernel_size=int(raw["kernel_size"]),
        dilations=tuple(int(value) for value in raw["dilations"]),
        dropout=float(raw["dropout"]),
    ).to(device)
    model = GestureDetectionMQTCN(
        baseline,
        num_queries=int(raw["num_queries"]),
        num_query_classes=int(raw["num_query_classes"]),
        attention_heads=int(raw["attention_heads"]),
        decoder_layers=int(raw["decoder_layers"]),
        feedforward_dim=int(raw["feedforward_dim"]),
        frame_memory_length=int(raw["frame_memory_length"]),
        dropout=float(raw["dropout"]),
    ).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), {
        "architecture": "gesture_detection_mqtcn",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual_hash,
        "selected_epoch": int(checkpoint["epoch"]),
        "parameter_count": int(model.parameter_count),
        "trainable_parameter_count": int(model.trainable_parameter_count),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latency_summary(values: Sequence[float]) -> Mapping[str, Optional[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(array.max()),
    }


def _extract_hand(result: Any) -> Tuple[np.ndarray, np.ndarray, bool, Optional[str], Optional[float]]:
    if result.hand_landmarks and result.hand_world_landmarks:
        image = np.asarray(
            [[point.x, point.y, point.z] for point in result.hand_landmarks[0]],
            dtype=np.float32,
        )
        world = np.asarray(
            [[point.x, point.y, point.z] for point in result.hand_world_landmarks[0]],
            dtype=np.float32,
        )
        hand = None
        score = None
        if result.handedness and result.handedness[0]:
            category = result.handedness[0][0]
            hand = getattr(category, "category_name", None) or getattr(
                category, "display_name", None
            )
            category_score = getattr(category, "score", None)
            score = None if category_score is None else float(category_score)
        return image, world, True, hand, score
    zeros = np.zeros((21, 3), dtype=np.float32)
    return zeros, zeros.copy(), False, None, None


def _fallback_hand(source_id: str) -> Optional[str]:
    upper = str(source_id).upper()
    if "_L_" in upper:
        return "Left"
    if "_R_" in upper:
        return "Right"
    return None


def _draw_hand(frame: Any, landmarks: np.ndarray, *, mirrored: bool) -> None:
    import cv2

    height, width = frame.shape[:2]
    points = []
    for point in landmarks:
        x = 1.0 - float(point[0]) if mirrored else float(point[0])
        points.append((int(round(x * width)), int(round(float(point[1]) * height))))
    for first, second in HAND_CONNECTIONS:
        cv2.line(frame, points[first], points[second], (40, 220, 80), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, (40, 170, 255), -1, cv2.LINE_AA)


def _wrap_text(
    text: str, *, max_width: int, font_scale: float, thickness: int
) -> Sequence[str]:
    import cv2

    words = str(text).split()
    if not words:
        return ("",)
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0][0]
        if width <= int(max_width):
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return tuple(lines)


def _compose_display(
    video_frame: Any,
    fields: Sequence[Tuple[str, str]],
    *,
    target_width: int,
    target_height: int,
) -> Any:
    """Build a fixed-size letterboxed video plus a separate information panel."""

    import cv2

    canvas_width = int(target_width)
    canvas_height = int(target_height)
    panel_width = min(max(360, int(round(canvas_width * 0.34))), canvas_width - 320)
    video_width = canvas_width - panel_width
    canvas = np.full((canvas_height, canvas_width, 3), (18, 20, 24), dtype=np.uint8)

    source_height, source_width = video_frame.shape[:2]
    scale = min(
        max(1, video_width - 24) / float(source_width),
        max(1, canvas_height - 24) / float(source_height),
    )
    fitted_width = max(1, int(round(source_width * scale)))
    fitted_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    fitted = cv2.resize(
        video_frame, (fitted_width, fitted_height), interpolation=interpolation
    )
    video_x = max(0, (video_width - fitted_width) // 2)
    video_y = max(0, (canvas_height - fitted_height) // 2)
    canvas[
        video_y : video_y + fitted_height,
        video_x : video_x + fitted_width,
    ] = fitted

    panel_x = video_width
    cv2.rectangle(
        canvas,
        (panel_x, 0),
        (canvas_width - 1, canvas_height - 1),
        (30, 34, 42),
        -1,
    )
    cv2.line(canvas, (panel_x, 0), (panel_x, canvas_height), (66, 72, 84), 1)
    left = panel_x + 24
    usable_width = panel_width - 48
    cv2.putText(
        canvas,
        "GestureDetection-MQTCN",
        (left, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (80, 210, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Continuous gesture recognition",
        (left, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (172, 180, 194),
        1,
        cv2.LINE_AA,
    )
    y = 105
    for label, value in fields:
        cv2.putText(
            canvas,
            str(label).upper(),
            (left, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (126, 196, 235),
            1,
            cv2.LINE_AA,
        )
        y += 25
        for line in _wrap_text(
            str(value), max_width=usable_width, font_scale=0.58, thickness=1
        ):
            cv2.putText(
                canvas,
                line,
                (left, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (240, 242, 246),
                1,
                cv2.LINE_AA,
            )
            y += 24
        y += 20
    cv2.putText(
        canvas,
        "Q / Esc: exit",
        (left, canvas_height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (142, 150, 164),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _event_with_name(event: Mapping[str, object], fps: float) -> Mapping[str, object]:
    output = dict(event)
    class_id = int(output["class_id"])
    output["class_name"] = IPN_CLASS_NAMES[class_id]
    if fps > 0:
        output["start_seconds"] = float(int(output["start_frame"]) / fps)
        output["end_seconds"] = float(int(output["end_frame_exclusive"]) / fps)
    return output


def run(args: argparse.Namespace) -> Path:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "Demo requires OpenCV and MediaPipe: pip install opencv-python mediapipe"
        ) from exc

    root = args.project_root.resolve()
    source, manifest_entry = _resolve_source(root, args)
    config, config_path = _load_config(root, args.config)
    checkpoint_path = resolve(
        root,
        args.checkpoint
        if args.checkpoint is not None
        else Path(str(config["artifact"]["checkpoint"])),
    )
    hand_model = resolve(
        root,
        args.hand_model
        if args.hand_model is not None
        else Path(str(config["hand_landmarker"]["model_path"])),
    )
    if not hand_model.is_file():
        raise FileNotFoundError(hand_model)
    if args.max_frames is not None and int(args.max_frames) <= 0:
        raise ValueError("--max-frames must be positive")
    if int(args.cpu_threads) <= 0:
        raise ValueError("--cpu-threads must be positive")
    if int(args.display_width) < 800 or int(args.display_height) < 450:
        raise ValueError("display canvas must be at least 800x450")
    for value in (
        args.min_detection_confidence,
        args.min_presence_confidence,
        args.min_tracking_confidence,
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("MediaPipe confidence thresholds must be in [0,1]")

    torch.set_num_threads(int(args.cpu_threads))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    expected_hand_hash = str(config["hand_landmarker"].get("model_sha256", ""))
    actual_hand_hash = file_sha256(hand_model)
    if expected_hand_hash and actual_hand_hash != expected_hand_hash:
        raise RuntimeError(f"MediaPipe model SHA-256 mismatch: {hand_model}")
    device = _choose_device(args.device)
    model, model_metadata = _load_model(
        root,
        config,
        checkpoint_path,
        device,
        verify_expected_hash=args.checkpoint is None,
    )
    pipeline = RealtimePipeline(
        model,
        feature_config=_feature_config(config),
        frame_decoder_config=decoder_config_from_mapping(config["frame_decoder"]),
        query_decoder_config=query_decoder_config_from_mapping(config["query_decoder"]),
        query_stride=int(config["model"]["query_stride"]),
        device=device,
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=float(args.min_detection_confidence),
        min_hand_presence_confidence=float(args.min_presence_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    if source["kind"] == "camera":
        capture = cv2.VideoCapture(int(source["camera_index"]))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.camera_width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.camera_height))
        capture.set(cv2.CAP_PROP_FPS, float(args.camera_fps))
    else:
        capture = cv2.VideoCapture(str(source["path"]))
    if not capture.isOpened():
        landmarker.close()
        raise RuntimeError(f"cannot open source: {source}")

    nominal_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(nominal_fps) or nominal_fps <= 0:
        nominal_fps = float(args.camera_fps) if source["kind"] == "camera" else 30.0
    run_id = (
        f"{_safe_name(str(source['source_id']))}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_json = (
        resolve(root, args.output_json)
        if args.output_json is not None
        else root / "outputs" / "demo" / run_id / "events.json"
    )
    save_video = resolve(root, args.save_video) if args.save_video is not None else None
    if save_video is not None:
        save_video.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    frame_index = 0
    valid_frames = 0
    reset_count = 0
    query_count = 0
    statuses: Counter[str] = Counter()
    decode_latencies: List[float] = []
    mediapipe_latencies: List[float] = []
    model_latencies: List[float] = []
    total_latencies: List[float] = []
    last_hand = _fallback_hand(str(source["source_id"]))
    last_hand_score: Optional[float] = None
    last_timestamp = -1
    latest_event: Optional[Mapping[str, object]] = None
    stopped_reason = "end_of_stream"
    wall_started = time.perf_counter()
    camera_started = wall_started
    processing_error: Optional[BaseException] = None

    print(f"GestureDetection-MQTCN source: {source['source_id']} | device: {device}")
    print("Press q or Esc to stop." if not args.no_display else "Press Ctrl+C to stop.")
    if not args.no_display:
        cv2.namedWindow("GestureDetection-MQTCN Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "GestureDetection-MQTCN Demo", int(args.display_width), int(args.display_height)
        )
    try:
        while args.max_frames is None or frame_index < int(args.max_frames):
            total_started = time.perf_counter()
            decode_started = time.perf_counter()
            success, bgr = capture.read()
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if not success:
                break
            height, width = bgr.shape[:2]
            media_started = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if source["kind"] == "camera":
                timestamp_ms = int(round((time.perf_counter() - camera_started) * 1000.0))
            else:
                timestamp_ms = int(round(frame_index * 1000.0 / nominal_fps))
            timestamp_ms = max(last_timestamp + 1, timestamp_ms)
            last_timestamp = timestamp_ms
            detection = landmarker.detect_for_video(image, timestamp_ms)
            media_ms = (time.perf_counter() - media_started) * 1000.0
            image_landmarks, world_landmarks, valid, detected_hand, hand_score = _extract_hand(detection)
            if valid:
                valid_frames += 1
                if detected_hand:
                    last_hand = str(detected_hand)
                last_hand_score = hand_score

            _synchronize(device)
            model_started = time.perf_counter()
            step = pipeline.process_skeleton_frame(
                image_landmarks=image_landmarks,
                world_landmarks=world_landmarks,
                valid=valid,
                width=width,
                height=height,
                frame_index=frame_index,
                metadata_hand=last_hand,
                stream_id=STREAM_ID,
            )
            _synchronize(device)
            model_ms = (time.perf_counter() - model_started) * 1000.0
            statuses[step.status] += 1
            if step.status == "reset_missing":
                reset_count += 1
            if step.model is not None and step.model.query_executed:
                query_count += 1
                current = pipeline.events_for_stream(STREAM_ID)["fusion"]
                if current:
                    latest_event = current[-1]
            for event in step.emitted_events:
                named = _event_with_name(event, nominal_fps)
                print(
                    f"[event] frame={frame_index} {named['class_name']} "
                    f"[{named['start_frame']},{named['end_frame_exclusive']}) "
                    f"score={float(named['score']):.3f}"
                )

            video_display = cv2.flip(bgr, 1) if args.mirror_display else bgr.copy()
            if valid:
                _draw_hand(
                    video_display,
                    image_landmarks,
                    mirrored=bool(args.mirror_display),
                )
            current_label = "not processed"
            if step.model is not None:
                probabilities = torch.softmax(
                    step.model.frame_outputs["frame_logits"], dim=-1
                ).detach().cpu()
                index = int(torch.argmax(probabilities).item())
                name = "Background" if index == 0 else IPN_CLASS_NAMES[index - 1]
                current_label = f"{name} ({float(probabilities[index]):.3f})"
            final_label = "none"
            if latest_event is not None:
                final_label = (
                    f"{IPN_CLASS_NAMES[int(latest_event['class_id'])]} "
                    f"[{latest_event['start_frame']},{latest_event['end_frame_exclusive']})"
                )
            total_ms = (time.perf_counter() - total_started) * 1000.0
            display = _compose_display(
                video_display,
                (
                    ("Source", str(source["source_id"])),
                    ("Runtime", f"FP32 | {device} | frame {frame_index}"),
                    (
                        "Tracking",
                        f"{last_hand or 'unknown'} | detected={valid} | {step.status}",
                    ),
                    ("Frame hint", current_label),
                    ("Latest fusion event", final_label),
                    (
                        "Timing",
                        f"{total_ms:.1f} ms current | source {nominal_fps:.2f} FPS",
                    ),
                ),
                target_width=int(args.display_width),
                target_height=int(args.display_height),
            )
            if writer is None and save_video is not None:
                writer = cv2.VideoWriter(
                    str(save_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    nominal_fps,
                    (int(args.display_width), int(args.display_height)),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"cannot create annotated video: {save_video}")
            if writer is not None:
                writer.write(display)
            if not args.no_display:
                cv2.imshow("GestureDetection-MQTCN Demo", display)
                delay = 1
                if args.realtime_playback and source["kind"] != "camera":
                    delay = max(1, int(round(1000.0 / nominal_fps - total_ms)))
                key = cv2.waitKey(delay) & 0xFF
                if key in (27, ord("q")):
                    stopped_reason = "user_requested"
                    frame_index += 1
                    decode_latencies.append(decode_ms)
                    mediapipe_latencies.append(media_ms)
                    model_latencies.append(model_ms)
                    total_latencies.append((time.perf_counter() - total_started) * 1000.0)
                    break

            decode_latencies.append(decode_ms)
            mediapipe_latencies.append(media_ms)
            model_latencies.append(model_ms)
            total_latencies.append((time.perf_counter() - total_started) * 1000.0)
            frame_index += 1
        else:
            stopped_reason = "max_frames"
    except KeyboardInterrupt:
        stopped_reason = "keyboard_interrupt"
    except BaseException as exc:
        stopped_reason = "error"
        processing_error = exc
    finally:
        capture.release()
        landmarker.close()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    tracks = pipeline.finish_stream(STREAM_ID)
    wall_seconds = time.perf_counter() - wall_started
    source_payload: Dict[str, Any] = dict(source)
    source_payload.update(
        {
            "nominal_fps": nominal_fps,
            "ground_truth": [segment.as_dict() for segment in manifest_entry.annotations]
            if manifest_entry is not None
            else None,
        }
    )
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "failed" if processing_error is not None else "completed",
        "stopped_reason": stopped_reason,
        "model": {
            **model_metadata,
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "primary_output": "fusion",
            "query_stride": int(config["model"]["query_stride"]),
            "frame_memory_length": int(config["model"]["frame_memory_length"]),
            "precision": "fp32",
        },
        "hand_landmarker": {
            "model_path": str(hand_model),
            "model_sha256": actual_hand_hash,
            "min_detection_confidence": float(args.min_detection_confidence),
            "min_presence_confidence": float(args.min_presence_confidence),
            "min_tracking_confidence": float(args.min_tracking_confidence),
        },
        "source": source_payload,
        "runtime": {
            "device": str(device),
            "cpu_threads": int(args.cpu_threads),
            "display_canvas": {
                "width": int(args.display_width),
                "height": int(args.display_height),
                "layout": "letterboxed_video_with_right_information_panel",
            },
            "frames_processed": int(frame_index),
            "skeleton_valid_frames": int(valid_frames),
            "skeleton_valid_ratio": float(valid_frames / frame_index) if frame_index else 0.0,
            "state_reset_count": int(reset_count),
            "query_execution_count": int(query_count),
            "status_counts": dict(statuses),
            "wall_seconds": float(wall_seconds),
            "processing_fps": float(frame_index / wall_seconds) if wall_seconds > 0 else 0.0,
            "latency": {
                "capture_decode": _latency_summary(decode_latencies),
                "mediapipe": _latency_summary(mediapipe_latencies),
                "feature_and_model": _latency_summary(model_latencies),
                "end_to_end_with_render": _latency_summary(total_latencies),
            },
        },
        "events": {
            name: [_event_with_name(event, nominal_fps) for event in tracks[name]]
            for name in ("frame", "query", "fusion")
        },
        "artifacts": {
            "annotated_video": str(save_video) if save_video is not None else None,
            "event_json": str(output_json),
        },
        "error": None if processing_error is None else repr(processing_error),
    }
    atomic_json(output_json, report)
    print(f"Event report: {output_json}")
    print(
        f"Fusion events: {len(report['events']['fusion'])} | "
        f"frames: {frame_index} | processing FPS: {report['runtime']['processing_fps']:.2f}"
    )
    if processing_error is not None:
        raise processing_error
    return output_json


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
