"""Evaluate a trained GestureDetection-MQTCN checkpoint on official IPN Test."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import yaml

from datasets.continuous import (
    ContinuousCache,
    continuous_cache_signature,
    prepare_continuous_cache,
)
from datasets.memory_query import (
    EncodedContinuousCache,
    encoded_cache_signature,
    prepare_encoded_cache,
)
from engine.model_evaluator import compact_mqtcn_metrics, evaluate_memory_query
from engine.model_trainer import MemoryQueryLoss, MemoryQueryLossConfig
from models.baseline import ContinuousBaseline
from models.gesture_detection_mqtcn import GestureDetectionMQTCN
from models.query.matcher import MatchCost
from preprocessing.feature_builder import FeatureBuilderConfig, SkeletonFeatureBuilder
from streaming.decoder import decoder_config_from_mapping
from streaming.query_tracker import query_decoder_config_from_mapping
from utils.io import atomic_json, file_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/model/checkpoints/final.pt"),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--video-batch-size", type=int)
    parser.add_argument("--limit-videos", type=int, help="Smoke-test only; omit for official metrics.")
    return parser


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def choose_device(value: str) -> torch.device:
    requested = str(value).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_config(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"training config must be a mapping: {path}")
    return payload


def feature_config(payload: Mapping[str, Any]) -> FeatureBuilderConfig:
    raw = payload["feature_builder"]
    return FeatureBuilderConfig(
        preprocessing_profile=str(raw.get("preprocessing_profile", "p3")),
        coordinate_source=str(raw["coordinate_source"]),
        motion_lags=tuple(int(value) for value in raw["motion_lags"]),
        max_hold_frames=int(raw["max_hold_frames"]),
        missing_clip_frames=int(raw["missing_clip_frames"]),
        include_handedness=bool(raw["include_handedness"]),
    )


def build_model(payload: Mapping[str, Any], device: torch.device) -> GestureDetectionMQTCN:
    raw = payload["model"]
    baseline = ContinuousBaseline(
        int(raw["input_dim"]),
        architecture="b1",
        hidden_dim=int(raw["hidden_dim"]),
        num_classes=int(raw["num_frame_classes"]),
        kernel_size=int(raw["kernel_size"]),
        dilations=tuple(int(value) for value in raw["dilations"]),
        dropout=float(raw["dropout"]),
    )
    return GestureDetectionMQTCN(
        baseline,
        num_queries=int(raw["num_queries"]),
        num_query_classes=int(raw["num_query_classes"]),
        attention_heads=int(raw["attention_heads"]),
        decoder_layers=int(raw["decoder_layers"]),
        feedforward_dim=int(raw["feedforward_dim"]),
        frame_memory_length=int(raw["frame_memory_length"]),
        dropout=float(raw["dropout"]),
    ).to(device)


def build_loss(payload: Mapping[str, Any], device: torch.device) -> MemoryQueryLoss:
    match = payload["matching"]
    loss = payload["loss"]
    return MemoryQueryLoss(
        match_cost=MatchCost(
            class_cost=float(match["class_cost"]),
            boundary_l1_cost=float(match["boundary_l1_cost"]),
            tiou_cost=float(match["tiou_cost"]),
        ),
        config=MemoryQueryLossConfig(
            query_class_weight=float(loss["query_class"]),
            boundary_l1_weight=float(loss["boundary_l1"]),
            tiou_weight=float(loss["tiou"]),
            eos_coef=float(loss["eos_coef"]),
        ),
    ).to(device)


def run(args: argparse.Namespace) -> Path:
    root = args.project_root.resolve()
    config_path = resolve(root, args.config)
    checkpoint_path = resolve(root, args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.limit_videos is not None and int(args.limit_videos) <= 0:
        raise ValueError("--limit-videos must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = choose_device(args.device)
    config = load_config(config_path)
    builder = SkeletonFeatureBuilder(feature_config(config))
    boundary_radius = int(config["targets"]["boundary_radius"])
    continuous_signature = continuous_cache_signature(
        root, builder.config, boundary_radius=boundary_radius
    )
    continuous_root = root / "outputs" / "cache" / "continuous" / continuous_signature
    prepare_continuous_cache(
        project_root=root,
        cache_root=continuous_root,
        split="test",
        builder=builder,
        boundary_radius=boundary_radius,
        signature=continuous_signature,
    )
    source = ContinuousCache(continuous_root, "test")
    model = build_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    checkpoint_hash = file_sha256(checkpoint_path)
    encoded_signature = encoded_cache_signature(continuous_signature, checkpoint_hash)
    encoded_root = root / "outputs" / "cache" / "memory_query" / encoded_signature
    prepare_encoded_cache(
        cache_root=encoded_root,
        source=source,
        baseline=model.baseline,
        checkpoint_sha256=checkpoint_hash,
        device=device,
        video_batch_size=int(config["encoded_cache"]["video_batch_size"]),
    )
    cache = EncodedContinuousCache(encoded_root, source)
    output_dir = (
        resolve(root, args.output_dir)
        if args.output_dir is not None
        else root
        / "outputs"
        / "evaluation"
        / datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    metrics = evaluate_memory_query(
        model,
        cache,
        build_loss(config, device),
        device,
        query_stride=int(config["model"]["query_stride"]),
        video_batch_size=(
            int(args.video_batch_size)
            if args.video_batch_size is not None
            else int(config["test_monitor"]["video_batch_size"])
        ),
        frame_decoder_config=decoder_config_from_mapping(config["frame_decoder"]),
        query_decoder_config=query_decoder_config_from_mapping(config["query_decoder"]),
        query_chunk_steps=int(config["training"]["query_chunk_steps"]),
        predictions_dir=output_dir / "predictions",
        limit_videos=args.limit_videos,
    )
    payload = {key: value for key, value in metrics.items() if key != "predictions_by_video"}
    payload.update(
        {
            "protocol": "official_test" if args.limit_videos is None else "test_subset_smoke",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "selected_epoch": int(checkpoint.get("epoch", -1)),
            "config": str(config_path),
        }
    )
    metrics_path = output_dir / "metrics.json"
    atomic_json(metrics_path, payload)
    compact = compact_mqtcn_metrics(payload)
    print(f"Official Test videos: {payload['evaluated_videos']}")
    print(f"Levenshtein accuracy: {compact['levenshtein_accuracy']:.4f}")
    print(f"Event F1@0.5: {compact['event_f1_at_0_5']:.4f}")
    print(f"False positive/min: {compact['false_positive_per_minute']:.4f}")
    print(f"Metrics: {metrics_path}")
    return metrics_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
