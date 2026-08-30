"""Train the released Memory-Query model with a frozen causal backbone."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
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
from engine.json_logger import TrainingJSONLogger, atomic_torch_save
from engine.model_evaluator import compact_mqtcn_metrics, evaluate_memory_query
from engine.model_trainer import (
    MemoryQueryLoss,
    MemoryQueryLossConfig,
    train_memory_query_epoch,
)
from engine.backbone_trainer import seed_everything
from models.baseline import ContinuousBaseline
from models.gesture_detection_mqtcn import GestureDetectionMQTCN
from models.query.matcher import MatchCost
from preprocessing.feature_builder import FeatureBuilderConfig, SkeletonFeatureBuilder
from streaming.decoder import decoder_config_from_mapping
from streaming.query_tracker import query_decoder_config_from_mapping
from utils.io import atomic_json, atomic_text, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        help="Override backbone.checkpoint in the training config.",
    )
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--seed", type=int, help="Override seed and record it in the effective config snapshot")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    requested = str(value).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_config(root: Path, path: Path) -> Tuple[Mapping[str, Any], Path]:
    resolved = path if path.is_absolute() else root / path
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("MQTCN config must be a mapping")
    return payload, resolved.resolve()


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


def tensor_digest(state: Mapping[str, torch.Tensor], prefix: str) -> str:
    digest = hashlib.sha256()
    for key in sorted(value for value in state if value.startswith(prefix)):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_baseline(
    checkpoint: Mapping[str, Any], input_dim: int, device: torch.device
) -> ContinuousBaseline:
    source_config = checkpoint["config"]["model"]
    model = ContinuousBaseline(
        input_dim,
        architecture="b1",
        hidden_dim=int(source_config["hidden_dim"]),
        num_classes=int(source_config.get("num_classes", 14)),
        kernel_size=int(source_config.get("kernel_size", 3)),
        dilations=tuple(int(value) for value in source_config["dilations"]),
        dropout=float(source_config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def build_model(
    baseline: ContinuousBaseline,
    payload: Mapping[str, Any],
) -> GestureDetectionMQTCN:
    raw = payload["model"]
    return GestureDetectionMQTCN(
        baseline,
        num_queries=int(raw["num_queries"]),
        num_query_classes=int(raw["num_query_classes"]),
        attention_heads=int(raw["attention_heads"]),
        decoder_layers=int(raw["decoder_layers"]),
        feedforward_dim=int(raw["feedforward_dim"]),
        frame_memory_length=int(raw["frame_memory_length"]),
        dropout=float(raw["dropout"]),
    )


def save_model_report(
    path: Path,
    *,
    model: GestureDetectionMQTCN,
    run_id: str,
    metrics: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    lines = [
        "# GestureDetection-MQTCN Training Report",
        "",
        f"- Run: `{run_id}`",
        f"- Parameters: {model.parameter_count:,}",
        f"- Trainable parameters: {model.trainable_parameter_count:,}",
        f"- Fixed checkpoint: `{checkpoint_path.resolve()}`",
        "- The causal backbone and frame/boundary heads are frozen after epoch 80.",
        "- Selection uses the fixed final epoch; Official Test is monitor-only.",
        "",
        "| Output | Lev.Acc | Event F1@0.5 | FP/min | Predicted events |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("frame_only", "query_only", "fusion"):
        output = metrics["outputs"][name]
        lines.append(
            f"| {name} | {output['levenshtein_accuracy']:.4f} | "
            f"{output['event_f1']['0.5']['f1']:.4f} | "
            f"{output['false_positive_per_minute']:.4f} | "
            f"{output['predicted_events']} |"
        )
    atomic_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    root = args.project_root.resolve()
    config, config_path = load_config(root, args.config)
    config = copy.deepcopy(dict(config))
    if args.seed is not None:
        config["seed"] = int(args.seed)
    seed = int(config["seed"])
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = choose_device(args.device)
    builder = SkeletonFeatureBuilder(feature_config(config))
    boundary_radius = int(config["targets"]["boundary_radius"])
    continuous_signature = continuous_cache_signature(
        root, builder.config, boundary_radius=boundary_radius
    )
    continuous_root = root / "outputs/cache/continuous" / continuous_signature
    for split in ("train", "test"):
        prepare_continuous_cache(
            project_root=root,
            cache_root=continuous_root,
            split=split,
            builder=builder,
            boundary_radius=boundary_radius,
            signature=continuous_signature,
        )
    train_source = ContinuousCache(continuous_root, "train")
    test_source = ContinuousCache(continuous_root, "test")

    backbone_value = (
        args.backbone_checkpoint
        if args.backbone_checkpoint is not None
        else Path(str(config["backbone"]["checkpoint"]))
    )
    backbone_path = (
        backbone_value.resolve()
        if backbone_value.is_absolute()
        else (root / backbone_value).resolve()
    )
    expected_hash = str(config["backbone"]["checkpoint_sha256"]).lower()
    actual_hash = file_sha256(backbone_path).lower()
    if args.backbone_checkpoint is None and expected_hash and actual_hash != expected_hash:
        raise ValueError("frozen backbone checkpoint SHA-256 differs from config")
    try:
        recorded_backbone = backbone_path.relative_to(root).as_posix()
    except ValueError:
        recorded_backbone = str(backbone_path)
    config["backbone"] = dict(config["backbone"])
    config["backbone"]["checkpoint"] = recorded_backbone
    config["backbone"]["checkpoint_sha256"] = actual_hash
    checkpoint = torch.load(backbone_path, map_location=device, weights_only=False)
    baseline = build_baseline(checkpoint, builder.output_dim, device)
    encoded_signature = encoded_cache_signature(continuous_signature, actual_hash)
    encoded_root = root / "outputs/cache/memory_query" / encoded_signature
    for source in (train_source, test_source):
        prepare_encoded_cache(
            cache_root=encoded_root,
            source=source,
            baseline=baseline,
            checkpoint_sha256=actual_hash,
            device=device,
            video_batch_size=int(config["encoded_cache"]["video_batch_size"]),
        )
    if args.prepare_only:
        print(f"MQ encoded cache ready: {encoded_root}", flush=True)
        return
    train_cache = EncodedContinuousCache(encoded_root, train_source)
    test_cache = EncodedContinuousCache(encoded_root, test_source)

    # Re-seed so cache preparation/reuse cannot change model initialization.
    seed_everything(seed)
    model = build_model(baseline, config).to(device)
    frozen_digest = tensor_digest(model.state_dict(), "baseline.")
    match = config["matching"]
    loss = config["loss"]
    criterion = MemoryQueryLoss(
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
    training = config["training"]
    epochs = 2 if args.smoke else int(training["epochs"])
    train_limit = 8 if args.smoke else None
    test_limit = 2 if args.smoke else None
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    frame_decoder = decoder_config_from_mapping(config["frame_decoder"])
    query_decoder = query_decoder_config_from_mapping(config["query_decoder"])

    default_run_id = (
        f"mqtcn_seed{seed}_"
        + datetime.now().strftime("%Y%m%dT%H%M%S")
        + ("_smoke" if args.smoke else "")
    )
    run_id = args.run_id or default_run_id
    run_dir = root / "outputs" / "training" / "mqtcn" / run_id
    checkpoint_path = run_dir / "checkpoints/last.pt"
    log_path = run_dir / "training_log.json"
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory already contains files: {run_dir}")
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    atomic_text(
        config_snapshot_path,
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
    )
    logger = TrainingJSONLogger(
        log_path,
        run={
            "run_id": run_id,
            "model": "gesture_detection_mqtcn",
            "seed": seed,
            "split_protocol": "official_train_test",
            "checkpoint_policy": "last_epoch",
            "test_monitor_for_checkpoint_selection": False,
            "test_informed": True,
            "blind_holdout_claim": False,
            "official_test_history_note": (
                "Official Test was observed in stages 1-4 and remains monitor-only; "
                "it is not used for checkpoint selection or early stopping."
            ),
            "config_path": str(config_path),
            "config_hash": file_sha256(config_snapshot_path),
            "source_config_hash": file_sha256(config_path),
            "effective_config_path": str(config_snapshot_path.resolve()),
            "backbone_checkpoint": str(backbone_path.resolve()),
            "backbone_checkpoint_sha256": actual_hash,
            "continuous_cache_signature": continuous_signature,
            "encoded_cache_signature": encoded_signature,
            "device": str(device),
            "torch_version": torch.__version__,
            "precision": "strict_fp32_tf32_disabled",
            "parameter_count": model.parameter_count,
            "trainable_parameter_count": model.trainable_parameter_count,
            "epochs_fixed_before_training": epochs,
            "training_protocol": training["protocol"],
            "primary_output": config["evaluation"]["primary_output"],
            "primary_metric": config["evaluation"]["primary_metric"],
            "smoke": bool(args.smoke),
        },
        resume=args.resume,
    )
    start_epoch = 1
    if args.resume:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"]) + 1
        if logger.completed_epochs != start_epoch - 1:
            raise ValueError("checkpoint/log epoch mismatch during resume")

    final_metrics: Optional[Mapping[str, Any]] = None
    try:
        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_metrics = train_memory_query_epoch(
                model,
                train_cache,
                optimizer,
                criterion,
                device,
                epoch=epoch,
                seed=seed,
                video_batch_size=(8 if args.smoke else int(training["video_batch_size"])),
                query_chunk_steps=int(training["query_chunk_steps"]),
                query_stride=int(config["model"]["query_stride"]),
                gradient_clip=float(training["gradient_clip"]),
                shuffle_video_streams=bool(training["shuffle_video_streams"]),
                limit_videos=train_limit,
            )
            test_metrics = evaluate_memory_query(
                model,
                test_cache,
                criterion,
                device,
                query_stride=int(config["model"]["query_stride"]),
                video_batch_size=(2 if args.smoke else int(config["test_monitor"]["video_batch_size"])),
                frame_decoder_config=frame_decoder,
                query_decoder_config=query_decoder,
                query_chunk_steps=int(training["query_chunk_steps"]),
                limit_videos=test_limit,
            )
            scheduler.step()
            current_digest = tensor_digest(model.state_dict(), "baseline.")
            if current_digest != frozen_digest:
                raise AssertionError("frozen backbone/head changed during MQ training")
            atomic_torch_save(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "architecture": "gesture_detection_mqtcn",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                    "backbone_checkpoint_sha256": actual_hash,
                    "frozen_baseline_digest": frozen_digest,
                },
            )
            compact = compact_mqtcn_metrics(test_metrics)
            logger.append_epoch(
                epoch=epoch,
                learning_rate=learning_rate,
                elapsed_seconds=time.perf_counter() - started,
                train=train_metrics,
                test_monitor=compact,
            )
            print(
                f"MQTCN epoch {epoch:03d}/{epochs:03d} "
                f"train={train_metrics['loss_total']:.4f} "
                f"test_lev={compact['levenshtein_accuracy']:.4f} "
                f"test_f1@0.5={compact['event_f1_at_0_5']:.4f} "
                f"pred={compact['predicted_events']}",
                flush=True,
            )
            final_metrics = test_metrics
        if final_metrics is None:
            raise RuntimeError("training finished without evaluation")
        final_metrics = evaluate_memory_query(
            model,
            test_cache,
            criterion,
            device,
            query_stride=int(config["model"]["query_stride"]),
            video_batch_size=(2 if args.smoke else int(config["test_monitor"]["video_batch_size"])),
            frame_decoder_config=frame_decoder,
            query_decoder_config=query_decoder,
            query_chunk_steps=int(training["query_chunk_steps"]),
            predictions_dir=run_dir / "predictions",
            limit_videos=test_limit,
        )
        final_payload = {
            key: value
            for key, value in final_metrics.items()
            if key != "predictions_by_video"
        }
        final_payload.update(
            {
                "architecture": "gesture_detection_mqtcn",
                "run_id": run_id,
                "selected_epoch": epochs,
                "selection_reason": "fixed_last_epoch",
                "parameter_count": model.parameter_count,
                "trainable_parameter_count": model.trainable_parameter_count,
                "backbone_checkpoint_sha256": actual_hash,
                "frozen_baseline_digest": frozen_digest,
            }
        )
        atomic_json(run_dir / "metrics.json", final_payload)
        save_model_report(
            run_dir / "memory_query_report.md",
            model=model,
            run_id=run_id,
            metrics=final_payload,
            checkpoint_path=checkpoint_path,
        )
        logger.complete(
            selected_epoch=epochs,
            checkpoint_path=checkpoint_path.resolve(),
            official_test_metrics=compact_mqtcn_metrics(final_payload),
        )
        print(f"MQTCN complete: {run_dir.resolve()}", flush=True)
    except Exception as error:
        logger.fail(
            epoch=max(start_epoch, logger.completed_epochs + 1),
            error=f"{type(error).__name__}: {error}",
        )
        raise


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
