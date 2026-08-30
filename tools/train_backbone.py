"""Train the released causal frame backbone on official IPN Train/Test."""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.continuous import (
    ContinuousCache,
    StratifiedContinuousClipDataset,
    continuous_cache_signature,
    prepare_continuous_cache,
)
from engine.backbone_evaluator import compact_continuous_metrics, evaluate_full_videos
from engine.json_logger import TrainingJSONLogger, atomic_torch_save
from engine.backbone_trainer import (
    BaselineLoss,
    BaselineLossConfig,
    boundary_positive_weight,
    frame_class_weights,
    seed_everything,
    train_one_epoch,
)
from models.baseline import ContinuousBaseline
from preprocessing.feature_builder import FeatureBuilderConfig, SkeletonFeatureBuilder
from preprocessing.augmentation import SkeletonAugmentationConfig
from streaming.decoder import decoder_config_from_mapping
from utils.io import atomic_json, atomic_text, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/train_backbone.yaml"))
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--seed", type=int, help="Override seed and record it in the effective config snapshot")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config(root: Path, path: Path) -> Tuple[Mapping[str, Any], Path]:
    resolved = path if path.is_absolute() else root / path
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"config must be a mapping: {resolved}")
    return payload, resolved.resolve()


def feature_builder_config(payload: Mapping[str, Any]) -> FeatureBuilderConfig:
    raw = payload["feature_builder"]
    return FeatureBuilderConfig(
        preprocessing_profile=str(raw.get("preprocessing_profile", "p3")),
        coordinate_source=str(raw.get("coordinate_source", "image_xyz")),
        motion_lags=tuple(int(value) for value in raw.get("motion_lags", [1])),
        max_hold_frames=int(raw.get("max_hold_frames", 5)),
        missing_clip_frames=int(raw.get("missing_clip_frames", 30)),
        include_handedness=bool(raw.get("include_handedness", True)),
    )


def choose_device(requested: str) -> torch.device:
    value = str(requested).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def skeleton_augmentation_config(
    payload: Mapping[str, Any], builder_config: FeatureBuilderConfig
) -> Tuple[float, Optional[SkeletonAugmentationConfig]]:
    raw = payload.get("augmentation", {})
    if not isinstance(raw, Mapping) or not bool(raw.get("enabled", False)):
        return 0.0, None
    probability = float(raw.get("probability", 1.0))
    config = SkeletonAugmentationConfig(
        coordinate_noise=float(raw.get("coordinate_noise", 0.0)),
        joint_dropout=float(raw.get("joint_dropout", 0.0)),
        frame_dropout=float(raw.get("frame_dropout", 0.0)),
        scale_range=tuple(float(value) for value in raw.get("scale_range", [1.0, 1.0])),
        translation_range=tuple(
            float(value) for value in raw.get("translation_range", [0.0, 0.0])
        ),
        speed_range=tuple(float(value) for value in raw.get("speed_range", [1.0, 1.0])),
        max_hold_frames=builder_config.max_hold_frames,
        missing_clip_frames=builder_config.missing_clip_frames,
    )
    return probability, config


def save_report(
    path: Path,
    *,
    architecture: str,
    run_id: str,
    model: ContinuousBaseline,
    metrics: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    events = metrics["events"]
    frame = metrics["frame"]
    boundary = events["boundary"]
    delay = events["delay"]
    lines = [
        f"# {architecture.upper()} Continuous Baseline Report",
        "",
        f"- Run: `{run_id}`",
        f"- Parameters: {model.parameter_count:,}",
        f"- Receptive field: {model.receptive_field} frames",
        f"- Fixed checkpoint: `{checkpoint_path.resolve()}`",
        "- Selection: fixed last epoch; Test monitor was not used for checkpoint selection.",
        "- Decoder parameters were fixed in the config before Official Test evaluation.",
        "",
        "## Official Test continuous metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Frame Accuracy | {frame['accuracy']:.4f} |",
        f"| Frame Macro-F1 | {frame['macro_f1']:.4f} |",
        f"| Levenshtein Accuracy | {events['levenshtein_accuracy']:.4f} |",
        f"| Event F1@0.3 | {events['event_f1']['0.3']['f1']:.4f} |",
        f"| Event F1@0.5 | {events['event_f1']['0.5']['f1']:.4f} |",
        f"| Event F1@0.7 | {events['event_f1']['0.7']['f1']:.4f} |",
        f"| False Positive/min | {events['false_positive_per_minute']:.4f} |",
        f"| Start MAE (frames) | {boundary['start_mae_frames']:.2f} |",
        f"| End MAE (frames) | {boundary['end_mae_frames']:.2f} |",
        f"| Completion delay (frames) | {delay['completion_delay_frames']:.2f} |",
        f"| Predicted / GT events | {events['predicted_events']} / {events['ground_truth_events']} |",
        "",
        "> Results use complete videos and are continuous-recognition metrics, not GT-cropped isolated classification.",
    ]
    atomic_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    root = args.project_root.resolve()
    config, config_path = load_config(root, args.config)
    config = copy.deepcopy(dict(config))
    if args.seed is not None:
        config["seed"] = int(args.seed)
    seed = int(config.get("seed", 1234))
    seed_everything(seed)
    builder = SkeletonFeatureBuilder(feature_builder_config(config))
    augmentation_probability, augmentation_config = skeleton_augmentation_config(
        config, builder.config
    )
    target_config = config["targets"]
    boundary_radius = int(target_config.get("boundary_radius", 2))
    signature = continuous_cache_signature(
        root, builder.config, boundary_radius=boundary_radius
    )
    cache_root = root / "outputs" / "cache" / "continuous" / signature
    for split in ("train", "test"):
        report = prepare_continuous_cache(
            project_root=root,
            cache_root=cache_root,
            split=split,
            builder=builder,
            boundary_radius=boundary_radius,
            signature=signature,
        )
        print(
            f"continuous cache {split}: {report['total_frames']} frames, "
            f"{len(report['videos'])} videos",
            flush=True,
        )
    if args.prepare_only:
        print(f"cache ready: {cache_root}", flush=True)
        return

    training = config["training"]
    model_config = config["model"]
    clip_length = int(training["clip_length"])
    supervised_length = int(training["supervised_length"])
    actual_context = clip_length - supervised_length
    minimum_context = int(training.get("minimum_warmup_context", 126))
    if actual_context < minimum_context:
        raise ValueError(
            f"actual context {actual_context} < required minimum {minimum_context}"
        )
    epochs = 2 if args.smoke else int(training["epochs"])
    samples_per_epoch = 64 if args.smoke else int(training["samples_per_epoch"])
    batch_size = min(16, int(training["batch_size"])) if args.smoke else int(training["batch_size"])
    test_limit = 2 if args.smoke else None
    device = choose_device(args.device)
    amp = bool(training.get("amp", False)) and device.type == "cuda"
    train_cache = ContinuousCache(cache_root, "train")
    test_cache = ContinuousCache(cache_root, "test")
    if builder.output_dim != int(model_config["input_dim"]):
        raise ValueError(
            f"feature dim {builder.output_dim} != config input {model_config['input_dim']}"
        )
    dataset = StratifiedContinuousClipDataset(
        train_cache,
        clip_length=clip_length,
        supervised_length=supervised_length,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
        center_jitter=int(training.get("center_jitter", 32)),
        missing_augmentation_probability=float(
            training.get("missing_augmentation_probability", 0.0)
        ),
        missing_augmentation_max_frames=int(
            training.get("missing_augmentation_max_frames", 12)
        ),
        skeleton_augmentation_probability=augmentation_probability,
        skeleton_augmentation_config=augmentation_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = ContinuousBaseline(
        builder.output_dim,
        architecture="b1",
        hidden_dim=int(model_config["hidden_dim"]),
        num_classes=int(model_config.get("num_classes", 14)),
        kernel_size=int(model_config.get("kernel_size", 3)),
        dilations=tuple(int(value) for value in model_config.get("dilations", [1, 2, 4, 8, 16, 32])),
        dropout=float(model_config.get("dropout", 0.1)),
    ).to(device)
    train_counts = np.asarray(train_cache.index["frame_class_counts"], dtype=np.int64)
    weights = frame_class_weights(train_counts)
    loss_weights = training.get("loss_weights", {})
    criterion = BaselineLoss(
        torch.from_numpy(weights).to(device),
        start_positive_weight=boundary_positive_weight(
            int(train_cache.index["total_frames"]),
            int(train_cache.index["start_positive_frames"]),
        ),
        end_positive_weight=boundary_positive_weight(
            int(train_cache.index["total_frames"]),
            int(train_cache.index["end_positive_frames"]),
        ),
        config=BaselineLossConfig(
            frame_weight=float(loss_weights.get("frame", 1.0)),
            start_weight=float(loss_weights.get("start", 0.5)),
            end_weight=float(loss_weights.get("end", 0.5)),
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    decoder_config = decoder_config_from_mapping(config["decoder"])

    default_run_id = (
        f"backbone_seed{seed}_" + datetime.now().strftime("%Y%m%dT%H%M%S")
        + ("_smoke" if args.smoke else "")
    )
    run_id = args.run_id or default_run_id
    run_dir = root / "outputs" / "training" / "backbone" / run_id
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
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
            "model": "causal_backbone",
            "seed": seed,
            "split_protocol": "official_train_test",
            "checkpoint_policy": "last_epoch",
            "test_monitor_for_checkpoint_selection": False,
            "config_path": str(config_path),
            "config_hash": file_sha256(config_snapshot_path),
            "source_config_hash": file_sha256(config_path),
            "effective_config_path": str(config_snapshot_path.resolve()),
            "blind_holdout_claim": False,
            "official_test_history_note": (
                "Official Test was observed in earlier stages and remains monitor-only; "
                "it is not used for checkpoint selection or early stopping."
            ),
            "cache_signature": signature,
            "device": str(device),
            "torch_version": torch.__version__,
            "feature_dim": builder.output_dim,
            "receptive_field": model.receptive_field,
            "parameter_count": model.parameter_count,
            "epochs_fixed_before_training": epochs,
            "actual_warmup_context": actual_context,
            "minimum_warmup_context": minimum_context,
            "decoder_source": "fixed_config_before_official_test",
            "preprocessing_profile": builder.config.preprocessing_profile,
            "skeleton_augmentation": dict(config.get("augmentation", {"enabled": False})),
            "boundary_head_enabled": True,
            "smoke": bool(args.smoke),
        },
        resume=args.resume,
    )
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if logger.completed_epochs != start_epoch - 1:
            raise ValueError("checkpoint/log epoch mismatch during resume")

    final_metrics: Optional[Mapping[str, Any]] = None
    try:
        for epoch in range(start_epoch, epochs + 1):
            dataset.set_epoch(epoch)
            started = time.perf_counter()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_metrics = train_one_epoch(
                model,
                loader,
                optimizer,
                criterion,
                device,
                amp=amp,
                gradient_clip=float(training.get("gradient_clip", 1.0)),
            )
            test_metrics = evaluate_full_videos(
                model,
                test_cache,
                criterion,
                device,
                decoder_config,
                chunk_size=int(config["evaluation"].get("chunk_size", 2048)),
                amp=amp,
                video_batch_size=int(config["evaluation"].get("video_batch_size", 8)),
                limit_videos=test_limit,
            )
            scheduler.step()
            atomic_torch_save(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "architecture": "causal_backbone",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                    "cache_signature": signature,
                    "frame_class_weights": weights,
                },
            )
            compact = compact_continuous_metrics(test_metrics)
            logger.append_epoch(
                epoch=epoch,
                learning_rate=learning_rate,
                elapsed_seconds=time.perf_counter() - started,
                train=train_metrics,
                test_monitor=compact,
            )
            print(
                f"Backbone epoch {epoch:03d}/{epochs:03d} "
                f"train_loss={train_metrics['loss_total']:.4f} "
                f"test_frame_f1={compact['frame_macro_f1']:.4f} "
                f"test_lev={compact['levenshtein_accuracy']:.4f} "
                f"test_event_f1@0.5={compact['event_f1_at_0_5']:.4f}",
                flush=True,
            )
            final_metrics = test_metrics
        if final_metrics is None:
            raise RuntimeError("training finished without evaluation")
        final_metrics = evaluate_full_videos(
            model,
            test_cache,
            criterion,
            device,
            decoder_config,
            chunk_size=int(config["evaluation"].get("chunk_size", 2048)),
            amp=amp,
            video_batch_size=int(config["evaluation"].get("video_batch_size", 8)),
            predictions_dir=run_dir / "predictions",
            limit_videos=test_limit,
        )
        final_payload = {
            **final_metrics,
            "architecture": "causal_backbone",
            "run_id": run_id,
            "selected_epoch": epochs,
            "selection_reason": "fixed_last_epoch",
            "parameter_count": model.parameter_count,
            "receptive_field": model.receptive_field,
            "cache_signature": signature,
        }
        atomic_json(run_dir / "metrics.json", final_payload)
        save_report(
            run_dir / "baseline_report.md",
            architecture="causal_backbone",
            run_id=run_id,
            model=model,
            metrics=final_payload,
            checkpoint_path=checkpoint_path,
        )
        logger.complete(
            selected_epoch=epochs,
            checkpoint_path=checkpoint_path.resolve(),
            official_test_metrics=compact_continuous_metrics(final_payload),
        )
        print(f"Backbone complete: {run_dir.resolve()}", flush=True)
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
