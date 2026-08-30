from pathlib import Path

import numpy as np
import torch

from engine.model_trainer import MemoryQueryLoss
from models.baseline import ContinuousBaseline
from models.gesture_detection_mqtcn import GestureDetectionMQTCN
from preprocessing.feature_builder import FeatureBuilderConfig, SkeletonFeatureBuilder
from preprocessing.p3_features import P3FeatureConfig, StreamingP3FeatureBuilder
from tools.preprocess.ipn_source import load_official_frame_counts, read_annotations


def _landmarks(frames: int) -> np.ndarray:
    values = np.zeros((frames, 21, 3), dtype=np.float32)
    for frame in range(frames):
        values[frame, :, 0] = np.linspace(0.2, 0.7, 21) + frame * 0.002
        values[frame, :, 1] = np.linspace(0.3, 0.8, 21) - frame * 0.001
        values[frame, :, 2] = np.linspace(-0.05, 0.05, 21)
    return values


def test_batch_and_streaming_p3_features_are_equivalent() -> None:
    image = _landmarks(8)
    world = image.copy()
    valid = np.asarray([True, True, False, False, True, False, False, False])
    batch = SkeletonFeatureBuilder(
        FeatureBuilderConfig(
            preprocessing_profile="p3",
            coordinate_source="image_xyz",
            motion_lags=(1,),
            max_hold_frames=2,
            missing_clip_frames=30,
            include_handedness=True,
        )
    ).build(
        image_landmarks=image,
        world_landmarks=world,
        valid_mask=valid,
        width=320,
        height=240,
        metadata_hand="Right",
    )
    streaming = StreamingP3FeatureBuilder(
        P3FeatureConfig(
            coordinate_source="image_xyz",
            motion_lags=(1,),
            max_hold_frames=2,
            missing_clip_frames=30,
            include_handedness=True,
        )
    )
    rows = []
    reset = []
    for index in range(len(valid)):
        result = streaming.step(
            image_landmarks=image[index],
            world_landmarks=world[index],
            valid=bool(valid[index]),
            width=320,
            height=240,
            metadata_hand="Right",
            stream_id="test",
        )
        rows.append(result.features)
        reset.append(result.reset_required)
    np.testing.assert_allclose(np.stack(rows), batch.features, atol=1.0e-6)
    np.testing.assert_array_equal(np.asarray(reset), batch.reset_mask)


def test_final_query_path_backpropagates_without_unfreezing_backbone() -> None:
    torch.manual_seed(7)
    baseline = ContinuousBaseline(
        8,
        architecture="b1",
        hidden_dim=16,
        num_classes=14,
        kernel_size=3,
        dilations=(1, 2),
        dropout=0.0,
    )
    model = GestureDetectionMQTCN(
        baseline,
        num_queries=2,
        num_query_classes=14,
        attention_heads=4,
        decoder_layers=1,
        feedforward_dim=32,
        frame_memory_length=4,
        dropout=0.0,
    )
    encoded = torch.randn(2, 10, 16)
    query_times = torch.tensor([[4, 8, 10], [4, 8, 10]])
    active = torch.tensor([[True, True, True], [True, True, False]])
    output = model.query_sequence(
        encoded,
        torch.ones(2, 10, dtype=torch.bool),
        query_times,
        active_mask=active,
    )
    assert output["pred_logits"].shape == (2, 3, 2, 14)
    assert output["pred_intervals"].shape == (2, 3, 2, 2)
    target_valid = torch.zeros(2, 3, 2, dtype=torch.bool)
    target_valid[0, 1, 0] = True
    targets = {
        "query_step_valid": active,
        "target_valid": target_valid,
        "target_classes": torch.tensor(
            [[[-1, -1], [3, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1]]]
        ),
        "target_boundaries": torch.zeros(2, 3, 2, 2),
        "left_censored": torch.zeros(2, 3, 2, dtype=torch.bool),
    }
    targets["target_boundaries"][0, 1, 0] = torch.tensor([0.25, 0.75])
    losses = MemoryQueryLoss()(output, targets)
    losses["loss_total"].backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.query_decoder.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model.baseline.parameters())


def test_official_source_parsers(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "Annot_List.txt").write_text(
        "video,label,id,t_start,t_end,frames\nV1,D0X,1,1,3,3\n",
        encoding="utf-8",
    )
    (annotations / "Video_TrainList.txt").write_text("V1 3\n", encoding="utf-8")
    (annotations / "Video_TestList.txt").write_text("V2 4\n", encoding="utf-8")
    rows = read_annotations(annotations / "Annot_List.txt")
    assert [(row.video, row.start, row.end) for row in rows] == [("V1", 1, 3)]
    assert load_official_frame_counts(annotations) == {"V1": 3, "V2": 4}
