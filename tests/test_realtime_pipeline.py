from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from models.baseline import ContinuousBaseline
from models.gesture_detection_mqtcn import GestureDetectionMQTCN
from preprocessing.p3_features import P3FeatureConfig
from streaming.realtime_pipeline import RealtimePipeline
from streaming.decoder import DecoderConfig
from streaming.event_tracker import ContinuousEvent
from streaming.query_tracker import QueryDecoderConfig
from tools.realtime_demo import _feature_config, _load_config, _load_model
from utils.io import file_sha256


class RealtimePipelineTests(unittest.TestCase):
    def test_final_artifacts_load_without_training_code(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config, _ = _load_config(root, Path("configs/runtime.yaml"))
        checkpoint = root / str(config["artifact"]["checkpoint"])
        hand_model = root / str(config["hand_landmarker"]["model_path"])
        model, metadata = _load_model(
            root, config, checkpoint, torch.device("cpu")
        )
        self.assertEqual(model.architecture, "gesture_detection_mqtcn")
        self.assertEqual(metadata["parameter_count"], 373408)
        self.assertEqual(
            metadata["checkpoint_sha256"],
            config["artifact"]["checkpoint_sha256"],
        )
        self.assertEqual(
            file_sha256(hand_model), config["hand_landmarker"]["model_sha256"]
        )
        self.assertEqual(
            135,
            RealtimePipeline(
                model,
                feature_config=_feature_config(config),
                frame_decoder_config=DecoderConfig(),
                query_decoder_config=QueryDecoderConfig(),
                query_stride=int(config["model"]["query_stride"]),
                device=torch.device("cpu"),
            ).feature_builder.output_dim,
        )

    def build_pipeline(self) -> RealtimePipeline:
        torch.manual_seed(19)
        feature_config = P3FeatureConfig(
            coordinate_source="image_xyz",
            motion_lags=(1,),
            max_hold_frames=5,
            missing_clip_frames=30,
            include_handedness=True,
        )
        baseline = ContinuousBaseline(
            135,
            architecture="b1",
            hidden_dim=16,
            kernel_size=3,
            dilations=(1, 2),
            dropout=0.0,
        ).eval()
        model = GestureDetectionMQTCN(
            baseline,
            num_queries=3,
            attention_heads=4,
            feedforward_dim=32,
            frame_memory_length=12,
            dropout=0.0,
        ).eval()
        return RealtimePipeline(
            model,
            feature_config=feature_config,
            frame_decoder_config=DecoderConfig(),
            query_decoder_config=QueryDecoderConfig(),
            query_stride=4,
            device=torch.device("cpu"),
        )

    @staticmethod
    def landmarks() -> tuple[np.ndarray, np.ndarray]:
        joints = np.arange(21, dtype=np.float32)
        image = np.stack(
            (0.20 + joints * 0.012, 0.30 + joints * 0.006, joints * 0.001),
            axis=1,
        )
        world = image.copy()
        return image, world

    def step(
        self, pipeline: RealtimePipeline, frame: int, *, valid: bool
    ):
        image, world = self.landmarks()
        if not valid:
            image.fill(0.0)
            world.fill(0.0)
        return pipeline.process_skeleton_frame(
            image_landmarks=image,
            world_landmarks=world,
            valid=valid,
            width=640,
            height=480,
            frame_index=frame,
            metadata_hand="Right",
            stream_id="video",
        )

    def test_query_cadence_and_atomic_missing_reset(self) -> None:
        pipeline = self.build_pipeline()
        results = [self.step(pipeline, frame, valid=True) for frame in range(4)]
        self.assertEqual(
            [item.model.query_executed for item in results],
            [False, False, False, True],
        )
        for frame in range(4, 9):
            result = self.step(pipeline, frame, valid=False)
            self.assertEqual(result.status, "processed")
            self.assertTrue(result.feature.held_last)
            self.assertFalse(result.feature.observed_valid)
        reset = self.step(pipeline, 9, valid=False)
        self.assertEqual(reset.status, "reset_missing")
        self.assertIsNone(reset.model)
        waiting = self.step(pipeline, 10, valid=False)
        self.assertEqual(waiting.status, "waiting_for_skeleton")
        resumed = self.step(pipeline, 11, valid=True)
        self.assertEqual(resumed.status, "processed_after_reset")
        state = pipeline.runtime.backbone.get_stream_state("video")
        self.assertIsNotNone(state)
        self.assertEqual(state.processed_frames, 1)

    def test_finalized_events_survive_missing_reset_but_not_source_reset(self) -> None:
        pipeline = self.build_pipeline()
        self.step(pipeline, 0, valid=True)
        decoder = pipeline.runtime._frame_decoders["video"]
        decoder.tracker.add(
            ContinuousEvent(
                class_id=2,
                start_frame=0,
                end_frame_exclusive=8,
                score=0.8,
                emitted_at_frame=8,
            )
        )
        query_event = {
            "class_id": 2,
            "start_frame": 0,
            "end_frame_exclusive": 8,
            "score": 0.9,
            "emitted_at_frame": 8,
            "emitted_at_prefix": 9,
            "source": "query",
            "query_slot": 0,
        }
        pipeline.runtime._query_trackers["video"].add(query_event)
        pipeline.runtime._fusion_trackers["video"].add(
            {**query_event, "source": "fusion"}
        )
        for frame in range(1, 7):
            self.step(pipeline, frame, valid=False)

        tracks = pipeline.events_for_stream("video")
        self.assertEqual(len(tracks["frame"]), 1)
        self.assertEqual(len(tracks["query"]), 1)
        self.assertEqual(len(tracks["fusion"]), 1)
        mutable_copy = tracks["fusion"][0]
        mutable_copy["score"] = -1.0
        self.assertEqual(
            pipeline.events_for_stream("video")["fusion"][0]["score"], 0.9
        )

        pipeline.reset_stream("video", next_frame_index=20)
        self.assertEqual(pipeline.events_for_stream("video")["fusion"], ())


if __name__ == "__main__":
    unittest.main()
