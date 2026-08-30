"""One valid JSON file per run containing Train and Test-monitor epochs."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch

from utils.io import atomic_json


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_torch_save(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)


class TrainingJSONLogger:
    def __init__(
        self,
        path: Union[str, Path],
        *,
        run: Mapping[str, Any],
        resume: bool = False,
    ) -> None:
        self.path = Path(path)
        if resume and self.path.is_file():
            import json

            self.payload: Dict[str, Any] = json.loads(
                self.path.read_text(encoding="utf-8")
            )
            if str(self.payload["run"]["run_id"]) != str(run["run_id"]):
                raise ValueError("resume run_id does not match existing log")
            if self.payload["run"].get("status") == "completed":
                raise ValueError("cannot resume an already completed run")
            self.payload["run"]["status"] = "running"
            self.payload["run"]["resumed_at"] = now_iso()
            for key in ("failed_epoch", "error", "finished_at"):
                self.payload["run"].pop(key, None)
            atomic_json(self.path, self.payload)
        elif self.path.exists():
            raise FileExistsError(f"training log already exists: {self.path}")
        else:
            metadata = dict(run)
            metadata.setdefault("started_at", now_iso())
            metadata["status"] = "running"
            self.payload = {
                "schema_version": "1.0",
                "run": metadata,
                "epochs": [],
                "final": None,
            }
            atomic_json(self.path, self.payload)

    @property
    def completed_epochs(self) -> int:
        return max(
            (int(item["epoch"]) for item in self.payload.get("epochs", [])),
            default=0,
        )

    def append_epoch(
        self,
        *,
        epoch: int,
        learning_rate: float,
        elapsed_seconds: float,
        train: Mapping[str, Any],
        test_monitor: Optional[Mapping[str, Any]],
        warnings: Optional[Any] = None,
    ) -> None:
        epoch = int(epoch)
        existing = {int(item["epoch"]) for item in self.payload["epochs"]}
        if epoch in existing:
            raise ValueError(f"epoch {epoch} is already present in log")
        if existing and epoch != max(existing) + 1:
            raise ValueError(f"epoch {epoch} is not contiguous after {max(existing)}")
        record: Dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": float(learning_rate),
            "elapsed_seconds": float(elapsed_seconds),
            "train": dict(train),
            "test_monitor": None if test_monitor is None else dict(test_monitor),
        }
        if warnings:
            record["warnings"] = warnings
        self.payload["epochs"].append(record)
        atomic_json(self.path, self.payload)

    def complete(
        self,
        *,
        selected_epoch: int,
        checkpoint_path: Union[str, Path],
        official_test_metrics: Mapping[str, Any],
    ) -> None:
        if int(selected_epoch) != self.completed_epochs:
            raise ValueError(
                f"selected_epoch={selected_epoch} does not equal logged "
                f"last epoch={self.completed_epochs}"
            )
        self.payload["run"]["status"] = "completed"
        self.payload["final"] = {
            "selected_epoch": int(selected_epoch),
            "selection_reason": "fixed_last_epoch",
            "checkpoint_path": str(checkpoint_path),
            "official_test_metrics": dict(official_test_metrics),
            "finished_at": now_iso(),
        }
        atomic_json(self.path, self.payload)

    def fail(self, *, epoch: int, error: str) -> None:
        self.payload["run"]["status"] = "failed"
        self.payload["run"]["failed_epoch"] = int(epoch)
        self.payload["run"]["error"] = str(error)
        self.payload["run"]["finished_at"] = now_iso()
        atomic_json(self.path, self.payload)


__all__ = ["TrainingJSONLogger", "atomic_torch_save", "now_iso"]
