from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class TrajectoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    episode_id: str
    step_index: int = Field(..., ge=0)
    state: dict[str, float]
    action: dict[str, float]
    reward: float
    terminated: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryTracker:
    def __init__(self, base_dir: str | Path = "artifacts/trajectories") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_episode(self, episode_id: str) -> Path:
        return self.base_dir / f"{episode_id}.parquet"

    def log(self, record: TrajectoryRecord) -> Path:
        record_path = self._path_for_episode(record.episode_id)
        row = {
            "timestamp": record.timestamp.isoformat(),
            "episode_id": record.episode_id,
            "step_index": record.step_index,
            "reward": float(record.reward),
            "terminated": bool(record.terminated),
            "truncated": bool(record.truncated),
            "state": json.dumps(record.state, separators=(",", ":")),
            "action": json.dumps(record.action, separators=(",", ":")),
            "metadata": json.dumps(record.metadata, separators=(",", ":")),
        }

        if record_path.exists():
            existing = pl.read_parquet(record_path)
            combined = pl.concat([existing, pl.DataFrame([row])], how="vertical_relaxed")
            combined.write_parquet(record_path)
        else:
            pl.DataFrame([row]).write_parquet(record_path)

        return record_path

    def log_batch(self, records: Iterable[TrajectoryRecord]) -> list[Path]:
        paths: list[Path] = []
        for record in records:
            paths.append(self.log(record))
        return paths

    def read_episode(self, episode_id: str) -> pl.DataFrame:
        record_path = self._path_for_episode(episode_id)
        if not record_path.exists():
            raise FileNotFoundError(f"No trajectory data found for episode: {episode_id}")
        return pl.read_parquet(record_path)
