from __future__ import annotations

from typing import Any

__all__ = [
    "ControlState",
    "DataEngine",
    "RLModelAdapter",
    "TrajectoryRecord",
    "TrajectoryTracker",
    "app",
]


def __getattr__(name: str) -> Any:
    if name == "ControlState":
        from .data_engine import ControlState

        return ControlState
    if name == "DataEngine":
        from .data_engine import DataEngine

        return DataEngine
    if name == "RLModelAdapter":
        from .model_adapter import RLModelAdapter

        return RLModelAdapter
    if name == "TrajectoryRecord":
        from .tracker import TrajectoryRecord

        return TrajectoryRecord
    if name == "TrajectoryTracker":
        from .tracker import TrajectoryTracker

        return TrajectoryTracker
    if name == "app":
        from .inference_api import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
