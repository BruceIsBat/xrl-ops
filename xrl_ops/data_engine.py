from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data_engine import ControlState, DataEngine
from .model_adapter import RLModelAdapter
from .tracker import TrajectoryRecord, TrajectoryTracker


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: ControlState
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExplainRequest(PredictRequest):
    pass


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: dict[str, float]
    confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExplainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attribution: dict[str, float]
    top_features: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="XRL-Ops Inference API",
    version="0.1.0",
    description="High-throughput inference and explanation endpoints for RL-based control agents.",
)

engine = DataEngine()
model_adapter = RLModelAdapter()
tracker = TrajectoryTracker()


def _state_vector(state: ControlState) -> np.ndarray:
    return np.asarray(
        [
            state.temperature,
            state.pressure,
            state.flow_rate,
            state.vibration,
            state.power_draw,
            state.setpoint,
            state.torque,
            state.coolant_temp,
        ],
        dtype=np.float32,
    )


def _action_dict(action_vector: np.ndarray) -> dict[str, float]:
    action_names = model_adapter.action_names()
    return {
        action_name: float(value)
        for action_name, value in zip(action_names, action_vector.astype(np.float64))
    }


async def _log_prediction(request: PredictRequest, action: dict[str, float], confidence: float) -> None:
    record = TrajectoryRecord(
        episode_id=str(request.metadata.get("episode_id", "inference")),
        step_index=int(request.metadata.get("step_index", 0)),
        state=request.state.model_dump(mode="python", exclude={"timestamp"}),
        action=action,
        reward=float(confidence),
        terminated=False,
        truncated=False,
        metadata={
            "endpoint": "predict",
            "source": "stable-baselines3-ppo",
            "confidence": confidence,
        },
    )
    await asyncio.to_thread(tracker.log, record)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        prepared = engine.prepare_for_rl([request.state.model_dump(mode="python", exclude={"timestamp"})])
        feature_df = prepared["features"]
        if feature_df.height == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No valid state features were produced.",
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    state_vector = _state_vector(request.state)
    action_vector = await asyncio.to_thread(model_adapter.predict, state_vector)
    action = _action_dict(action_vector)
    confidence = float(np.clip(np.linalg.norm(action_vector) / np.sqrt(action_vector.size), 0.0, 1.0))

    await _log_prediction(request, action, confidence)

    return PredictResponse(
        action=action,
        confidence=confidence,
        metadata={
            "feature_columns": prepared["feature_columns"],
            "state_shape": feature_df.shape,
            "source": "stable-baselines3-ppo",
            "model_path": str(model_adapter.model_path),
        },
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest) -> ExplainResponse:
    try:
        prepared = engine.prepare_for_rl([request.state.model_dump(mode="python", exclude={"timestamp"})])
        feature_df = prepared["features"]
        if feature_df.height == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No valid state features were produced.",
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    state_vector = _state_vector(request.state)
    attribution = await asyncio.to_thread(model_adapter.explain, state_vector)
    top_features = sorted(attribution, key=attribution.get, reverse=True)[:5]

    return ExplainResponse(
        attribution=attribution,
        top_features=top_features,
        metadata={
            "feature_columns": prepared["feature_columns"],
            "explanation_method": "shap_kernel_explainer",
            "state_shape": feature_df.shape,
            "model_path": str(model_adapter.model_path),
        },
    )
