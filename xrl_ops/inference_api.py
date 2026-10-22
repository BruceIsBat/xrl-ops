from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data_engine import DataEngine


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: dict[str, float] = Field(..., min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("state")
    @classmethod
    def validate_state_keys(cls, values: dict[str, float]) -> dict[str, float]:
        allowed = {
            "temperature",
            "pressure",
            "flow_rate",
            "vibration",
            "power_draw",
            "setpoint",
            "torque",
            "coolant_temp",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unsupported state keys: {sorted(unknown)}")
        return values


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


def _action_from_state(state: dict[str, float]) -> dict[str, float]:
    return {
        "coolant_flow_adjustment": float(state["setpoint"] * 0.18 + state["temperature"] * 0.12),
        "motor_torque_target": float(state["torque"] * 0.35 + state["power_draw"] * 0.05),
        "pressure_trim": float((state["pressure"] - state["setpoint"]) * 0.04),
    }


def _explain_state(state: dict[str, float]) -> dict[str, float]:
    raw_scores = {
        "temperature": abs(state.get("temperature", 0.0)),
        "pressure": abs(state.get("pressure", 0.0)),
        "flow_rate": abs(state.get("flow_rate", 0.0)),
        "vibration": abs(state.get("vibration", 0.0)),
        "power_draw": abs(state.get("power_draw", 0.0)),
        "setpoint": abs(state.get("setpoint", 0.0)),
        "torque": abs(state.get("torque", 0.0)),
        "coolant_temp": abs(state.get("coolant_temp", 0.0)),
    }
    total = sum(raw_scores.values()) or 1.0
    return {key: value / total for key, value in raw_scores.items()}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        prepared = engine.prepare_for_rl([request.state])
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

    action = _action_from_state(request.state)
    confidence = float(np.clip(np.mean(list(request.state.values())) / 100.0, 0.0, 1.0))

    return PredictResponse(
        action=action,
        confidence=confidence,
        metadata={
            "feature_columns": prepared["feature_columns"],
            "state_shape": feature_df.shape,
            "source": "deterministic_policy_stub",
        },
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest) -> ExplainResponse:
    try:
        prepared = engine.prepare_for_rl([request.state])
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

    attribution = _explain_state(request.state)
    top_features = sorted(attribution, key=attribution.get, reverse=True)[:5]

    return ExplainResponse(
        attribution=attribution,
        top_features=top_features,
        metadata={
            "feature_columns": prepared["feature_columns"],
            "explanation_method": "heuristic_attribution",
            "state_shape": feature_df.shape,
        },
    )
