from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ControlState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    temperature: float = Field(..., ge=0.0, le=200.0)
    pressure: float = Field(..., ge=0.0, le=300.0)
    flow_rate: float = Field(..., ge=0.0, le=500.0)
    vibration: float = Field(..., ge=0.0, le=50.0)
    power_draw: float = Field(..., ge=0.0, le=2000.0)
    setpoint: float = Field(..., ge=0.0, le=200.0)
    torque: float = Field(..., ge=0.0, le=500.0)
    coolant_temp: float = Field(..., ge=0.0, le=150.0)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise TypeError("timestamp must be a datetime or ISO-8601 string")


class FeatureWindowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lag_features: list[int] = Field(default_factory=lambda: [1, 3, 5])
    rolling_windows: list[int] = Field(default_factory=lambda: [3, 5, 10])
    required_columns: list[str] = Field(
        default_factory=lambda: [
            "temperature",
            "pressure",
            "flow_rate",
            "vibration",
            "power_draw",
            "setpoint",
            "torque",
            "coolant_temp",
        ]
    )


@dataclass(slots=True)
class DataEngine:
    config: FeatureWindowConfig = field(default_factory=FeatureWindowConfig)

    def validate_records(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        validated: list[dict[str, Any]] = []
        for item in records:
            if not isinstance(item, dict):
                raise TypeError(f"Expected dict-like record, got {type(item).__name__}")
            validated.append(ControlState.model_validate(item).model_dump(mode="python"))
        return validated

    def to_polars(self, records: Iterable[dict[str, Any]]) -> pl.DataFrame:
        validated = self.validate_records(records)
        df = pl.DataFrame(validated)

        missing = [column for column in self.config.required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df.sort("timestamp")

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        working = df.clone()

        for column in self.config.required_columns:
            working = working.with_columns(pl.col(column).cast(pl.Float64).alias(column))

        for lag in self.config.lag_features:
            for column in self.config.required_columns:
                working = working.with_columns(
                    pl.col(column).shift(lag).alias(f"{column}_lag_{lag}"),
                    (pl.col(column) - pl.col(column).shift(lag)).alias(f"{column}_delta_{lag}"),
                )

        for window in self.config.rolling_windows:
            for column in self.config.required_columns:
                working = working.with_columns(
                    pl.col(column)
                    .rolling_mean(window_size=window, min_periods=1)
                    .alias(f"{column}_rolling_mean_{window}"),
                    pl.col(column)
                    .rolling_std(window_size=window, min_periods=1)
                    .alias(f"{column}_rolling_std_{window}"),
                )

        for column in self.config.required_columns:
            mean_value = working[column].mean()
            std_value = working[column].std()
            if std_value is None or std_value == 0:
                std_value = 1.0
            working = working.with_columns(
                ((pl.col(column) - mean_value) / std_value).alias(f"{column}_zscore")
            )

        return working

    def prepare_for_rl(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        df = self.to_polars(records)
        engineered = self.engineer_features(df)
        feature_columns = [column for column in engineered.columns if column != "timestamp"]
        return {
            "raw": df,
            "features": engineered,
            "feature_columns": feature_columns,
            "n_rows": engineered.height,
            "schema": engineered.schema,
        }
