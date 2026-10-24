from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
import shap
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

FEATURE_NAMES: list[str] = [
    "temperature",
    "pressure",
    "flow_rate",
    "vibration",
    "power_draw",
    "setpoint",
    "torque",
    "coolant_temp",
]

LOWER_BOUNDS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
UPPER_BOUNDS = np.array(
    [200.0, 300.0, 500.0, 50.0, 2000.0, 200.0, 500.0, 150.0],
    dtype=np.float32,
)
ACTION_NAMES: list[str] = [
    "coolant_flow_adjustment",
    "motor_torque_target",
    "pressure_trim",
]


class IndustrialControlEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=LOWER_BOUNDS.copy(),
            high=UPPER_BOUNDS.copy(),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng(42)
        self._last_obs = self.observation_space.sample()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last_obs = self._rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS).astype(np.float32)
        return self._last_obs.copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (3,):
            raise ValueError(f"Expected action of shape (3,), got {action_array.shape}")

        temperature = float(self._last_obs[0])
        pressure = float(self._last_obs[1])
        flow_rate = float(self._last_obs[2])
        vibration = float(self._last_obs[3])
        power_draw = float(self._last_obs[4])
        setpoint = float(self._last_obs[5])
        torque = float(self._last_obs[6])
        coolant_temp = float(self._last_obs[7])

        temperature = temperature + 2.5 * action_array[0] + self._rng.normal(0.0, 0.25)
        pressure = pressure + 8.0 * action_array[2] + self._rng.normal(0.0, 1.0)
        flow_rate = flow_rate + 10.0 * action_array[0] + self._rng.normal(0.0, 0.7)
        vibration = max(0.0, vibration + 0.8 * abs(action_array[1]) + self._rng.normal(0.0, 0.3))
        power_draw = power_draw + 25.0 * abs(action_array[1]) + self._rng.normal(0.0, 5.0)
        coolant_temp = coolant_temp + 1.8 * action_array[0] + self._rng.normal(0.0, 0.2)
        torque = torque + 35.0 * action_array[1] + self._rng.normal(0.0, 3.0)

        next_obs = np.array(
            [
                np.clip(temperature, LOWER_BOUNDS[0], UPPER_BOUNDS[0]),
                np.clip(pressure, LOWER_BOUNDS[1], UPPER_BOUNDS[1]),
                np.clip(flow_rate, LOWER_BOUNDS[2], UPPER_BOUNDS[2]),
                np.clip(vibration, LOWER_BOUNDS[3], UPPER_BOUNDS[3]),
                np.clip(power_draw, LOWER_BOUNDS[4], UPPER_BOUNDS[4]),
                np.clip(setpoint, LOWER_BOUNDS[5], UPPER_BOUNDS[5]),
                np.clip(torque, LOWER_BOUNDS[6], UPPER_BOUNDS[6]),
                np.clip(coolant_temp, LOWER_BOUNDS[7], UPPER_BOUNDS[7]),
            ],
            dtype=np.float32,
        )

        reward = (
            -abs(next_obs[0] - setpoint)
            - 0.05 * abs(next_obs[1] - setpoint)
            - 0.02 * next_obs[3]
            - 0.01 * next_obs[4]
            - 0.25 * abs(next_obs[6])
        )
        terminated = False
        truncated = False
        self._last_obs = next_obs
        return next_obs, float(reward), terminated, truncated, {}


@dataclass(slots=True)
class RLModelAdapter:
    feature_names: list[str] = field(default_factory=lambda: FEATURE_NAMES)
    model_path: Path = field(default_factory=lambda: Path("artifacts/models/ppo_control_model.zip"))
    background_data: np.ndarray = field(init=False)
    _model: PPO | None = field(init=False, default=None)
    _explainer: shap.Explainer | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_model_path = Path("artifacts/models/mock_ppo_control.zip")
        if not self.model_path.exists() and legacy_model_path.exists():
            self.model_path = legacy_model_path
        self._model = self._load_or_create_model()
        self.background_data = self._build_background_data()
        self._explainer = shap.Explainer(
            self._predict_action_batch,
            self.background_data,
            algorithm="kernel",
        )

    def _build_background_data(self) -> np.ndarray:
        rng = np.random.default_rng(123)
        samples = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(32, len(self.feature_names)))
        return samples.astype(np.float32)

    def _load_or_create_model(self) -> PPO:
        if self.model_path.exists():
            return PPO.load(str(self.model_path))

        vec_env = DummyVecEnv([IndustrialControlEnv])
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=64,
            batch_size=16,
            gamma=0.99,
            learning_rate=3e-4,
            verbose=0,
            seed=42,
        )
        model.learn(total_timesteps=256)
        model.save(str(self.model_path))
        return model

    def _predict_action_batch(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        predicted: list[np.ndarray] = []
        for row in arr:
            action, _ = self._model.predict(row, deterministic=True)
            predicted.append(np.asarray(action, dtype=np.float32))
        return np.asarray(predicted, dtype=np.float32)

    @property
    def model(self) -> PPO:
        if self._model is None:
            raise RuntimeError("Model has not been initialized.")
        return self._model

    def predict(self, state: Sequence[float]) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (len(self.feature_names),):
            raise ValueError(
                f"Expected state with {len(self.feature_names)} features, got {state_array.shape}"
            )
        action, _ = self.model.predict(state_array[np.newaxis, :], deterministic=True)
        return np.asarray(action[0], dtype=np.float32)

    def explain(self, state: Sequence[float]) -> dict[str, float]:
        state_array = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if self._explainer is None:
            raise RuntimeError("Explainer has not been initialized.")

        explanation = self._explainer(state_array)
        values = np.asarray(explanation.values)

        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim == 2:
            aggregated = np.abs(values).mean(axis=1)
        elif values.ndim == 1:
            aggregated = np.abs(values)
        else:
            aggregated = np.abs(values.reshape(values.shape[0], -1)).mean(axis=1)

        return {
            feature: float(score)
            for feature, score in zip(self.feature_names, aggregated.astype(np.float64))
        }

    def action_names(self) -> list[str]:
        return ACTION_NAMES
