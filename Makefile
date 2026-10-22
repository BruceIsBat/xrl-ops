from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from xrl_ops.model_adapter import IndustrialControlEnv


MAX_SAFE_TEMPERATURE = 180.0
MAX_SAFE_PRESSURE = 260.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy on a fixed benchmark of episodes.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/models/ppo_control_model.zip"),
        help="Path to the trained PPO policy file.",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("artifacts/benchmarks/eval_report.json"),
        help="Where to persist the benchmark report JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for deterministic evaluation order.",
    )
    return parser.parse_args()


def evaluate_policy(model_path: Path, episodes: int, max_steps: int, seed: int) -> dict[str, float | int | list[float] | dict[str, int | float]]:
    model = PPO.load(str(model_path))
    rewards: list[float] = []
    temperature_violations = 0
    pressure_violations = 0
    per_episode_summary: list[dict[str, float | int]] = []

    for episode_index in range(episodes):
        env = IndustrialControlEnv()
        observation, _ = env.reset(seed=seed + episode_index)
        total_reward = 0.0
        step_count = 0

        for _ in range(max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            step_count += 1

            if observation[0] > MAX_SAFE_TEMPERATURE:
                temperature_violations += 1
            if observation[1] > MAX_SAFE_PRESSURE:
                pressure_violations += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        per_episode_summary.append(
            {
                "episode_index": episode_index,
                "reward": float(total_reward),
                "steps": step_count,
            }
        )

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    reward_variance = float(np.var(rewards)) if rewards else 0.0

    return {
        "episodes": episodes,
        "max_steps": max_steps,
        "mean_reward": mean_reward,
        "reward_variance": reward_variance,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
        "constraint_violations": {
            "temperature_exceeded": temperature_violations,
            "pressure_exceeded": pressure_violations,
        },
        "safe_thresholds": {
            "temperature": MAX_SAFE_TEMPERATURE,
            "pressure": MAX_SAFE_PRESSURE,
        },
        "per_episode_rewards": [float(result["reward"]) for result in per_episode_summary],
    }


def main() -> int:
    args = parse_args()
    report = evaluate_policy(
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    args.benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
