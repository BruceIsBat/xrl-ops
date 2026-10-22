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
        "--benchmark-dir",
        type=Path,
        default=Path("artifacts/benchmarks"),
        help="Directory in which to write benchmark artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for deterministic evaluation order.",
    )
    parser.add_argument(
        "--temp-violation-tolerance",
        type=int,
        default=10,
        help="Maximum tolerated temperature constraint violations across the benchmark.",
    )
    parser.add_argument(
        "--pressure-violation-tolerance",
        type=int,
        default=10,
        help="Maximum tolerated pressure constraint violations across the benchmark.",
    )
    return parser.parse_args()


def evaluate_policy(
    model_path: Path,
    episodes: int,
    max_steps: int,
    seed: int,
    temp_violation_tolerance: int,
    pressure_violation_tolerance: int,
) -> dict[str, object]:
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

    reward_array = np.asarray(rewards, dtype=np.float64)
    mean_reward = float(np.mean(reward_array)) if reward_array.size else 0.0
    std_reward = float(np.std(reward_array, ddof=0)) if reward_array.size else 0.0
    total_violations = temperature_violations + pressure_violations
    safety_status = (
        "PASS"
        if temperature_violations <= temp_violation_tolerance
        and pressure_violations <= pressure_violation_tolerance
        else "FAIL"
    )

    return {
        "episodes": episodes,
        "max_steps": max_steps,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": float(np.min(reward_array)) if reward_array.size else 0.0,
        "max_reward": float(np.max(reward_array)) if reward_array.size else 0.0,
        "total_constraint_violations": total_violations,
        "constraint_violations": {
            "temperature_exceeded": temperature_violations,
            "pressure_exceeded": pressure_violations,
        },
        "safe_thresholds": {
            "temperature": MAX_SAFE_TEMPERATURE,
            "pressure": MAX_SAFE_PRESSURE,
        },
        "safety_check": {
            "status": safety_status,
            "temperature_violation_tolerance": temp_violation_tolerance,
            "pressure_violation_tolerance": pressure_violation_tolerance,
        },
        "per_episode_rewards": [float(result["reward"]) for result in per_episode_summary],
    }


def build_markdown_report(report: dict[str, object]) -> str:
    safety_status = str(report["safety_check"]["status"])  # type: ignore[index]
    temp_violations = int(report["constraint_violations"]["temperature_exceeded"])  # type: ignore[index]
    pressure_violations = int(report["constraint_violations"]["pressure_exceeded"])  # type: ignore[index]
    total_violations = int(report["total_constraint_violations"])  # type: ignore[index]
    mean_reward = float(report["mean_reward"])  # type: ignore[index]
    std_reward = float(report["std_reward"])  # type: ignore[index]
    temp_limit = int(report["safe_thresholds"]["temperature"])  # type: ignore[index]
    pressure_limit = int(report["safe_thresholds"]["pressure"])  # type: ignore[index]
    temp_tol = int(report["safety_check"]["temperature_violation_tolerance"])  # type: ignore[index]
    pressure_tol = int(report["safety_check"]["pressure_violation_tolerance"])  # type: ignore[index]

    return f"""# XRL-Ops Benchmark Report

## Summary
- Mean reward: {mean_reward:.4f}
- Reward standard deviation: {std_reward:.4f}
- Total constraint violations: {total_violations}
- Temperature violations: {temp_violations}
- Pressure violations: {pressure_violations}
- Safety check: {safety_status}

## Safety Assessment
The benchmark is marked as **{safety_status}** based on the configured violation tolerances for the control envelope.

- Temperature threshold: {temp_limit:.1f}
- Pressure threshold: {pressure_limit:.1f}
- Pass condition: temperature violations <= {temp_tol} and pressure violations <= {pressure_tol}

## Notes
This report is generated from empirical rollouts of the trained PPO policy under fixed environment seeds. It is intended to support safe deployment review, failure analysis, and policy auditability.
"""


def main() -> int:
    args = parse_args()
    report = evaluate_policy(
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        temp_violation_tolerance=args.temp_violation_tolerance,
        pressure_violation_tolerance=args.pressure_violation_tolerance,
    )

    args.benchmark_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.benchmark_dir / "metrics.json"
    markdown_path = args.benchmark_dir / "benchmark_report.md"

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved benchmark metrics to: {metrics_path}")
    print(f"Saved benchmark report to: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
