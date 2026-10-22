from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from xrl_ops.model_adapter import IndustrialControlEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent for the XRL-Ops industrial control environment.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Number of training timesteps to run before saving the policy.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/models/ppo_control_model.zip"),
        help="Destination for the trained PPO policy artifact.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training runs.",
    )
    return parser.parse_args()


def train_policy(total_timesteps: int, model_path: Path, seed: int) -> PPO:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vec_env = DummyVecEnv([IndustrialControlEnv])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(str(model_path))
    return model


def main() -> int:
    args = parse_args()
    model = train_policy(
        total_timesteps=args.timesteps,
        model_path=args.model_path,
        seed=args.seed,
    )

    report = {
        "model_path": str(args.model_path),
        "timesteps": args.timesteps,
        "seed": args.seed,
        "policy_type": type(model.policy).__name__,
        "env": "IndustrialControlEnv",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
