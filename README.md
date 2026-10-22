# XRL-Ops: Explainable Reinforcement Learning for Control Systems

XRL-Ops is a production-grade MLOps backend for deploying, evaluating, and explaining RL agents in industrial and robotic control environments.

The project combines:
- Polars-based state ingestion and feature engineering
- Strict Pydantic validation for control states
- Async FastAPI inference endpoints for prediction and explanation
- Stable-Baselines3 PPO policy serving
- SHAP-based attribution for post-hoc explainability
- Empirical benchmark validation and safety auditing
- DVC artifact tracking and Dockerized reproducibility

## Project structure

- `xrl_ops/data_engine.py` — state validation and feature engineering
- `xrl_ops/model_adapter.py` — PPO model adapter, mock Gymnasium control environment, and SHAP explainer
- `xrl_ops/inference_api.py` — FastAPI `/predict` and `/explain` endpoints
- `xrl_ops/tracker.py` — empirical state-action-reward logging
- `train.py` — PPO training pipeline
- `evaluate.py` — benchmark evaluation with metrics and markdown report output
- `Makefile` — local command runner for install, train, and evaluation
- `pyproject.toml` — uv-managed project dependencies
- `Dockerfile` — container runtime for deployment
- `artifacts/` — generated model and benchmark artifacts
- `.github/workflows/ci.yml` — CI workflow for policy training and benchmark validation

## Quick start

### 1. Install dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Or, using the provided Make target:

```bash
make install
```

### 2. Train the policy

```bash
python train.py --timesteps 50000 --model-path artifacts/models/ppo_control_model.zip --seed 42
```

Or:

```bash
make train
```

This saves the model artifact in `artifacts/models/`.

### 3. Evaluate policy safety and performance

```bash
python evaluate.py --episodes 100 --max-steps 500 --model-path artifacts/models/ppo_control_model.zip --benchmark-dir artifacts/benchmarks --seed 42
```

Or:

```bash
make evaluate
```

This writes:
- `artifacts/benchmarks/metrics.json`
- `artifacts/benchmarks/benchmark_report.md`

### 4. Run the API server

```bash
uvicorn xrl_ops.inference_api:app --reload --host 0.0.0.0 --port 8000
```

## API usage

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "temperature": 72.5,
      "pressure": 120.0,
      "flow_rate": 310.0,
      "vibration": 1.8,
      "power_draw": 540.0,
      "setpoint": 75.0,
      "torque": 120.0,
      "coolant_temp": 65.0,
      "timestamp": "2026-10-22T12:00:00Z"
    }
  }'
```

### Explain

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "temperature": 72.5,
      "pressure": 120.0,
      "flow_rate": 310.0,
      "vibration": 1.8,
      "power_draw": 540.0,
      "setpoint": 75.0,
      "torque": 120.0,
      "coolant_temp": 65.0,
      "timestamp": "2026-10-22T12:00:00Z"
    }
  }'
```

The explain endpoint returns SHAP-based attribution values for the operating state, allowing reviewers to inspect which variables contributed most significantly to the chosen control action.

## Benchmarking and safety review

The evaluation process measures:
- mean reward
- reward standard deviation
- min/max reward
- total constraint violations
- temperature and pressure violation counts
- safety pass/fail classification based on configured tolerances

The benchmark report is generated as a Markdown document to support human review and auditability.

## DVC artifact versioning

To initialize DVC and track generated model/benchmark artifacts:

```bash
dvc init
dvc add artifacts/models artifacts/benchmarks
```

This version-controls the trained policy and evaluation outputs so that experiments and model releases remain reproducible.

## CI

The repository includes a GitHub Actions workflow that triggers on pushes to `main` and runs:
- dependency installation
- PPO training
- benchmark evaluation
- artifact upload for `metrics.json` and `benchmark_report.md`

## Engineering philosophy

This backend is built with a strict empirical methodology:
- no unverified assertions
- all state inputs are validated by Pydantic
- all policy outputs are benchmarked under fixed seeds
- all safety-critical metrics are logged and auditable
- model artifacts and benchmark reports are tracked for reproducibility
