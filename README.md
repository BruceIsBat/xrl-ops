# XRL-Ops

XRL-Ops is a production-ready backend for explainable reinforcement learning in industrial control and robotics. It focuses on safe deployment, auditability, and high-throughput inference for process-level control policies.

## Features

- Polars-first feature engineering for raw control-state streams
- Strict Pydantic validation for sensor and actuator state payloads
- Async FastAPI inference endpoints for predict and explain
- Trajectory tracking for benchmarking, failure analysis, and safety auditing
- Reproducible dependency management with uv
- Docker-ready deployment

## Project layout

- `xrl_ops/data_engine.py` — ingestion, validation, feature engineering, RL preparation
- `xrl_ops/inference_api.py` — FastAPI router with `/predict` and `/explain`
- `xrl_ops/tracker.py` — empirical logging and trajectory persistence
- `pyproject.toml` — uv project configuration
- `Dockerfile` — container build and runtime setup

## Quick start

1. Install uv if needed:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

3. Run the API:

   ```bash
   uvicorn xrl_ops.inference_api:app --reload --host 0.0.0.0 --port 8000
   ```

4. Send a request:

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
         "coolant_temp": 65.0
       }
     }'
   ```

## Example explanation payload

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
      "coolant_temp": 65.0
    }
  }'
```

## Production notes

- Replace the heuristic policy stub in `xrl_ops/inference_api.py` with a trained RL model once policy weights are available.
- Integrate SHAP or a policy-gradient attribution method for deployment-level explainability.
- Persist benchmark traces in a DVC-managed dataset or object-store for reproducible experiments.
