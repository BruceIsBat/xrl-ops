# XRL-Ops: Explainable Reinforcement Learning for Control Systems

**XRL-Ops** is a production-grade MLOps backend engineered to deploy, evaluate, and explain sequential decision optimization agents in physical, industrial, and robotic environments. 

Academic reinforcement learning often struggles to transition from simulated environments to real-world industrial deployment due to the "black box" nature of deep RL and the lack of robust serving infrastructure. XRL-Ops solves this by wrapping continuous control policies (via Stable-Baselines3) in a high-concurrency API layer and computing real-time state-action attributions (via SHAP) to ensure safe, auditable decision-making.

## 🏗️ Core Architecture

- **High-Throughput State Ingestion**: Utilizes **Polars** for multi-core processing of time-series sensor data and state variables, strictly avoiding memory bottlenecks.
- **Explainable Inference (The "X")**: An asynchronous **FastAPI** layer exposes `/predict` (policy action) and `/explain` (SHAP-based feature importance), making the agent's decision matrix fully transparent to human operators.
- **Empirical Benchmarking**: Integrated `tracker.py` and `evaluate.py` modules record state-action-reward trajectories and track constraint violations to ensure physical safety thresholds are never breached.
- **Reproducible ML Lifecycle**: Data and model artifacts are versioned via **DVC**, with the entire Python environment pinned immutably using `uv` and containerized via **Docker**.

## 🚀 Quick Start

**1. Environment Setup**
Install `uv` for reproducible dependency management:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -e .
```

**2. Train & Evaluate the Agent**
Train the PPO agent on the continuous control environment and evaluate its constraint compliance:
```bash
python scripts/train.py --timesteps 100000
python scripts/evaluate.py --episodes 100
```
*(Note: Model weights are automatically versioned into `artifacts/models/` via DVC).*

**3. Launch the Inference Server**
```bash
uvicorn xrl_ops.inference_api:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Usage

**Get Optimal Control Action:**
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

**Audit Decision via SHAP Attribution:**
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{ "state": { ... } }'
```
*Returns the optimal action alongside the SHAP values indicating which specific sensor states drove the policy's output.*

## 🔬 Engineering Philosophy
Built under a strict empirical methodology: no unverified assertions. All state inputs are strongly typed via Pydantic, and all policy updates require pre-registered benchmark traces to ensure constraint compliance before deployment.
