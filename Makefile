PYTHON ?= python
UV ?= uv
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python

.PHONY: install train evaluate benchmark dvc-init dvc-add

install:
	$(UV) venv --python 3.12 $(VENV)
	$(UV) pip install --python $(PYTHON_BIN) -e ".[dev]"

train:
	$(UV) run --python $(PYTHON_BIN) python train.py --timesteps 50000 --model-path artifacts/models/ppo_control_model.zip --seed 42

evaluate:
	$(UV) run --python $(PYTHON_BIN) python evaluate.py \
		--episodes 100 \
		--max-steps 500 \
		--model-path artifacts/models/ppo_control_model.zip \
		--benchmark-dir artifacts/benchmarks \
		--seed 42 \
		--temp-violation-tolerance 10 \
		--pressure-violation-tolerance 10

benchmark: evaluate

dvc-init:
	dvc init

dvc-add:
	dvc add artifacts/models artifacts/benchmarks
