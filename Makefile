.PHONY: setup setup-gpu test smoke smoke-gpu

setup:
	pip install -e ".[dev]"

setup-gpu:
	pip install -e ".[dev,gpu]"

test:
	python -m pytest tests/ -q

# tiny-model end-to-end (needs huggingface.co; runs on CPU or GPU)
smoke:
	bash scripts/smoke_e2e.sh

# same DAG with the real 1.5B model on 5 problems (first command on the GPU box)
smoke-gpu:
	bash scripts/smoke_e2e.sh   # then:
	python -m reasoncontrol.stages.prepare_data --config configs/base.yaml --datasets gsm8k
	python -m reasoncontrol.stages.generate --config configs/base.yaml \
		--set datasets='[gsm8k]' --set gen.n_rollouts=1 --set gen.max_think_tokens=2048
