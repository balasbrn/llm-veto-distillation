.PHONY: install test demo

install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev,import]"

test:
	.venv/bin/python -m pytest -q

demo:
	.venv/bin/vd demo --out results/synthetic
