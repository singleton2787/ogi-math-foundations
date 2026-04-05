.PHONY: benchmark lint format help

help:
	@echo "ogi-math-foundations"
	@echo ""
	@echo "  make benchmark   Run primary validation (Theorem 4 + Lemma 4.2)"
	@echo "  make lipschitz   Run Lipschitz verification (Theorem 1)"
	@echo "  make lint        Run flake8"
	@echo "  make format      Run black"

benchmark:
	python experiments/ogi_benchmark.py

lipschitz:
	python experiments/lipschitz_test.py

lint:
	flake8 experiments/ instruments/

format:
	black experiments/ instruments/
