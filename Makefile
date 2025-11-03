.PHONY: help install update reinstall clean test format lint

help:
	@echo "DocRAG Makefile Commands"
	@echo ""
	@echo "  make install     - Install DocRAG with pipx (editable mode)"
	@echo "  make update      - Pull latest changes and reinstall if needed"
	@echo "  make reinstall   - Force reinstall DocRAG"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make test        - Run tests"
	@echo "  make format      - Format code with black"
	@echo "  make lint        - Lint code with ruff"

install:
	@echo "Installing DocRAG with pipx..."
	pipx install -e .
	@echo "✓ Installation complete"

update:
	@echo "Updating DocRAG..."
	./update.sh

reinstall:
	@echo "Reinstalling DocRAG..."
	@if command -v pipx >/dev/null 2>&1; then \
		pipx uninstall docrag 2>/dev/null || true; \
		pipx install -e .; \
	else \
		pip install -e . --force-reinstall; \
	fi
	@echo "✓ Reinstallation complete"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Clean complete"

test:
	@echo "Running tests..."
	pytest tests/ -v

format:
	@echo "Formatting code..."
	black docrag/
	@echo "✓ Format complete"

lint:
	@echo "Linting code..."
	ruff check docrag/
	@echo "✓ Lint complete"
