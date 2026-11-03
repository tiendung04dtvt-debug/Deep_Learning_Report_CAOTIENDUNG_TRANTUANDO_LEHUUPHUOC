# ============================================================
# TrafficAI (ai-traffic-intelligence) — Makefile
# Common management and reproducibility shortcuts
# ============================================================

.PHONY: install run pipeline clean

# --- Setup ---
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-ai.txt

# --- Execution ---
run:
	python web_app.py

pipeline:
	python main.py

# --- Cleanup ---
clean:
	@if exist __pycache__ (rmdir /s /q __pycache__)
	@if exist src\__pycache__ (rmdir /s /q src\__pycache__)
	@if exist static\crops (rmdir /s /q static\crops)
	@echo Cleanup complete.
