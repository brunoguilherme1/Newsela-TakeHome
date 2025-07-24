# Makefile

PYTHON := python3.11
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate

.PHONY: setup train docker predict

setup:
	@echo "📦 Creating virtual environment and installing dependencies..."
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt
	@echo "✅ Environment setup complete"

train:
	@echo "🚀 Running training pipeline..."
	$(ACTIVATE) && $(PYTHON) train.py
	@echo "✅ Model training complete"

docker:
	@echo "🐳 Building Docker image..."
	docker build -t topic-api .
	@echo "✅ Docker image 'topic-api' built successfully"

predict:
	@echo "🔍 Running evaluation on content_20.csv..."
	$(ACTIVATE) && $(PYTHON) predict.py
	@echo "✅ Evaluation complete"
