.PHONY: all data features train evaluate api dashboard clean

PYTHON = python

all: data features train evaluate

data:
	@echo "=== Generating synthetic sensor data ==="
	$(PYTHON) -m src.data.generate_data

features:
	@echo "=== Engineering features ==="
	$(PYTHON) -m src.features.feature_engineering

train:
	@echo "=== Training models ==="
	$(PYTHON) -m src.models.train

evaluate:
	@echo "=== Evaluating models ==="
	$(PYTHON) -m src.models.evaluate

api:
	@echo "=== Starting FastAPI server ==="
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	@echo "=== Starting Streamlit dashboard ==="
	streamlit run src/dashboard/app.py --server.port 8501

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	rm -rf data/raw/*.csv data/processed/*.csv
	rm -rf models/*.joblib
	rm -rf results/figures/*.png results/metrics/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
