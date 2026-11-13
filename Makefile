# Makefile for E-Commerce Product Recommendation Engine

.PHONY: help install data preprocess train api ui test clean all

help:
	@echo "Available commands:"
	@echo "  make install     : Install dependencies from requirements.txt"
	@echo "  make data        : Generate synthetic dataset (seed=42)"
	@echo "  make preprocess  : Run data preprocessing & chronological split"
	@echo "  make train       : Train all 4 models and run ranking evaluation"
	@echo "  make api         : Start FastAPI inference server (port 8000)"
	@echo "  make ui          : Start Streamlit interactive dashboard (port 8501)"
	@echo "  make test        : Run pytest test suite"
	@echo "  make all         : Generate data, preprocess, train, and test"
	@echo "  make clean       : Remove caches and temporary files"

install:
	pip install -r requirements.txt

data:
	python data/generate_data.py

preprocess:
	python -c "from src.preprocessing import run_preprocessing_pipeline; run_preprocessing_pipeline()"

train:
	python src/train.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v --tb=short

all: data train test

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
