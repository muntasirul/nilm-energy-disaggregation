.PHONY: help setup data preprocess train evaluate serve dashboard test lint

help:
	@echo "make setup       - create venv + install deps"
	@echo "make data        - download REDD low_freq (house_1 by default; ALL=1 for all houses)"
	@echo "make preprocess  - parse raw .dat -> parquet"
	@echo "make train       - train Seq2Point (uses config/config.yaml)"
	@echo "make evaluate    - evaluate latest checkpoint on test split"
	@echo "make serve       - run FastAPI on :8000"
	@echo "make dashboard   - run Streamlit dashboard on :8501"
	@echo "make test        - run pytest"
	@echo "make lint        - run ruff"

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:
ifdef ALL
	python3 -m src.data.download --all
else
	python3 -m src.data.download --house house_1
endif

preprocess:
	python3 -m src.data.preprocess

train:
	python3 -m src.models.train

evaluate:
	python3 -m src.models.evaluate

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/dashboard/app.py

test:
	pytest -q

lint:
	ruff check src tests
