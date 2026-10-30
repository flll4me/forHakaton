PYTHON ?= python
IN ?= data/raw/handout/tickets_valid.jsonl
OUT ?= pred.jsonl
PORT ?= 8000

.PHONY: install data train predict test evaluate run dashboard analyze demo demo-check safety-audit

install:
	$(PYTHON) -m pip install -r requirements.txt

# распаковать архив кейса (положите grand-support-copilot-handout-*.tgz в корень)
data:
	mkdir -p data/raw && tar -xzf grand-support-copilot-handout-*.tgz -C data/raw --exclude '._*' --exclude '*/._*'

analyze:
	$(PYTHON) scripts/analyze_data.py

# weak supervision + обучение и сравнение классификаторов (A/B/C), лучший → models/classifier.joblib
train:
	$(PYTHON) scripts/prepare_dataset.py
	$(PYTHON) scripts/train_classifier.py --approaches A,B,C

predict:
	$(PYTHON) scripts/predict.py --in $(IN) --out $(OUT)

test:
	$(PYTHON) -m pytest -q tests

evaluate:
	$(PYTHON) scripts/evaluate.py

run:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

safety-audit:
	$(PYTHON) scripts/safety_audit.py

demo-check:
	$(PYTHON) scripts/demo_check.py

# проверка сценария, затем API (:8000) в фоне и админка (:8501) в демо-режиме; Ctrl+C останавливает оба
demo: demo-check
	export DEMO=1 FEEDBACK_DB=data/processed/feedback_demo.sqlite OPENBLAS_NUM_THREADS=1; \
	$(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port $(PORT) & API=$$!; \
	trap "kill $$API" EXIT; \
	$(PYTHON) -m streamlit run dashboard/app.py --server.headless true
