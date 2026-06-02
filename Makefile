.PHONY: install run test lint
install:
	python -m pip install -r requirements.txt
run:
	python main.py --target-description "a generic demo assistant"
test:
	pytest -q
lint:
	ruff check .
