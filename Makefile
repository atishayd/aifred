.PHONY: test clean build run

test:
	python -m pytest tests/ -v --cov=src

clean:
	rm -rf build dist logs *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

build:
	python src/build.py

run:
	python run.py 