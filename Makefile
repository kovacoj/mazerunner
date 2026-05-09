SHELL := /bin/sh

.PHONY: clean clean-python clean-latex clean-notebooks

clean: clean-python clean-latex clean-notebooks

clean-python:
	rm -rf build dist .eggs htmlcov .pytest_cache .mypy_cache .ruff_cache
	find . -type d \( -name __pycache__ -o -name "*.egg-info" \) -prune -exec rm -rf {} +
	rm -f .coverage

clean-latex:
	find report -type f \( \
		-name "*.aux" -o \
		-name "*.bbl" -o \
		-name "*.bcf" -o \
		-name "*.blg" -o \
		-name "*.fdb_latexmk" -o \
		-name "*.fls" -o \
		-name "*.lof" -o \
		-name "*.log" -o \
		-name "*.lot" -o \
		-name "*.out" -o \
		-name "*.run.xml" -o \
		-name "*.synctex.gz" -o \
		-name "*.toc" \
	\) -delete

clean-notebooks:
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
