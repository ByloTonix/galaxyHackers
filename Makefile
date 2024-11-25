FORMAT_DIRS=galaxy
LINTER_DIRS=galaxy
TOML_FILES=poetry.lock pyproject.toml
POETRY_EXEC=poetry
PYTHON_EXEC=$(POETRY_EXEC) run python


.PHONY: help
help:  ## Показать это сообщение
	@grep -E -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'



.PHONY: install
install:  ## Установить зависимости
	poetry install --only main

.PHONY: pretty
pretty: isort autoflake black ## Отформатировать код

.PHONY: lint
lint: isort-check autoflake-check black-check pylint ## Проверить код на корректность

.PHONY: black
black:
	$(PYTHON_EXEC) -m black -t py310 $(FORMAT_DIRS)

.PHONY: black-check
black-check:
	$(PYTHON_EXEC) -m black -t py310 $(FORMAT_DIRS) --check

.PHONY: isort
isort:
	$(PYTHON_EXEC) -m isort $(FORMAT_DIRS)

.PHONY: isort-check
isort-check:
	$(PYTHON_EXEC) -m isort $(FORMAT_DIRS) --check

.PHONY: autoflake
autoflake:
	$(PYTHON_EXEC) -m autoflake -i -r --verbose --ignore-init-module-imports --remove-all-unused-imports --expand-star-imports $(FORMAT_DIRS)

.PHONY: autoflake-check
autoflake-check:
	$(PYTHON_EXEC) -m autoflake -i -r --verbose --ignore-init-module-imports --remove-all-unused-imports --expand-star-imports $(FORMAT_DIRS) --check


.PHONY: flake8
flake8:
	$(POETRY_EXEC) run flake8 --jobs 4 --statistics --show-source $(LINTER_DIRS)

.PHONY: pylint
pylint:
	$(POETRY_EXEC) run pylint --jobs 4 --rcfile=setup.cfg --extension-pkg-whitelist='pydantic' $(LINTER_DIRS)


.PHONY: prune_branches
prune_branches:
	git branch | grep -v "dev" | grep -v "master" | xargs git branch -D


.PHONY: run
run:
	python -m galaxy.main


.PHONY: build_docs
build_docs:
	poetry run make -C docs html
	touch docs/_build/html/.nojekyll
	mkdir -p docs/html
	cp -r docs/_build/html/* docs/
