## Makefile to automate commands like ci/cd, installing dependencies, and creating a virtual environment

## Excecutes all commands in a single shell (allows us to activate a virtual env and install dependencies in it)
.ONESHELL:

.PHONY: venv test require install create

## Variables set at declaration time
PROJECT_NAME := diffusion
VENV_NAME := diffusion
REQUIREMENTS := requirements.txt

## Recursively expanded variables
python_source = ${PROJECT_NAME} scripts/  # Location of python files 
activate = . .venv/bin/activate
activate_mac = source .venv/bin/activate
activate_windows = source .venv/Scripts/activate

python = python3

venv_linux: ## Create virtual environment
	sudo apt install python3-venv
	sudo apt install python3-pip
	${python} -m venv .venv

venv_windows: ## Create virtual environment
	python -m venv .venv

venv_mac: ## Create virtual environment
	python3 -m venv .venv

test: ## Put pytests here
	. 
	pytest tests/

format: ## Reformats your code to a specific coding style
	${activate}
	black ${python_source}
	isort ${python_source}

check: ## Check to see if any linux formatting 
	${activate}
	black --check ${python_source}
	isort --check-only ${python_source}
	mypy ${python_source}
	pylint ${python_source}

require:
	pip install pip-tools
	pip-compile --output-file requirements.txt pyproject.toml 

install_reqs: ## Install for linux only; we also need to upgrade pip to support editable installation with only pyproject.toml file
	${activate}
	${python} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117  
	${python} -m pip install -r ${REQUIREMENTS}
	${python} -m pip install -e . --no-deps

install_reqs_mac: ## Install for linux only; we also need to upgrade pip to support editable installation with only pyproject.toml file
	${activate}
	${python} -m pip install --upgrade pip
	${python} -m pip install -e .

create_linux: venv_linux install_reqs ## Create virtual environment and install dependencies and the project itself

create_windows: venv_windows install_reqs_windows  

# Doesn't really work on mac, should probably install deps another way and just use make file for formatting
create_mac: venv_mac install_reqs_mac

activate_test: 
	${activate_mac}