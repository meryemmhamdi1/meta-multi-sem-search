#!/bin/bash

# This file is adapted from https://git.corp.adobe.com/3di/python-scaffold

# ask conda to show folder name instead of tag or abs path for envs in non default location.
conda config --set env_prompt '({name}) '

conda env update -q $CONDA_DEBUG_FLAG --prefix .venv/ --file "tools/conda.yaml" || exit -1
conda activate .venv/ || exit -1
pip install -e .

# store the path to the conda installed python
# this will always be there even if you activate the env with conda and not use the script.
conda env config vars set MULTIMETASSD_PYTHON_PATH=$(python -c 'import sys; print(sys.executable)')
conda deactivate

# The first time the env var is set, it won't be visible until the env is refreshed
conda activate .venv/ || exit -1
