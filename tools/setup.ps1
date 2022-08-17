# This file is adapted from https://git.corp.adobe.com/3di/python-scaffold

#Activate conda env
conda config --set ssl_verify no
# ask conda to show folder name instead of tag or abs path for envs in non default location.
conda config --set env_prompt '({name}) '

if( -not $? ) { exit -1 }
conda env update $ENV:CONDA_DEBUG_FLAG --prefix .venv\ --file "tools\conda.yaml"
if( -not $? ) { exit -1 }
conda activate .venv\
if( -not $? ) { exit -1 }

# Install package in dev mode
pip install -U -e .
if( -not $? ) { exit -1 }

# store the path to the conda installed python
# this will always be there even if you activate the env with conda and not use the script.
conda env config vars set MULTIMETASSD_PYTHON_PATH=$(python -c "import sys; print(sys.executable)")
if( -not $? ) { exit -1 }

# The first time the env var is set, it won't be visible until the env
# is refreshed
conda activate .venv\
if( -not $? ) { exit -1 }
