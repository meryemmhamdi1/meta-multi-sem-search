# This file is adapted from https://git.corp.adobe.com/3di/python-scaffold
if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Usage 'bash install_conda.sh installation-directory'"
    exit
fi

if command -v conda &> /dev/null; then
    echo "WARNING: conda is already installed and available on command line."
fi

OUTPUT_DIRECTORY="${1}"
echo "OUTPUT_DIRECTORY is ${OUTPUT_DIRECTORY}"

mkdir -p ${OUTPUT_DIRECTORY}

echo "Getting conda"
SCRIPT="Miniforge3-$(uname)-$(uname -m).sh"
LINK="https://github.com/conda-forge/miniforge/releases/latest/download/${SCRIPT}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    curl -L -s -S -O $LINK
else
    wget -nv $LINK
fi

echo "Installing conda locally"
bash "$SCRIPT" -bfp "${OUTPUT_DIRECTORY}" || (echo "Failed to run installer script";  exit -1)
rm -f "$SCRIPT"

echo "conda installed successfuly"

echo "To activate conda run \"source ${OUTPUT_DIRECTORY}/etc/profile.d/conda.sh\""