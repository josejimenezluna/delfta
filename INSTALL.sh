CONDA_DIR=$(conda info --base)

source ${CONDA_DIR}/etc/profile.d/conda.sh

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ENVFILE="environment.yml"
elif [[ "$OSTYPE" == "darwin"* ]]; then
        ENVFILE="environment_osx.yml"
else
    echo "OS type not supported"
fi

conda env create -f ${ENVFILE}
