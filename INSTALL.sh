CONDA_DIR=$(conda info --base)

source ${CONDA_DIR}/etc/profile.d/conda.sh
conda env create -f environment.yml
