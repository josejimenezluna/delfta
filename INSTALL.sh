CONDA_DIR=$(conda info --base)
source $CONDA_DIR/etc/profile.d/conda.sh

conda env create -f environment.yml
conda activate delfta

CUDA="cu102"
TORCH="1.7.0"

if [[ ${CUDA} == 'cpuonly' ]]; then
  conda install pytorch==${TORCH} torchvision torchaudio cpuonly -c pytorch -y
elif [[ ${CUDA} == 'cu92' ]]; then
  conda install pytorch==${TORCH} torchvision torchaudio cudatoolkit=9.2 -c pytorch -y
elif [[ ${CUDA} == 'cu101' ]]; then
  conda install pytorch==${TORCH} torchvision torchaudio cudatoolkit=10.1 -c pytorch -y
elif [[ ${CUDA} == 'cu102' ]]; then
  conda install pytorch==${TORCH} torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
elif [[ ${CUDA} == 'cu110' ]]; then
  conda install pytorch==${TORCH} torchvision torchaudio cudatoolkit=11.0 -c pytorch -y
fi

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
