CONDA_DIR=$(conda info --base)
CUDA="cu102"
TORCH="1.7.0"

source $CONDA_DIR/etc/profile.d/conda.sh

conda env create -f environment.yml
conda activate delfta
conda install pytorch==${TORCH} torchvision torchaudio cudatoolkit=10.2 -c pytorch -y

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
