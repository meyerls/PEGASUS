#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh

conda update -n base -c defaults conda

conda create -y -n pegasus python=3.8
conda activate pegasus

conda install -c conda-forge cudatoolkit=11.6 -y
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit -y
conda install pip=22.3.1 -y
conda install pyopengl -y


pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

# install submodules
pip install submodules/gaussian-splatting-pegasus/submodules/depth-diff-gaussian-rasterization
pip install submodules/gaussian-splatting-pegasus/submodules/simple-knn

# install XMEM
pip install -r submodules/XMem/requirements_demo.txt
pip install -r submodules/XMem/requirements.txt

pip install Pillow==9.5.0

# Download XMEM weights
./submodules/XMem/scripts/download_models_demo.sh