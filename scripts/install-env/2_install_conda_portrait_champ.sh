#!/bin/bash
#############################################
ENV_NAME=portrait-champ
PYTHON_VERSION=3.10
#############################################

source ../common.sh
setup_conda_env
set_torch_cuda_arch

# pip3 install torch torchvision torchaudio
print_green "Installing PyTorch"
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_portrait_champ.txt
conda install -y -c conda-forge libstdcxx-ng

print_green "Installing detectron2..."
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

print_green "Installing xformers..."
pip install git+https://github.com/facebookresearch/xformers.git@v0.0.25.post1

cd $SUBMODULE_PATH
print_green "Installing LivePortrait..."
clone_if_not_exists https://github.com/KwaiVGI/LivePortrait.git
cd LivePortrait
git checkout 4d881b17597dfc6f0b1050e161423f101edc9342
pip install -r requirements.txt