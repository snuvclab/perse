#!/bin/bash
#############################################
ENV_NAME=guidance
PYTHON_VERSION=3.8
#############################################

source ../common.sh
setup_conda_env
set_torch_cuda_arch

print_green "Install PyTorch 1.12.1 CUDA 11.6"
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

print_green "Install other dependencies"
pip install -r requirements_guidance.txt

print_green "Install PyTorch3D"
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6

print_green "Install DWPose related..."
cd $SUBMODULE_PATH
clone_if_not_exists https://github.com/IDEA-Research/DWPose.git
cd DWPose
git checkout 3dca5db79d9f9ffdd378753ddf6ec66535aace88

print_green "Install EMOCA..."
cd $SUBMODULE_PATH
clone_if_not_exists https://github.com/radekd91/emoca.git
cd emoca
git checkout c1cf7e450ec1dc23a43ff9b3b0f60097d4037c20
pip install . 

print_green "finished"