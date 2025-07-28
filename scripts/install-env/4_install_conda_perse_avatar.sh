#!/bin/bash
#############################################
ENV_NAME=perse-avatar
PYTHON_VERSION=3.9
#############################################

source ../common.sh
setup_conda_env
set_torch_cuda_arch

print_green "Installing PyTorch"
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

print_green "Installing other dependencies"
pip install -r requirements_perse_avatar.txt

print_green "Installing pytorch3d"
cd $SUBMODULE_PATH
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6

print_green "Installing gaussian-splatting"
cd $SUBMODULE_PATH
clone_if_not_exists https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting
git checkout 472689c0dc70417448fb451bf529ae532d32c095
cd submodules/diff-gaussian-rasterization
git checkout 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
pip install .

print_green "Installing CLIP"
pip install git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1

print_green "Installing DiffMorpher..."
cd $SUBMODULE_PATH
clone_if_not_exists https://github.com/HyunsooCha/diffmorpher-for-perse.git

print_green "Installing sapiens"
pip install git+https://github.com/bcmi/libcom.git@9d3180e4e8b2532a1e39f953590c748e5c7fa434
cd $SUBMODULE_PATH
clone_if_not_exists https://github.com/facebookresearch/sapiens.git
cd sapiens
git checkout 7bd3594ccc98a64ffa40ce3d577eff6afa6efba7
pip_install_editable "engine"
pip_install_editable "cv"
pip install -r "cv/requirements/optional.txt"  # Install optional requirements
pip_install_editable "pretrain"
pip_install_editable "pose"
pip_install_editable "det"
pip_install_editable "seg"
pip install -U openmim
mim install mmcv
print_green "Installation done!"