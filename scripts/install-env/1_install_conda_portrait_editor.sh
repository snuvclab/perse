#!/bin/bash
#############################################
ENV_NAME=portrait-editor
PYTHON_VERSION=3.9
#############################################

source ../common.sh
setup_conda_env
set_torch_cuda_arch

pip install -r requirements_portrait_editor.txt

mkdir -p $SUBMODULE_PATH

print_green "Installing ControlNetPlus..."
cd $SUBMODULE_PATH
clone_if_not_exists https://github.com/HyunsooCha/controlnetplus-for-perse.git
cd controlnetplus-for-perse
pip install -r requirements.txt

print_green "Installing ControlNet FLUX Inpainting"
cd $SUBMODULE_PATH
clone_if_not_exists https://github.com/alimama-creative/FLUX-Controlnet-Inpainting.git
cd FLUX-Controlnet-Inpainting
git checkout 45c744b78ae7658c8940e51b074314c78702ea86

print_green "Installing controlnet_aux..."
pip install git+https://github.com/huggingface/controlnet_aux.git@3aec3bc8e3d6197c585f6a6e001ab8c5462a30ea

print_green "Installing diffusers v0.30.2..."
pip install git+https://github.com/huggingface/diffusers.git@v0.30.2 accelerate --upgrade

print_green "Installing pytorch 2.1.0"
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

print_green "Installing sapiens..."
cd $SUBMODULE_PATH
clone_if_not_exists https://github.com/facebookresearch/sapiens.git
cd sapiens
git checkout 7bd3594ccc98a64ffa40ce3d577eff6afa6efba7
pip install git+https://github.com/bcmi/libcom.git@9d3180e4e8b2532a1e39f953590c748e5c7fa434

cd $SUBMODULE_PATH/sapiens
pip_install_editable "engine"
pip_install_editable "cv"
pip install -r "cv/requirements/optional.txt"  # Install optional requirements
pip_install_editable "pretrain"
pip_install_editable "pose"
pip_install_editable "det"
pip_install_editable "seg"
print_green "Installation done!"