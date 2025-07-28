#!/bin/bash
#############################################
# Customize these variables as needed
WORK_DIR=$HOME/GitHub/perse-dev
#############################################
# Predefined Paths
SUBMODULE_PATH=$WORK_DIR/submodules
DATASET_PATH=$WORK_DIR/data/datasets/$DATASET_NAME
PORTRAIT_EDITOR_PATH=$WORK_DIR/code/dataset-generator/portrait-editor
GUIDANCE_PATH=$WORK_DIR/code/dataset-generator/guidance
PORTRAIT_CHAMP_PATH=$WORK_DIR/code/dataset-generator/portrait-champ
AVATAR_MODEL_PATH=$WORK_DIR/code/avatar-model
SAPIENS_CHECKPOINT_PATH=$SUBMODULE_PATH/sapiens/sapiens_host
#############################################
# Utility Functions
print_green() {
    echo -e "\033[0;32m$1\033[0m"
}
pip_install_editable() {
    print_green "Installing $1..."
    cd "$1" || exit
    pip install . -v
    cd - || exit
    print_green "Finished installing $1."
}
clone_if_not_exists() {
    local repo_url="$1"
    local dir_name=$(basename "$repo_url" .git)

    if [[ ! -d "$dir_name" ]]; then
        echo "Directory '$dir_name' not found. Cloning repository..."
        git clone "$repo_url" --recursive
    else
        echo "Directory '$dir_name' already exists. Skipping clone."
    fi
}
setup_conda_env() {
    source "$HOME/conda/etc/profile.d/conda.sh"
    conda activate
    echo "Removing existing conda environment: $ENV_NAME"
    # conda deactivate || true
    conda remove -y -n "$ENV_NAME" --all

    echo "Creating new conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
    set -e
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

    conda activate "$ENV_NAME"
}
activate_env() {
    source $HOME/conda/etc/profile.d/conda.sh
    conda activate
    echo "Activating conda environment: $ENV_NAME"
    conda activate "$ENV_NAME"
    export CUDA_VISIBLE_DEVICES=$DEVICE
}
set_torch_cuda_arch() {
    CUDA_ARCHITECTURES="70;75;80;86"
    CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
    TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
    MAX_JOBS=1
    export MAX_JOBS
    FORCE_CUDA="1"
    export FORCE_CUDA

    CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
    echo "Detected CUDA version: $CUDA_VERSION"

    if [[ ${CUDA_VERSION} == 9.0* ]]; then
        export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;7.0+PTX"
    elif [[ ${CUDA_VERSION} == 9.2* ]]; then
        export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0+PTX"
    elif [[ ${CUDA_VERSION} == 10.* ]]; then
        export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5+PTX"
    elif [[ ${CUDA_VERSION} == 11.0* ]]; then
        export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0+PTX"
    elif [[ ${CUDA_VERSION} == 11.* ]]; then
        export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    else
        echo "❌ Unsupported CUDA version: $CUDA_VERSION"
        return 1
    fi

    echo "✅ TORCH_CUDA_ARCH_LIST set to: $TORCH_CUDA_ARCH_LIST"
}
