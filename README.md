<div align="center">

<h1>PERSE: Personalized 3D Generative Avatars from a Single Portrait</h1>

<div>
    <a href='https://hyunsoocha.github.io/' target='_blank'>Hyunsoo Cha</a>&emsp;
    <a href='https://blog.sulwon.com/' target='_blank'>Inhee Lee</a>&emsp;
    <a href='https://jhugestar.github.io/' target='_blank'>Hanbyul Joo</a>
</div>

<div align='Center'>
   Seoul National University
</div>

<div align='Center'>
<i><strong><a href='https://arxiv.org/abs/2412.21206' target='_blank'>CVPR 2025</a></strong></i>
</div>

<div align='Center'>
    <a href='https://hyunsoocha.github.io/perse/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/abs/2412.21206'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://youtu.be/zX881Zx03o4?si=t7j_CJMzeE4g9jYJ'><img src='https://badges.aleen42.com/src/youtube.svg'></a>
    <a href="https://huggingface.co/HyunsooCha/PERSE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow"></a>
    <a href="https://huggingface.co/datasets/HyunsooCha/PERSE_Dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-blue"></a>
</div>

https://github.com/user-attachments/assets/9e03af7e-b8a9-4e5e-bdf6-84d0a1ac7dd8

<h4 class="subtitle has-text-centered" style="margin-top: 5px">
TL;DR: Given a reference portrait image input, PERSE generates an
animatable 3D personalized avatar with disentangled and editable
control over various facial attributes.
</h4>

</div>

## 📣 News
- **`2025.07.28`**:  We release code for inference and training.
- **`2025.02.27`**:  🎉 PERSE got accepted into CVPR 2025!
- **`2024.12.31`**:  👏 [AK](https://x.com/_akhaliq/status/1874090429077217506) and [MrNeRF](https://x.com/janusch_patas/status/1874005568278716561) introduced our work! Thank you for sharing our paper.
- **`2024.12.30`**:  🎉 We release [paper](https://arxiv.org/abs/2412.21206) and [project page](https://hyunsoocha.github.io/perse/).

## 🌟 TODOs
- [x] Released the paper and project page on **`2024.12.30`**.
- [x] Release training, inference code.
- [ ] Release pretrained weights of avatar model.
- [ ] Release sample of datasets.

## 📦 Computational Requirements
Our code supports preprocessing, training, and inference on NVIDIA GPUs with at least 48 GB of memory, such as A6000.

## 🚀 Install
Our code has been tested with CUDA 11.8 on Ubuntu 22.04.
Functionality in other environments is not guaranteed.
We recommend installing and running it via Docker.

(1) Clone our repository
```
git clone https://github.com/snuvclab/perse.git
```
(2) Update WORK_DIR

Before running the installation script, please update the WORK_DIR in 
```
scripts/common.sh
```
. WORK_DIR is the location where the repository was cloned.

(3) Run install script. We use conda.
```
# in project root.
mkdir -p submodules data data/datasets data/experiments
cd scripts
bash 1_install_conda_portrait_editor.sh
bash 2_install_conda_portrait_champ.sh
bash 3_install_conda_guidance.sh
bash 4_install_conda_perse_avatar.sh
```
**NOTE:** In our preprocessing code and training of PERSE, we rely on the face segmentation model from Sapiens (Khirodkar et al.). However, unfortunately, we are unable to provide the face segmentation checkpoint. We recommend that you refer to [How to create a face segmentation checkpoint from Sapiens](https://github.com/facebookresearch/sapiens/blob/main/docs/finetune/SEG_README.md) to finetune the face segmentation model and use that.

## 🛠️ Preprocessing
```
# in project root.
cd scripts/run
bash 1_generate_prompt.sh
bash 2_edit_portrait.sh
bash 3_normal_portrait.sh
bash 4_generate_guidance.sh
bash 5_preprocess_guidance.sh
bash 6_portrait_champ.sh
```

## 📚 Train
```
# in project root.
cd scripts/run
bash 9_train_perse_avatar_model.sh
```

## 💡 Inference
```
# in project root.
cd scripts/run
bash 12_test_perse_avatar_model.sh
```

## 📢 License
Codes are available only for non-commercial research purposes.

## 📬 Contact
If you have any questions about the code, suggestions for improvement, or any other inquiries, please do not hesitate to contact the first author, Hyunsoo Cha (729steven@gmail.com or 243stephen@snu.ac.kr).

## ✏️ Citation
If you find our work useful, please cite our paper:

```
@InProceedings{Cha_2025_CVPR,
    author    = {Cha, Hyunsoo and Lee, Inhee and Joo, Hanbyul},
    title     = {PERSE: Personalized 3D Generative Avatars from A Single Portrait},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {15953-15962}
}
```