This repository is a **fork** of [OpenVLA](https://github.com/openvla/openvla), adapted for the *UE Recherche* project.

## Project Description

**Evaluating and Fine-Tuning Vision-Language-Action Models for Robotic Control in Novel Environments**

Vision-Language-Action (VLA) models, built upon pretrained Vision-Language Models (VLMs) and trained on large-scale robotics datasets, have demonstrated strong task, environment, and semantic generalization capabilities. However, it remains unclear how to efficiently adapt them to new environments, embodiments, and tasks during the post-training phase.  

In this work, we investigate the use of Parameter-Efficient Fine-Tuning (PEFT) techniques—specifically Low-Rank Adaptation (LoRA), originally developed for natural language processing—to adapt VLA models. We evaluate their effectiveness in terms of both task performance and fine-tuning computational cost, and compare them to naïve full fine-tuning as well as to an optimized fine-tuning technique specifically designed for VLAs in robotics.  

To this end, we will conduct experiments in the MuJoCo simulator using the LIBERO benchmark, a widely adopted framework for evaluating generalization and adaptation in robotic learning.

---

## Installation

The following instructions describe the setup used during the project.

```bash
# Clone and install the forked OpenVLA repository
git clone https://github.com/KilianLP/openvla_ue_recherche
cd openvla_ue_recherche
pip install -e .

# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  

# Install FlashAttention 2 for training
conda install -y -c nvidia "cuda-toolkit=12.1"
conda install -y -c conda-forge "gxx_linux-64=12" "cmake>=3.26" "ninja"

python -m pip install --no-build-isolation -v "flash-attn==2.6.3"
```

### Bibliography

You can find the beginning of the bibliography in the file **Bibliography.pdf**.


## Download data

To download the data, first go to the dataset directory:

```bash
cd modified_libero_rlds
```

Then run:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    local_dir="/home/k23preus/UE_rechrche/modified_libero_rlds",
    local_dir_use_symlinks=False,
    allow_patterns=["libero_object_no_noops/1.0.0/*"],
)
PY
```
---

## Launching training and evaluation in the background

You can use the bash scripts in the **bash_scripts/** directory to launch both the training and evaluation of a model in the background. These scripts use nohup, so the processes keep running even after you close the terminal.