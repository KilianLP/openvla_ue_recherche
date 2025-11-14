#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate openvla_2

nohup torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    /home/k23preus/UE_rechrche/OpenVLA/openvla_ue_recherche/vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir /home/k23preus/UE_rechrche/modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --lora_rank 32 \
    --batch_size 12 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --image_aug True \
    --save_steps 250 \
    --wandb_project OpenVLA_LoRA \
    --wandb_entity kilianleonhardpreuss \
    --use_lora True \
    > output.log 2>&1 &

