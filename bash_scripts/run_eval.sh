#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate openvla_2

nohup python /home/k23preus/UE_rechrche/OpenVLA/openvla_ue_recherche/experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  > output_eval.log 2>&1 &

