#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:6
#SBATCH --cpus-per-task=6
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --partition=cbmm

python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}