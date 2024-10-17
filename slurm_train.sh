#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --partition=cbmm

# python mae_pretrain.py --use_ae_decoder --total_epoch 200 --warmup_epoch 20 
python mae_pretrain.py --total_epoch 100 --warmup_epoch 10 