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

# python mae_pretrain.py --use_ae_decoder --total_epoch 200 --warmup_epoch 20 --log_dir 'mae_encoder_decoder' # job_output_39038136
# python mae_pretrain.py --total_epoch 200 --warmup_epoch 20 --train_ae_decoder --ae_decoder_epochs 200 --log_dir mae_encoder_ae_decoder # job_output_39038121
# python mae_pretrain.py --train_full_ae --total_epoch 200 --warmup_epoch 20 --log_dir 'ae_encoder_decoder' # job_output_39038119
# python mae_pretrain.py --train_mae_encoder_ae_decoder --total_epoch 200 --warmup_epoch 20 --log_dir 'mae_encoder_ae_decoder_end2end'
# python mae_pretrain.py --train_mae_encoder_ae_decoder --total_epoch 200 --warmup_epoch 20 --mask_ratio 0.1 --log_dir 'mae_encoder_ae_decoder_end2end_mask_ratio_0.1' #job_output_39045390.txt
# python mae_pretrain.py --train_mae_encoder_ae_decoder --total_epoch 200 --warmup_epoch 20 --mask_ratio 0 --log_dir 'mae_encoder_ae_decoder_end2end_mask_ratio_0'