#!/bin/bash
# Step 2: train the NoiseControl LoRA on CogVideoX-5B-I2V with flow-warped noise.
# Settings below are the ones used for the paper checkpoint (4 GPUs, 30k steps).
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the RealCam-Vid root (contains RealCam-Vid_train.npz)}"
MODEL="${MODEL:-THUDM/CogVideoX-5b-I2V}"
CACHE_ROOT="${CACHE_ROOT:-latent_cache}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-30000}"
OUTPUT_DIR="${OUTPUT_DIR:-./ckpts/noisecontrol-of-noise-lora}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export TORCHDYNAMO_VERBOSE=0
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

accelerate launch --config_file accelerate_config/experiment.yaml --gpu_ids "$GPU_IDS" \
  training/cogvideox/cogvideox_i2v_lora_of_noise.py \
  --pretrained_model_name_or_path "$MODEL" \
  --data_root "$DATA_ROOT" \
  --train_datafile RealCam-Vid_train.npz \
  --val_data_root "$DATA_ROOT" \
  --val_datafile RealCam-Vid_test.npz \
  --cache_root "$CACHE_ROOT" \
  --height_buckets 480 \
  --width_buckets 720 \
  --frame_buckets 49 \
  --max_num_frames 49 \
  --dataloader_num_workers 8 \
  --pin_memory \
  --num_validation_videos 1 \
  --validation_steps 500 \
  --seed 42 \
  --rank 2048 \
  --lora_alpha 2048 \
  --mixed_precision bf16 \
  --output_dir "$OUTPUT_DIR" \
  --train_batch_size 2 \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 3 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 400 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --noised_image_dropout 0.05 \
  --optimizer adamw \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 0.001 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb \
  --tracker_name noisecontrol-of-noise \
  --nccl_timeout 1800 \
  --resume_from_checkpoint latest
