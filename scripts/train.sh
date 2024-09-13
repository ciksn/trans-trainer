#!/bin/bash
export MODEL_LOAD='/home/zeyu/work/deep_learning/functional_files/trans_trainer/checkpoints'
export CONFIG_LOAD='/home/zeyu/work/deep_learning/functional_files/trans_trainer/checkpoints'
export TOKENIZER_LOAD='/home/zeyu/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
# export TOKENIZER_LOAD='/home/zeyu/mnt/drive0/dataset/driving/BDD-X/BDD-X-Dataset/tokenizer'

export CACHE_DIR='/home/zeyu/.cache/huggingface/hub/'

export DATASET_NAME="bddx_dataset"
export CAPTION_FILE_PATH="/home/zeyu/mnt/drive0/dataset/driving/BDD-X/BDD-X-Dataset/bddx.json"
export VIDEO_2D_PATH="/home/zeyu/work/deep_learning/extracted_dataset/bddx/CLIP-ViT_L14"
export VIDEO_3D_PATH="/home/zeyu/work/deep_learning/extracted_dataset/bddx/S3D"
export VIDEO_OBJECT_PATH="/home/zeyu/work/deep_learning/extracted_dataset/bddx/Fasterrcnn"

python ../train.py \
    --load_from_config False \
    --load_from_pretrained False \
    --model_name_or_path $MODEL_LOAD \
    --config_name_or_path $CONFIG_LOAD \
    --need_tokenizer True \
    --tokenizer_name_or_path $TOKENIZER_LOAD \
    --cache_dir $CACHE_DIR \
    --version v1 \
    --dataset_name $DATASET_NAME \
    --caption_file_path $CAPTION_FILE_PATH \
    --video_2d_path $VIDEO_2D_PATH \
    --video_3d_path $VIDEO_3D_PATH \
    --video_object_path $VIDEO_OBJECT_PATH \
    --caption_seq_len 40 \
    --video_seq_len 32 \
    --status "pretrain" \
    --optim "adamw_torch" \
    --load_best_model_at_end True \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-08 \
    --bf16 True \
    --dataloader_drop_last False \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --evaluation_strategy "epoch" \
    --eval_steps 1 \
    --gradient_checkpointing False \
    --group_by_length False \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --logging_strategy "steps" \
    --lr_scheduler_type "cosine" \
    --num_train_epochs 15 \
    --output_dir ../checkpoints/checkpoint \
    --report_to tensorboard \
    --resume_from_checkpoint None \
    --save_strategy "epoch" \
    --save_steps 30 \
    --save_total_limit 1 \
    --metric_for_best_model "action/CIDEr" \
    --seed 42 \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 16 \
    --weight_decay 0 \
    --warmup_ratio 0.03

python ../caption_test.py \
    --checkpoint "/home/zeyu/work/deep_learning/functional_files/trans_trainer/checkpoints/checkpoint"