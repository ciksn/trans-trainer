#!/bin/bash
export MODEL_LOAD='/home/zeyu/work/deep_learning/functional_files/trans_trainer/checkpoints'
export CONFIG_LOAD='/home/zeyu/work/deep_learning/functional_files/trans_trainer/checkpoints'
export TOKENIZER_LOAD='TinyLlama/TinyLlama_v1.1'

export VISUAL_BACKBONE='openai/clip-vit-large-patch14'
export LANGUAGE_MODEL='TinyLlama/TinyLlama_v1.1'

export CACHE_DIR='/home/zeyu/.cache/huggingface/hub/'

export DATASET_NAME="drama_dataset"

python ../train.py \
    --load_from_config False \
    --load_from_pretrained False \
    --model_name_or_path $MODEL_LOAD \
    --config_name_or_path $CONFIG_LOAD \
    --visual_backbone $VISUAL_BACKBONE \
    --language_model $LANGUAGE_MODEL \
    --need_tokenizer True \
    --tokenizer_from_pretrained False \
    --tokenizer_name_or_path $TOKENIZER_LOAD \
    --tokenizer_max_length 512 \
    --cache_dir $CACHE_DIR \
    --version v1 \
    --dataset_name $DATASET_NAME \
    --dataset_input_files '/home/zeyu/mnt/drive0/dataset/driving/drama/processed' \
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
    --num_train_epochs 10 \
    --output_dir ../checkpoints/checkpoint \
    --report_to tensorboard \
    --resume_from_checkpoint None \
    --save_strategy "epoch" \
    --save_steps 30 \
    --save_total_limit 1 \
    --metric_for_best_model "CIDEr" \
    --seed 42 \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 4 \
    --weight_decay 0 \
    --warmup_ratio 0.03

python ../caption_test.py \
    --checkpoint "/home/zeyu/work/deep_learning/functional_files/trans_trainer/checkpoints/checkpoint"