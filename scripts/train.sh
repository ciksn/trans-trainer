#!/bin/bash
MODEL_LOAD=''
CONFIG_LOAD=''
TOKENIZER_LOAD=''
CACHE_DIR=''


deepspeed train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_LOAD \
    --config_name_or_path $CONFIG_LOAD \
    --need_tokenizer False \
    --tokenizer_name_or_path $TOKENIZER_LOAD \
    --cache_dir $CACHE_DIR \
    --version v1 \
    --data_path $DATA_FILE \
    --dataset_name "" \
    --status "pretrain" \
    --optim "adamw_torch" \
    --load_best_model_at_end True \
    --load_from_config False \
    --load_from_pretrained False \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-08 \
    --bf16 True \
    --dataloader_drop_last False \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --eval_steps 1 \
    --evaluation_strategy "epoch" \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --group_by_length False \
    --learning_rate 2e-5 \
    --logging_steps 1 \
    --logging_strategy "steps" \
    --lr_scheduler_type "cosine" \
    --num_train_epochs 30 \
    --output_dir ./checkpoints \
    --report_to tensorboard \
    --resume_from_checkpoint None \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 5 \
    --seed 42 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --weight_decay 0. \
    --warmup_ratio 0.03
