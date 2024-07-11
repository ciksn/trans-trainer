#!/bin/bash
export MODEL_LOAD='/home/zeyu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574'
export CONFIG_LOAD='/home/zeyu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574'
export TOKENIZER_LOAD='/home/zeyu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574'
export CACHE_DIR='/home/zeyu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574'

export DATASET_NAME="msrvtt_dataset"
export CAPTION_FILE_PATH="/home/zeyu/work/deep_learning/row_dataset/video_captioning/msrvtt/train_val_test_annotation/train_val_test_videodatainfo.json"
export VIDEO_FOLDER_PATH="/home/zeyu/work/deep_learning/extracted_dataset/msrvtt/CLIP-vitL14"

deepspeed ../train.py \
    --deepspeed /home/zeyu/work/deep_learning/functional_files/trans-trainer/scripts/zero3.json \
    --model_name_or_path $MODEL_LOAD \
    --config_name_or_path $CONFIG_LOAD \
    --need_tokenizer True \
    --tokenizer_name_or_path $TOKENIZER_LOAD \
    --cache_dir $CACHE_DIR \
    --version v1 \
    --dataset_name $DATASET_NAME \
    --caption_file_path $CAPTION_FILE_PATH \
    --video_folder_path $VIDEO_FOLDER_PATH \
    --caption_seq_len 30 \
    --video_seq_len 30 \
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
    --gradient_checkpointing False \
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