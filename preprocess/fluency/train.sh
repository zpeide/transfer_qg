# !/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

python run_model.py \
       --config_name bert-base-uncased \
       --model_type bert \
       --output_dir ckpt \
       --overwrite_output_dir \
       --tokenizer_name bert-base-uncased \
       --train_data_file tgt.train.txt --eval_data_file tgt.dev.txt \
       --line_by_line \
       --learning_rate 2e-5 \
       --block_size 128 \
       --per_gpu_train_batch_size 16 \
       --per_gpu_eval_batch_size 8 \
       --do_train \
       --evaluate_during_training \
       --num_train_epochs 10
       
