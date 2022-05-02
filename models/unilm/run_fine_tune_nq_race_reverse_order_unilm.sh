MASTER_ADDR=127.0.0.1
MASTER_PORT=55663
# path of training data
TRAIN_FILE=../../data/nq/race_order/race_dis_ordered.jsonl 
# folder used to save fine-tuned checkpoints
OUTPUT_DIR=./nq_unilm_ckpt/race_order/high2low/
# folder used to cache package dependencies
CACHE_DIR=./cache
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
python -m torch.distributed.launch --nproc_per_node=4 --master_addr='127.0.0.1' --master_port=9901  run_seq2seq.py --train_file ${TRAIN_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --model_type unilm \
        --model_name_or_path unilm1.2-base-uncased  \
        --max_source_seq_length 464 \
        --max_target_seq_length 48   \
        --per_gpu_train_batch_size 6 \
        --gradient_accumulation_steps 1   \
        --learning_rate 1e-4 \
        --num_warmup_steps 500 \
        --num_training_epochs 10 \
        --no_shuffle \
        --cache_dir ${CACHE_DIR}
