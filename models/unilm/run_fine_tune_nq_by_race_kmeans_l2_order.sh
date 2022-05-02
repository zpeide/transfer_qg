NUM=$1
# path of training data
TRAIN_FILE=../../data/nq/race_kmeans_l2_order/${NUM}.jsonl 
# folder used to save fine-tuned checkpoints
OUTPUT_DIR=./nq_unilm_ckpt/kmeans_race_l2_order/${NUM}/
# folder used to cache package dependencies
MODEL_PATH=nq_unilm_ckpt/nq_random_ckpt/epoch-10/pytorch_model.bin
CACHE_DIR=./cache
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
python -m torch.distributed.launch --nproc_per_node=4 run_seq2seq.py --train_file ${TRAIN_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --model_type unilm \
        --model_name_or_path ${MODEL_PATH} \
        --tokenizer_name unilm1.2-base-uncased \
        --config_name unilm1.2-base-uncased \
        --max_source_seq_length 464 \
        --max_target_seq_length 48   \
        --per_gpu_train_batch_size 6 \
        --gradient_accumulation_steps 1   \
        --learning_rate 1e-5 \
	--num_training_steps 1000 \
	--save_steps 200 \
        --no_lr_schedule \
        --cache_dir ${CACHE_DIR}