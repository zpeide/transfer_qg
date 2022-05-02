FLUENCY=$1
PPL=$2
# path of training data
TRAIN_FILE=../../data/RACE/target.jsonl
# path of selected source data
SELECTED_FILE=../../data/nq/race_gmm_l2_order/1000.jsonl
# folder used to save fine-tuned checkpoints
OUTPUT_DIR=./nq_unilm_ckpt/race_pseudo_label/ds_finetuned_then_PL/fluency_and_PPL_aa/${FLUENCY}_${PPL}/
# checkpoint trained on source, need to transfer
CKPT_PATH=./nq_unilm_ckpt/race_gmm_l2_ans-aware/1000/ckpt-800/
# folder used to cache package dependencies
CACHE_DIR=./race_unilm_ckpt/cache
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
python ru_uda_s2s.py \
        --train_file ${TRAIN_FILE} \
        --selected_file ${SELECTED_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --model_type unilm \
        --model_name_or_path unilm1.2-base-uncased \
        --recover_path ${CKPT_PATH} \
        --tokenizer_name unilm1.2-base-uncased \
        --max_source_seq_length 464 \
        --max_target_seq_length 48   \
        --per_gpu_train_batch_size 8 \
        --gradient_accumulation_steps 1   \
        --learning_rate 1e-4 \
        --num_warmup_steps 500 \
        --num_training_epochs 10 \
        --keep_prob 0.15 \
        --random_prob 0.15 \
        --cache_dir ${CACHE_DIR} \
        --pred_device_num 1 \
        --pred_per_gpu_batch_size 64 \
        --beam_size 2 \
	--use_perplexity \
        --use_fluency \
	--perplexity_threshold ${PPL} \
        --fluency_threshold ${FLUENCY} \
        --only_in_domain
