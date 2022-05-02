export CUDA_VISIBLE_DEVICES=$3
# path of the fine-tuned checkpoint
MODEL_PATH=$1
# input file that you would like to decode
INPUT_JSON=$2 # ../../../data/squad/items.valid.jsonl
SPLIT=validation
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
python decode_seq2seq.py \
  --model_type unilm --tokenizer_name unilm1.2-base-uncased --input_file ${INPUT_JSON} --split $SPLIT --do_lower_case \
  --model_path ${MODEL_PATH} --max_seq_length 512 --max_tgt_length 48 --batch_size 64 --beam_size 5 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "."

