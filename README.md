This is the reposity for reproducing "Unsupervised Domain Adaptation for Question Generation with Domain Data Selection and Self-training".

## Data preprocess
- **Natural Question**
```bash
python nq_preprocess.py --data_file path_to/Google_Natural_Question/v1.0-simplified_simplified-nq-train.jsonl --outdir ../data/nq --prefix train 
python nq_preprocess.py --data_file path_to/Google_Natural_Question/v1.0-simplified_nq-dev-all.jsonl  --outdir ../data/nq  --prefix dev
```

- **SQuAD**
```bash
python squad_preprocess.py --infile ../data/squad/train-v1.1.json --outdir ../data/squad --prefix train
python squad_preprocess.py --infile ../data/squad/dev-v1.1.json --outdir ../data/squad --prefix dev 
```

- **RACE:**   race_preprocess.py


- **SciQ:** sciq_preprocess.py

- **MLQuestions:** mlquestions_preprocess.py

## Domain discriminator
```
cd preprocess/domain_discriminator
```
### Unsupervised Domain Clustering
- Create BERT encoding for each domain, and perform clustering.
```
python domain_data_selec_with_UDC.py
```
- Visualization Analysis, and create selected data for each domain.
```
(jupyternotebook) interactive
data_selection_UDC_analysis.ipynb
```


## Base model training
The base model and part of the code are adopted from [unilm](https://github.com/microsoft/unilm).
>- **NQ:**  ./run_fine_tune_nq_unilm.sh
>- **RACE:** ./run_fine_tune_race_unilm.sh
>- **SciQ:**  ./run_fine_tune_sciq_unilm.sh

## Transfer
### with Random selected data.
```
./run_fine_tune_nq_random_selection.sh 1000
```

### Re-fine-tuning NQ for RACE
>- **with gmm (l2 distance) RACE order:** ./run_fine_tune_nq_by_race_gmm_l2_order.sh 1000

### Re-fine-tuning NQ for SciQ
>- **with gmm (l2 distance) SciQ order:**  ./run_fine_tune_nq_by_sciq_gmm_l2_order.sh 1000



## Fine-tune with Pseudo-Labeling

### RACE
>- **pseudo-labeling only, no filter:** ./run_uda_race_no_filter_pseudo-only.sh  
>- **pseudo-labeling only, fluency:**    ./run_uda_race_fluency_pseudo-only.sh 10.5
>- **pseudo-labeling only, perplexity:** run_uda_race_perplexity_pseudo-only.sh 8.5
>- **pseudo-labeling only, fluency && perplexity:**  ./run_uda_race_fluency_and_PPL_pseudo-only.sh 10.5 8.5
>- **Fluency:** run_uda_race_fluency_reine-tuned.sh 10.5
>- **Perplexity:** run_uda_race_perplexity_reine-tuned.sh 8.5
>- **Fleuncy + Perplexity:** ./run_uda_race_fluency_and_PPL_reine-tuned.sh 10.5 8.5

### Selected data + Pseudo-Labeling

>- **No Filter:** ./run_uda_race_no_filter_ds+pl.sh
>- **Fluency:** ./run_uda_race_fluency_ds+pl.sh 10.5
>- **Perplexity:** ./run_uda_race_perplexity_ds+pl.sh 8.5 
>- **Fluency + Perplexity:** ./run_uda_race_fluency_and_PPL_ds+pl.sh 10.5 8.5

# Decoding

## NQ

### extract `src` from dev set to nq_unilm_ckpt/src.txt, and lower case of `tgt` to nq_unilm_ckpt/gold.txt, for further evaluation.

### run decoding. 
```
./run_unilm_decoding.sh nq_unilm_ckpt/nq_random_ckpt/epoch-10/ ../../data/MLQuestions/test.jsonl 0,1
```

### run evaluation
```
./score.sh squad_unilm_ckpt/ckpt/ squad_unilm_ckpt/gold.txt squad_unilm_ckpt/src.txt
```



