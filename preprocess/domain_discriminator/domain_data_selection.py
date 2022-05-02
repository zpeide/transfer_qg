import argparse
import json
import random
import numpy as np
from tqdm import tqdm


def main(args):
    random.seed(21)
    nq_ds = [json.loads(_) for _ in open(args.source_domain_file)]
    squad_ds = [json.loads(_) for _ in open(args.squad_domain_file)]
    race_ds = [json.loads(_) for _ in open(args.race_domain_file)]

    nq_passages = list({_['origin'].strip().lower():1 for _ in nq_ds}.keys())
    squad_passages = list({' '.join(_['text']).strip().lower():1 for _ in squad_ds}.keys())
    race_passages = list({_['article'].strip().lower(): 1 for _ in race_ds }.keys())

    random.shuffle(nq_passages)
    random.shuffle(squad_passages)
    random.shuffle(race_passages)

    f1 = open( "sq.train.jsonl", 'w')
    f2 = open( "rc.train.jsonl", 'w')
    for p_nq, p_sq, p_rc in zip(nq_passages[:args.train_size], squad_passages[:args.train_size], race_passages[:args.train_size]):
        d = {'src': p_nq, 'is_tgt_domain': 0}
        f1.write(json.dumps(d))
        f1.write('\n')
        d = {'src': p_sq, 'is_tgt_domain': 1}
        f1.write(json.dumps(d))
        f1.write('\n')

        d = {'src': p_nq, 'is_tgt_domain': 0}
        f2.write(json.dumps(d))
        f2.write('\n')
        d = {'src': p_rc, 'is_tgt_domain': 1}
        f2.write(json.dumps(d))
        f2.write('\n')
    f1.close()
    f2.close()    

    f1 = open( "sq.dev.jsonl", 'w')
    f2 = open( "rc.dev.jsonl", 'w')
    for p_nq, p_sq, p_rc in zip(nq_passages[args.train_size: args.train_size+args.dev_size], squad_passages[args.train_size: args.train_size+args.dev_size], race_passages[args.train_size: args.train_size+args.dev_size]):
        d = {'src': p_nq, 'is_tgt_domain': 0}
        f1.write(json.dumps(d))
        f1.write('\n')
        d = {'src': p_sq, 'is_tgt_domain': 1}
        f1.write(json.dumps(d))
        f1.write('\n')

        d = {'src': p_nq, 'is_tgt_domain': 0}
        f2.write(json.dumps(d))
        f2.write('\n')
        d = {'src': p_rc, 'is_tgt_domain': 1}
        f2.write(json.dumps(d))
        f2.write('\n')
    f1.close()
    f2.close()    

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--source_domain_file",
        type=str,
        default="../../data/nq/train.jsonl",
        help="train file from source domain."
    )

    parser.add_argument(
        "--squad_domain_file",
        type=str,
        default="../../data/squad/in_domain_selected_dev_mixed_train.jsonl", #train.jsonl",
        help="target domain - squad data ."
    )

    parser.add_argument(
        "--race_domain_file",
        type=str,
        default="../../data/RACE/in_domain_selected_dev_mixed_train.jsonl", #train.jsonl",
        help="target domain - race data ."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="../../data/",
        help="output folder ."
    )

    parser.add_argument(
        "--train_size",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--dev_size",
        type=int,
        default=1000,
    )


    args = parser.parse_args()

    main(args)
