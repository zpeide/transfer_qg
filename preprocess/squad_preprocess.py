import json
import os
import numpy as np
from itertools import chain
from nltk.corpus import stopwords
import argparse
import hashlib
import spacy
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
import torch
from rank_bm25 import BM25Okapi

SEP_TOK = '[SEP]'
SEP_POS = 'AUX'
SEP_NER = 'O'
SEP_DEP = 'det'


noun_chunk_pos = {_:1 for _ in ['PROPN', 'NUM', 'DET', 'ADJ', 'NOUN', 'ADP', 'PART']}
verb_chunk_pos = {_:1 for _ in ['VERB', 'AUX']}

# use spacy
def parse_squad_items(squad_path, out_path):
    #spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")

    items = json.load(open(squad_path))['data']
    all_qas = []
    no_ans_cnt = 0
    src_max_len = 384
    for iidx, item in enumerate(tqdm(items)):
        title = item['title']
        for pid, para in enumerate(tqdm(item['paragraphs'], desc='para')):
            context = para['context']
            context = context.replace('\n', '  ').strip()
            context_doc = nlp(context)

            sents = [_ for _ in context_doc.sents]
            sents_len = [len(_) for _ in sents]

            text = [_.text for _ in context_doc]
            lemma = [_.lemma_ for _ in context_doc]
            pos = [_.pos_ for _ in context_doc]
            ner = [_.ent_type_ for _ in context_doc]

            discussion = para['qas']
            for dis in discussion:
                answers = dis['answers']
                question = dis['question']
                q_id = dis['id']
                question = question.replace('\n', ' ').strip()
                qdoc = nlp(question)

                q_text = [_.text for _ in qdoc]
                q_lemma = [_.lemma_ for _ in qdoc]

                ans_mark = []
                ans_sentences = []
                enter_answer = False
                for si, sent in enumerate(sents):
                    ans_words = []
                    for i, token in enumerate(sent):
                        # whether token is in answer
                        in_ans = False
                        for ans in answers:
                            start = ans['answer_start']
                            end = ans['answer_start'] + len(ans['text']) 
                            token_end = token.idx + len(token)
                            if token.idx < end and token_end > start:
                                in_ans = True
                                break
                        if in_ans:
                            if not enter_answer:
                                # ans_mark.append(1)
                                enter_answer = True
                            ans_words.append(i)
                            ans_mark.append(1)
                        else:
                            if enter_answer:
                                # ans_mark.append(1)
                                enter_answer = False
                                ans_sentences.append((si, ans_words))
                                ans_words = []
                            ans_mark.append(0)
                    if enter_answer:
                        enter_answer = False
                        ans_sentences.append((si, ans_words))
                        ans_words = []

                if len(ans_sentences) == 0:
                    no_ans_cnt += 1
                    continue
                # check multiple answers in multiple sentences
                if len(ans_sentences) > 1:
                    bak_ans_sentences = {si: _ for si, _ in ans_sentences}
                    corpus = [[w.text.lower() for w in _] for _ in sents]
                    bm25 = BM25Okapi(corpus)
                    ans_scores = []
                    pseudo_scores = []
                    for si, answords in ans_sentences:
                        #print('si:', si, ' ,sents:', sents[si][answords[0]], answords)
                        anstext = [str(sents[si][_]).lower() for _ in answords]
                        query = [_.lower() for _ in q_text[:-1]] + anstext
                        doc_scores = bm25.get_scores(query)
                        msi = np.argmax(doc_scores)
                        # this means the label is not the real answer.
                        pseudo_scores.append(doc_scores[si])
                        if msi != si and msi in bak_ans_sentences:
                            ans_scores.append(-1)
                        else:
                            ans_scores.append(doc_scores[si])
                    target_ans_idx = np.argmax(ans_scores)
                    if ans_scores[target_ans_idx] == -1:
                        target_ans_idx = np.argmax(pseudo_scores)
                        print("wwwwwwwwwwrong item.", question)
                    ans_sentences = ans_sentences[target_ans_idx]
                else:
                    ans_sentences = ans_sentences[0]

                # src is for UNILM train. we surround the answer text with sep token in the context.
                start_w_idx = 0
                end_w_idx = len(text)

                if len(text) > src_max_len - 2:
                    start_sent_idx = ans_sentences[0]
                    end_sent_idx = min(len(sents_len), ans_sentences[0]+1)
                    while sum(sents_len[start_sent_idx-1:end_sent_idx+1]) < src_max_len and end_sent_idx - start_sent_idx < 5:
                        start_sent_idx = max(0, start_sent_idx - 1)
                        end_sent_idx = min(len(sents_len), end_sent_idx + 1)
                    start_w_idx = sum(sents_len[:start_sent_idx])
                    end_w_idx = sum(sents_len[:end_sent_idx])

                src = text.copy()
                src.insert(sum(sents_len[:ans_sentences[0]]) + np.min(ans_sentences[1]), SEP_TOK) 
                src.insert(sum(sents_len[:ans_sentences[0]]) + np.max(ans_sentences[1]) + 2, SEP_TOK) 
                    
                src = ' '.join(src[start_w_idx:end_w_idx+2])

                answer_text = [sents[ans_sentences[0]][_].text for _ in ans_sentences[1]]

                _ans_ = sum(sents_len[:ans_sentences[0]]) + np.min(ans_sentences[1])
                _ans_end_ = sum(sents_len[:ans_sentences[0]]) + np.max(ans_sentences[1]) + 2
                for i in range(_ans_):
                    ans_mark[i] = 0
                for i in range(_ans_end_, len(ans_mark)):
                    ans_mark[i] = 0

                qa_item = {
                    'title': title,
                    'qid': q_id,
                    'src': src,
                    'text': text[start_w_idx:end_w_idx],
                    'lemma': lemma[start_w_idx:end_w_idx],
                    'pos': pos[start_w_idx:end_w_idx],
                    'ner': ner[start_w_idx:end_w_idx],
                    'ans_mark': ans_mark[start_w_idx:end_w_idx],
                    'answer': answer_text,
                    'gold': q_text,
                    'tgt': q_text,
                    'q_text': q_text,
                    'q_lemma': q_lemma,
                }

                all_qas.append(qa_item)
    print(f"{no_ans_cnt} items have no answer provided.")
    outfile = open(out_path, "w")
    for qa in all_qas:
        outfile.write(json.dumps(qa))
        outfile.write("\n")

    outfile.close()

def gen_split1(args, dataset, nsplits=10):
    """Split dataset to nsplits in paragraph level. """
    import random
    ds = json.load(open(dataset))['all_data']
    para2ids = {}
    for i, d in enumerate(ds):
        para2ids[d['pid']] = para2ids.get(d['pid'], []) + [i]
    all_pids = list(para2ids.keys())
    random.shuffle(all_pids)
    split_len = len(all_pids) // nsplits + 1
    for n in range(nsplits):
        train_id = list(chain.from_iterable( [para2ids[_] for _ in (all_pids[: n*split_len] + all_pids[(n+1)*split_len:])]))
        train_data = [ds[_] for _ in train_id]
        dev_id = list(chain.from_iterable( [para2ids[_] for _ in (all_pids[n*split_len : (n+1)*split_len])]))
        dev_data = [ds[_] for _ in dev_id]
        json.dump(train_data, open(os.path.join(args.out_dir, "squad_split1_train_{}.json".format(n)), "w"))
        json.dump(dev_data, open(os.path.join(args.out_dir, "squad_split1_dev_{}.json".format(n)), "w"))
        print("train: {} - dev: {}".format(len(train_data), len(dev_data)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--infile",
        type=str,
        default="train-v1.1.json",
        help="squad train dataset file."
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default='train',
        help='phase.'
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="../data/squad",
        help="folder for preprocessed dataset file."
    )

    args = parser.parse_args()

    print(args)
    out_path = os.path.join(args.outdir, f"{args.prefix}.jsonl")
    parse_squad_items(args.infile, out_path)
