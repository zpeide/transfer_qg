import argparse
import enum
import json, glob, re, os
from nltk.translate.bleu_score import sentence_bleu
from nltk import tokenize
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np
import spacy
import torch
from transformers import BertTokenizer

SEP_TOK = '[SEP]'

def parse_race_items(in_path, out_path):
    bert_tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    SEP_ID = bert_tokenizer.convert_tokens_to_ids([SEP_TOK])[0]

    eqg_train = json.load(open(os.path.join(in_path, "key-race/train.json")))
    eqg_dev = json.load(open(os.path.join(in_path, "key-race/dev.json")))
    eqg_test = json.load(open(os.path.join(in_path, "key-race/test.json")))
    max_src_len  = 464

    def retrieve_raw(part):
        options_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        qa_ds = []
        fs = glob.glob(os.path.join(in_path, f"{part}/**/*.txt"))
        for f in fs:
            d = json.loads(open(f).readline())
            for i, (q, ans, options) in enumerate(zip(d['questions'], d['answers'], d['options'])):
                qa_ds.append({
                    'article_id': d['id'],
                    'answer': options[options_to_number[ans]],
                    'article': d['article'],
                    'tgt': q,
                })

        return qa_ds
    max_sents = 15
    max_idx_to_end_sent=15
    train_ds = retrieve_raw("train")
    dev_ds = retrieve_raw("dev")
    test_ds = retrieve_raw("test")
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer.add_special_case(SEP_TOK, [{'ORTH': SEP_TOK}])
    def filter_eqg_data(ds, eqg_ds):    
        out_ds = []
        q_dict = {_['question']: 1 for _ in eqg_ds}
        raw_q_dict =  {}
        raw_item_dict = {}
        def locate_answer_in_article(d):
            sents = tokenize.sent_tokenize(d['article'])
            corpus = [tokenize.word_tokenize(_.lower()) for _ in sents]
            bm25 = BM25Okapi(corpus)
            query = t.split() + tokenize.word_tokenize(d['answer'].lower())
            doc_scores = bm25.get_scores(query)
            idx = np.argmax(doc_scores)
            # print(doc_scores[idx], sents[idx], t)
            d['target_sent_num'] = int(idx)
            d['target_sent'] = sents[idx]
            d['target_bm25_score'] = float(doc_scores[idx])
            sent_words = [tokenize.word_tokenize(_) for _ in sents]
            swl = [len(_) for _ in sent_words]
            ans_words = tokenize.word_tokenize(d['answer'])
            s_idx = 0
            e_idx = len(sents)
            if len(sents) > max_sents or np.sum(swl) > max_src_len - len(ans_words) - 4 :
                s_idx = max(0, idx - 1)
                max_end = min(len(sents), idx+max_idx_to_end_sent)
                e_idx = min(max_end, idx+1)
                while np.sum(swl[s_idx-1: e_idx+1]) < max_src_len - len(ans_words) - 4:
                    s_idx = max(0, s_idx - 1)
                    if e_idx - s_idx >= max_sents:
                        break
                    e_idx = min(max_end, e_idx+1)
                    if s_idx == 0 and e_idx == max_end:
                        break
            src = ' '.join(sents[s_idx:e_idx])
            tokens = bert_tokenizer(src, d['answer'], truncation="only_first", max_length=max_src_len)
            src = bert_tokenizer.decode(tokens['input_ids'])

            d['src'] = src #' '.join(sents[s_idx:idx]) + ' ' + SEP_TOK + ' ' + d['answer'] + ' ' + SEP_TOK + ' ' + ' '.join(sents[idx:e_idx])
            # sep_pos = torch.where(torch.tensor(tokens['input_ids']) == SEP_ID)[0][0]
            # text_src = bert_tokenizer.decode(tokens['input_ids'][1:sep_pos]).split()
            text_src = bert_tokenizer.decode(tokens['input_ids'][1:]).split()
            d['text'] = text_src
            raw_item_dict[i] = 1
            return d

        for i, d in enumerate(tqdm(ds, desc='filter')):
            t = ' '.join(tokenize.word_tokenize(d['tgt'])).lower()
            t = t.replace("''", '"').replace("``", '"')
            raw_q_dict[t] = raw_q_dict.get(t, [])
            raw_q_dict[t].append(i)
            if t in q_dict:
                out_ds.append(locate_answer_in_article(d))
        
        print(len(out_ds), len(q_dict), len(eqg_ds))

        # find missing items.
        for j, q in enumerate(tqdm(q_dict.keys())):
            if q not in raw_q_dict:
                for rq, il in raw_q_dict.items():
                    b3 = sentence_bleu([rq.split()], q.split(), weights=(0.5, 0.5))
                    if b3 > 0.7:
                        print(rq)
                        print(q)
                        print(b3)
                        for i in il:
                            if i not in raw_item_dict:
                                out_ds.append(locate_answer_in_article(ds[i]))
                        break
        print(len(out_ds), len(q_dict), len(eqg_ds))

        # get lemma/pos/ner
        for i, d in enumerate(tqdm(out_ds, desc='ner/pos/spacy')):
            doc = nlp(' '.join(d['text']))
            sents = [[_ for _ in s] for s in doc.sents]
            in_ans = False
            text = []
            text_ans = []
            pos = []
            ner = []
            lemma = []
            ans_mark = []
            for sent in sents:
                for tok in sent:
                    if tok.text ==  SEP_TOK:
                        if not in_ans:
                            in_ans = True
                        else:
                            in_ans = False
                        continue
                    if in_ans:
                        ans_mark.append(1)
                    else:
                        ans_mark.append(0)
                        text.append(tok.text)
                    text_ans.append(tok.text)
                    pos.append(tok.pos_)
                    lemma.append(tok.lemma_)
                    ner.append(tok.ent_type_)
            d['text_ans'] = text_ans
            d['text'] = text
            d['pos'] = pos
            d['ner'] = ner
            d['lemma'] = lemma
            d['ans_mark'] = ans_mark
        print("num items: ", len(out_ds))
        return out_ds

    train_ds = filter_eqg_data(train_ds, eqg_train)
    dev_ds = filter_eqg_data(dev_ds, eqg_dev)
    test_ds = filter_eqg_data(test_ds, eqg_test)

    for ds, fname in zip([train_ds, dev_ds, test_ds], ['train.jsonl', 'dev.jsonl', 'test.jsonl']):  
        f = open(os.path.join(out_path, fname), "w")
        for d in ds:
            f.write(json.dumps(d))
            f.write('\n')
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--indir",
        type=str,
        default="../data/RACE",
        help="squad train dataset file."
    )


    parser.add_argument(
        "--outdir",
        type=str,
        default="../data/RACE",
        help="folder for preprocessed dataset file."
    )

    args = parser.parse_args()

    print(args)
    parse_race_items(args.indir, args.outdir)
