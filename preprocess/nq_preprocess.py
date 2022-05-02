import json
import re
import argparse
import os
from tqdm import tqdm
import spacy
from transformers.models import bert
from utils import clean_text
from spacy.attrs import ORTH, NORM
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer


SEP_TOKEN = {'token': '[SEP]', 'html_token': False}

max_src_len = 412
max_tokens_len  = 464

bert_tokenizer = BertTokenizer.from_pretrained("bert-large-cased")

def tokenize(text, pipeline):
    text = re.sub(r'\[\s*(sep|SEP)\s*\]', ' [SEP] ', text)
    text = bert_tokenizer.decode(bert_tokenizer(text)['input_ids'][1:-1])
    text = re.sub(r'\[\s*(sep|SEP)\s*\]', ' [SEP] ', text)
    doc = pipeline(text)
    src = [_.text for _ in doc]
    sents = [[_.text for _ in s] for s in doc.sents]
    sent_lens = [len(_) for _ in sents]
    lemma = [_.lemma_ for _ in doc]
    pos = [_.pos_ for _ in doc]
    ner = [_.ent_type_ for _ in doc] 
    # remove the added [SEP], which is used to find the answer position, and remove it for rnn qg.
    ans_tag, ans_text = [], []
    all_ans_text_options = {}
    text, r_pos, r_ner, r_lemma = [], [], [], []
    in_ans = False
    ans_positions = {}
    ans_start_pos = 0
    ans_end_pos = -1
    idx_in_para = 0
    for si, sent in enumerate(sents):
        sent_text = []
        sent_pos = []
        sent_ans = []
        sent_ner = []
        sent_lemma = []
        for i, w in enumerate(sent):
            if w == SEP_TOKEN['token']:
                if not in_ans:
                    ans_start_pos = i
                else:
                    ans_end_pos = i
                    ans_positions[si] = [ans_start_pos, ans_end_pos]
                    all_ans_text_options[si]= ans_text
                    ans_text = []
                in_ans = not in_ans
            else:
                sent_text.append(w)
                sent_pos.append(pos[idx_in_para])
                sent_ner.append(ner[idx_in_para])
                sent_lemma.append(lemma[idx_in_para])
                if in_ans:
                    sent_ans.append(1)
                    ans_text.append(w)
                else:
                    sent_ans.append(0)
            idx_in_para += 1

        text.append(sent_text)
        r_pos.append(sent_pos)
        r_ner.append(sent_ner)
        ans_tag.append(sent_ans)
        r_lemma.append(sent_lemma)

    return src, text, r_pos, r_ner, ans_tag, sents, ans_positions, all_ans_text_options, r_lemma


def process_kilt_nq(input_file, max_words=300):
    items = []
    f = open(input_file)
    lines = f.readlines()
    for line in tqdm(lines):
        nq = json.loads(line)
        question = nq['input']
        evidences = nq['output'][-1]['provenance'][0]['meta']['evidence_span']
        unique_evidences = []
        for evd in evidences:
            is_substring = False
            for i, t in enumerate(unique_evidences):
                # sub string
                if len(t) < len(evd) and evd.startswith(t):
                    is_substring = True
                    unique_evidences[i] = evd
                    break
                elif t.startswith(evd):
                    is_substring = True
                    break
            if is_substring:
                continue
            else:
                unique_evidences.append(evd)
        src = ' '.join(unique_evidences)
        question = nq['input']
        items.append({'src': src, 'tgt': question})
    f.close()
    return items


def process_raw_nq(args, input_file, max_words=386):
    nlp = spacy.load("en_core_web_sm")
    # not split [SEP]
    nlp.tokenizer.add_special_case(SEP_TOKEN['token'], [{'ORTH': SEP_TOKEN['token']}])

    items = []
    f = open(input_file)
    lines = f.readlines()
    none_ans_cnt = 0
    find_no_cand = 0
    stidx = len(lines)//args.nsubset * args.use_subset
    edidx = len(lines)//args.nsubset * (args.use_subset + 1)
    for line in tqdm(lines[stidx: edidx]):
        nq = json.loads(line)
        question = nq['question_text']
        is_eval = False
        if nq.get('document_tokens') is not None:
            tokens = nq['document_tokens']
            # tokens = [_['token'] for _ in tokens]
            is_eval = True
        else:
            tokens = nq['document_text'].split()
        annotations = nq['annotations']

        start_token = -1
        end_token = -1
        ans_idx = -1
        ans_len = -1
        for _i_, cand in enumerate(annotations):
            long_ans = cand['long_answer']
            short_ans = cand['short_answers']
            if len(short_ans) == 0:
                continue
            if is_eval:
                t = ' '.join([_['token'] for _ in tokens[long_ans['start_token']:long_ans['end_token']]]).lower()
            else:
                t = ' '.join(tokens[long_ans['start_token']:long_ans['end_token']]).lower()
            if len(re.findall(r'<table>.*</td>', t)) == 0 and long_ans['end_token'] - long_ans['start_token'] > ans_len:
                start_token = long_ans['start_token']
                end_token = long_ans['end_token']
                ans_len = end_token - start_token
                ans_idx = _i_
                continue
        if  start_token < 0 or end_token < 0 or ans_idx < 0:
            find_no_cand += 1
            continue
        short_answers = annotations[ans_idx]['short_answers']
        # remove all no short answer candidate.

        inserted_cnt = 0
        if short_answers is not None:
            for short_ans in short_answers:
                if is_eval:
                    tokens.insert(short_ans['start_token'] + inserted_cnt, SEP_TOKEN)
                else:
                    tokens.insert(short_ans['start_token'] + inserted_cnt, '[SEP]')
                inserted_cnt += 1
                if is_eval:
                    tokens.insert(short_ans['end_token'] + inserted_cnt, SEP_TOKEN)
                else:
                    tokens.insert(short_ans['end_token'] + inserted_cnt, '[SEP]')
                inserted_cnt += 1

        end_token += inserted_cnt
        if is_eval:
            text_tokens = [_['token'] for _ in tokens[start_token: end_token] if not _['html_token']]
        else:
            text_tokens = tokens[start_token: end_token]  #[_['token'] for _ in tokens[start_token: end_token] if not _['html_token']]
                
        src = ' '.join(text_tokens)
        if len(re.findall(r'<table>.*</td>', src)) > 0:
            find_no_cand += 1
            #print(src)
            continue
        src = clean_text(src)
        if len(src.strip()) < 10: 
            continue

        baksrc = src
        src, text, pos, ner, ans_tag, sents, all_ans_positions, ans_text_options, lemma = tokenize(src, nlp)
        question = clean_text(question)
        question, _, _, _, _, _, _, _, qlemma = tokenize(question, nlp)   
        question.append('?')
        qlemma = [_ for s in qlemma for _ in s]

        if len(src) > max_src_len:
            if len(all_ans_positions) > 1:                
                bak_ans_sentences = all_ans_positions
                ans_sentences = sorted(all_ans_positions.items(), key=lambda x: len(x[1]))
                corpus = [[w.lower() for w in _] for _ in sents]
                bm25 = BM25Okapi(corpus)
                ans_scores = []
                for si, answords in ans_sentences:
                    answords = ans_text_options[si]
                    #print('si:', si, ' ,sents:', sents[si][answords[0]], answords)
                    anstext = [ _.lower() for _ in ans_text_options[si]]
                    query = [_.lower() for _ in question[:-1]] + anstext
                    doc_scores = bm25.get_scores(query)
                    msi = np.argmax(doc_scores)
                    # this means the label is not the real answer.
                    if msi != si and msi in bak_ans_sentences:
                        ans_scores.append(-1)
                    else:
                        ans_scores.append(doc_scores[si])
                target_ans_idx = np.argmax(ans_scores)
                ans_sentences = ans_sentences[target_ans_idx]
                
            else:
                ans_sentences = list(all_ans_positions.items())[0]
            start_sent_idx = ans_sentences[0] - 1
            end_sent_idx = ans_sentences[0] + 1
            sent_lens = [len(_) for _ in sents]
            while sum(sent_lens[start_sent_idx-1:end_sent_idx+1]) < max_src_len and end_sent_idx - start_sent_idx < 5:
                start_sent_idx = max(0, start_sent_idx - 1)
                end_sent_idx = min(len(sents), end_sent_idx + 1)
                if start_sent_idx == 0 and end_sent_idx == len(sents):
                    break
            
            start_w_idx = sum(sent_lens[:start_sent_idx])
            end_w_idx = sum(sent_lens[:end_sent_idx])

            src = src[start_w_idx: end_w_idx]
            ans_text = ans_text_options[ans_sentences[0]]
            text = text[start_sent_idx: end_sent_idx] 
            pos = pos[start_sent_idx: end_sent_idx]
            ner = ner[start_sent_idx: end_sent_idx]
            ans_tag = ans_tag[start_sent_idx: end_sent_idx]
            lemma = lemma[start_sent_idx: end_sent_idx]
        else:
            try:
                ans_sentences = list(all_ans_positions.items())[0]
            except:
                print(all_ans_positions)
                print(baksrc)
                print(' '.join(src))
            ans_text = ans_text_options[ans_sentences[0]] #all_ans_positions[ans_sentences[0]]

        src = ' '.join(src)
        # src = re.sub(r'\[\s*(sep|SEP)\s*\]', '[SEP]', src)
        src = re.sub(r'\[\s*(sep|SEP)\s*\]', ' ', src)
        if type(ans_text) is list:
            ans = ' '.join(ans_text)
        else:
            ans = ans_text

        input_ids = bert_tokenizer(src, ans, max_length=max_tokens_len, truncation="only_first")['input_ids']
        src = bert_tokenizer.decode(input_ids)
        
        text = [_ for s in text for _ in s]
        pos = [_ for s in pos for _ in s]
        ner = [_ for s in ner for _ in s]
        ans_tag = [_ for s in ans_tag for _ in s]
        lemma = [_ for s in lemma for _ in s]

        item = {
            'origin': ' '.join(text),
            'src': src, 
            'tgt': question,
            'pos': pos,
            'ner': ner,
            'ans_mark': ans_tag,
            'answer': ans_text,
            'text': text,
            'lemma': lemma,
            'gold': question,
            'q_text': question,
            'q_lemma': qlemma,
        }
        items.append(item)
    f.close()
    print(none_ans_cnt, find_no_cand)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_file',
        type=str,
        help='json file for processed squad data.'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        help='oputput file name.'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default='train',
        help='phase.'
    )
    parser.add_argument(
        '--nsubset',
        type=int,
        default=1,
        help='split data items to n subsets.'
    )
    parser.add_argument(
        '--use_subset',
        type=int,
        default=0,
        help='use i subset.'
    )

    args = parser.parse_args()
    # items = process_kilt_nq(args.data_file)
    items = process_raw_nq(args, args.data_file)
    out_file = open(os.path.join(args.outdir, 'items.{}.{}.jsonl'.format(args.use_subset, args.prefix)), "w")
    for d in items:
        out_file.write(json.dumps(d))
        out_file.write('\n')
    out_file.close()

if __name__ == '__main__':
    main()
