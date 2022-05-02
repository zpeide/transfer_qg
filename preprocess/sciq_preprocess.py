import argparse
import json, os, sys 
from nltk import tokenize
from tqdm import tqdm
import spacy
from transformers import BertTokenizer


def parse_sciq_items(indir, outdir):
    max_src_len  = 464
    SEP_TOK = '[SEP]'
    bert_tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    train_ds = json.load(open(os.path.join(indir, "train.json")))
    dev_ds = json.load(open(os.path.join(indir, "valid.json")))
    test_ds = json.load(open(os.path.join(indir, "valid.json")))

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer.add_special_case(SEP_TOK, [{'ORTH': SEP_TOK}])

    def process_item(item):
        support = item['support']
        question = item['question']
        answer = item['correct_answer']
        tokens = bert_tokenizer(support, answer, truncation="only_first", max_length=max_src_len)
        src = bert_tokenizer.decode(tokens['input_ids'])
        text_src = bert_tokenizer.decode(tokens['input_ids'][1:])
        d = {'src': src}
        doc = nlp(text_src)
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
        d['text'] = text
        d['text_ans'] = text_ans
        d['pos'] = pos
        d['ner'] = ner
        d['lemma'] = lemma
        d['ans_mark'] = ans_mark
        d['tgt'] = question
        d['answer'] = answer
        return d        

    all_processed = []
    for ds in [train_ds, dev_ds, test_ds]:
        processed_ds = []
        for i, item in enumerate(tqdm(ds)):
            d = process_item(item)
            processed_ds.append(d)
        all_processed.append(processed_ds)

    
    for ds, fname in zip(all_processed, ['train.jsonl', 'dev.jsonl', 'test.jsonl']):  
        f = open(os.path.join(outdir, fname), "w")
        for d in ds:
            f.write(json.dumps(d))
            f.write('\n')
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--indir",
        type=str,
        default="../data/SciQ",
        help="sqiq dataset folder."
    )


    parser.add_argument(
        "--outdir",
        type=str,
        default="../data/SciQ",
        help="folder for preprocessed dataset file."
    )

    args = parser.parse_args()

    print(args)
    parse_sciq_items(args.indir, args.outdir)