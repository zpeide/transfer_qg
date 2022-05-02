import ujson
import argparse
# from cytoolz import identity, concat, curry
from cytoolz.curried import curry, identity, concat
from utils import  CONSTANTS
import numpy as np
import os
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

@curry
def make_vocab(parse_file_names, out_path, word_vec_file, vocab_size_limit=45000, least_wf=0, do_lower=True, embedding_size=300):
    word2idx = {
        CONSTANTS.PAD: CONSTANTS.PAD_ID,
        CONSTANTS.UNK: CONSTANTS.UNK_ID,
        CONSTANTS.START: CONSTANTS.START_ID,
        CONSTANTS.END: CONSTANTS.END_ID,
        CONSTANTS.SPLIT: CONSTANTS.SPLIT_ID
    }

    ner2idx = {}
    pos2idx = {}

    wordcnt = {}

    lines = []

    word2lemma = {}
    for pn in parse_file_names:
        lines.extend(open(pn, encoding='utf-8').readlines())

    for line in tqdm(lines):
        d = ujson.loads(line)
        question = d['tgt']
        if 'q_lemma' not in d:
            qlemma = [lemmatizer.lemmatize(_) for _ in question]
        else:
            qlemma = d['q_lemma']
        text = d['text']
        lemma = d['lemma']
        ner = d['ner']
        pos = d['pos']
        ans_mark = d['ans_mark']
        if type(question) is str:
            question = question.split()

        for idx, word in enumerate(text):
                wordcnt[word.lower()] = wordcnt.get(word.lower(), 0) + 1
                ner2idx[ner[idx]] = ner2idx.get(ner[idx], len(ner2idx))
                pos2idx[pos[idx]] = pos2idx.get(pos[idx], len(pos2idx))
                word2lemma[word.lower()] = lemma[idx].lower()
        for idx, word in enumerate(question[:-1]):
            wordcnt[word.lower()] = wordcnt.get(word.lower(), 0) + 1
            word2lemma[word.lower()] = qlemma[idx].lower()

    wordcnt = sorted(wordcnt.items(), key=lambda x: x[1], reverse=True)

    # use fix seed for random unknown word vector generation.
    word2embedding = {}
    np.random.seed(42)
    embedding_size = 300
    lines = open(word_vec_file, "r", encoding='utf-8').readlines()
    for line in tqdm(lines):
        word_vec = line.strip().split()
        word = word_vec[0].lower()
        vec = np.array(word_vec[-embedding_size:], dtype=np.float32)
        if word not in word2embedding:
            word2embedding[word] = vec

    if CONSTANTS.UNK not in word2embedding:
        word2embedding[CONSTANTS.UNK] = np.random.normal(loc=0, scale=0.5, size=(embedding_size))
    if CONSTANTS.START not in word2embedding:
        word2embedding[CONSTANTS.START] = np.random.normal(loc=0, scale=0.5, size=(embedding_size))
    if CONSTANTS.END not in word2embedding:
        word2embedding[CONSTANTS.END] = np.zeros(embedding_size)  # np.random.normal(loc=0., scale=0.5, size=(embedding_size))
    if CONSTANTS.PAD not in word2embedding:
        word2embedding[CONSTANTS.PAD] = np.zeros(embedding_size)  # np.random.normal(loc=0., scale=0.5, size=(embedding_size))
    if CONSTANTS.SPLIT not in word2embedding:
        word2embedding[CONSTANTS.SPLIT] = np.random.normal(loc=0, scale=0.5, size=(embedding_size))  # np.random.normal(loc=0., scale=0.5, size=(embedding_size))

    vocab_size = 0
    for w, c in wordcnt:
        if vocab_size > vocab_size_limit:
            break
        if c >= least_wf:
            if word2embedding.get(w) is not None or word2embedding.get(word2lemma[w]) is not None:
                word2idx[w] = len(word2idx)
                vocab_size += 1
    vocab_embedding = np.zeros((len(word2idx), embedding_size), dtype=np.float32)
    for w, idx in word2idx.items():
        if word2embedding.get(w) is not None:
            vocab_embedding[idx] = word2embedding[w]
        else:
            vocab_embedding[idx] = word2embedding[word2lemma[w]]
    np.save(os.path.join(out_path, "wordembedding.dat"), vocab_embedding)
    ujson.dump(word2idx, open(os.path.join(out_path, "word2idx.json"), "w", encoding='utf-8'))
    ujson.dump(ner2idx, open(os.path.join(out_path, "ner2idx.json"), "w", encoding='utf-8'))
    ujson.dump(pos2idx, open(os.path.join(out_path, "pos2idx.json"), "w", encoding='utf-8'))
    ujson.dump(wordcnt, open(os.path.join(out_path, "wordcnt.json"), "w", encoding='utf-8'))
    ujson.dump(word2lemma, open(os.path.join(out_path, "word2lemma.json"), "w", encoding='utf-8'))
    

def make_race_vocabulary(parse_file_names, out_path, word_vec_file, vocab_size_limit=45000, least_wf=0, do_lower=True, embedding_size=300):
    from nltk import tokenize
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    word2idx = {
        CONSTANTS.PAD: CONSTANTS.PAD_ID,
        CONSTANTS.UNK: CONSTANTS.UNK_ID,
        CONSTANTS.START: CONSTANTS.START_ID,
        CONSTANTS.END: CONSTANTS.END_ID,
        CONSTANTS.SPLIT: CONSTANTS.SPLIT_ID
    }

    wordcnt = {}

    lines = []

    word2lemma = {}
    for pn in parse_file_names:
        lines.extend(open(pn, encoding='utf-8').readlines())

    for line in tqdm(lines):
        d = ujson.loads(line)
        question = tokenize.word_tokenize(d['tgt'])
        text = tokenize.word_tokenize(d['article'])
        answer = tokenize.word_tokenize(d['answer'])
        
        for idx, word in enumerate(text):
                wordcnt[word.lower()] = wordcnt.get(word.lower(), 0) + 1
                word2lemma[word.lower()] =  lemmatizer.lemmatize(word.lower())

        for idx, word in enumerate(question[:-1]):
            wordcnt[word.lower()] = wordcnt.get(word.lower(), 0) + 1
            word2lemma[word.lower()] =lemmatizer.lemmatize(word.lower())
        
        for idx, word in enumerate(answer[:-1]):
            wordcnt[word.lower()] = wordcnt.get(word.lower(), 0) + 1
            word2lemma[word.lower()] =lemmatizer.lemmatize(word.lower())
        
    wordcnt = sorted(wordcnt.items(), key=lambda x: x[1], reverse=True)

    # use fix seed for random unknown word vector generation.
    word2embedding = {}
    np.random.seed(42)
    embedding_size = 300
    lines = open(word_vec_file, "r", encoding='utf-8').readlines()
    for line in tqdm(lines):
        word_vec = line.strip().split()
        word = word_vec[0].lower()
        vec = np.array(word_vec[-embedding_size:], dtype=np.float32)
        if word not in word2embedding:
            word2embedding[word] = vec

    if CONSTANTS.UNK not in word2embedding:
        word2embedding[CONSTANTS.UNK] = np.random.normal(loc=0, scale=0.5, size=(embedding_size))
    if CONSTANTS.START not in word2embedding:
        word2embedding[CONSTANTS.START] = np.random.normal(loc=0, scale=0.5, size=(embedding_size))
    if CONSTANTS.END not in word2embedding:
        word2embedding[CONSTANTS.END] = np.zeros(embedding_size)  # np.random.normal(loc=0., scale=0.5, size=(embedding_size))
    if CONSTANTS.PAD not in word2embedding:
        word2embedding[CONSTANTS.PAD] = np.zeros(embedding_size)  # np.random.normal(loc=0., scale=0.5, size=(embedding_size))
    if CONSTANTS.SPLIT not in word2embedding:
        word2embedding[CONSTANTS.SPLIT] = np.random.normal(loc=0, scale=0.5, size=(embedding_size))  # np.random.normal(loc=0., scale=0.5, size=(embedding_size))

    vocab_size = 0
    for w, c in wordcnt:
        if vocab_size > vocab_size_limit:
            break
        if c >= least_wf:
            if word2embedding.get(w) is not None or word2embedding.get(word2lemma[w]) is not None:
                word2idx[w] = len(word2idx)
                vocab_size += 1
    vocab_embedding = np.zeros((len(word2idx), embedding_size), dtype=np.float32)
    for w, idx in word2idx.items():
        if word2embedding.get(w) is not None:
            vocab_embedding[idx] = word2embedding[w]
        else:
            vocab_embedding[idx] = word2embedding[word2lemma[w]]
    np.save(os.path.join(out_path, "wordembedding.dat"), vocab_embedding)
    ujson.dump(word2idx, open(os.path.join(out_path, "word2idx.json"), "w", encoding='utf-8'))
    ujson.dump(wordcnt, open(os.path.join(out_path, "wordcnt.json"), "w", encoding='utf-8'))
    ujson.dump(word2lemma, open(os.path.join(out_path, "word2lemma.json"), "w", encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        help='In the debug mode, only perform actions for limited data items.',
        action='store_true',
        default=False
    )

    parser.add_argument(
        '--train_file',
        help='parsed graph file.',
        required=True,
        type=str
    )

    parser.add_argument(
        '--dev_file',
        help='parsed graph file.',
        required=False,
        type=str
    )

    parser.add_argument(
        '--test_file',
        help='parsed graph file.',
        required=False,
        type=str
    )

    parser.add_argument(
        '--outpath',
        help='path for putting all the dict files.',
        required=True,
        type=str
    )

    parser.add_argument(
        "--do_lower",
        help='use lower form.',
        type=bool,
        default=True
    )

    parser.add_argument(
        "--vocab_size",
        help='how many words kept in the voabulary.',
        type=int,
        default=45000
    )

    parser.add_argument(
        "--least_wf",
        help='the least word frequency.',
        type=int,
        default=3
    )

    parser.add_argument(
        "--word_vec_file",
        help='pretrained word vector file.',
        type=str,
        default="../../data/glove.840B.300d.txt"
    )

    parser.add_argument(
        "--word_lemma_only",
        help='only make the word and lemma vocabulary.',
        action="store_true",
        default=False
    )

    args = parser.parse_args()
    file_list = [_ for _ in [args.train_file, args.dev_file, args.test_file] if _ is not None]
    if args.word_lemma_only:
        make_race_vocabulary(file_list, args.outpath, args.word_vec_file)
    else:
        make_vocab(file_list, args.outpath, args.word_vec_file)


if __name__ == "__main__":
    main()

# python make_vocab.py --train_file ../../../data/squad/p.train.jsonl --dev_file ../../../data/squad/p.dev.jsonl --outpath ../../../data/squad/dicts 
