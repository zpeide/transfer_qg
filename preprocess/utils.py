import pandas
import argparse
import os
import re
from tqdm import tqdm
# import stanza
import spacy
import ujson


def clean_text(text):
    text = text.strip()
    text = text.replace('\n', ' ')
    # remove image notations
    text = re.sub(r'\(([\.\w\d\-/]*(png|jpg|jpeg))+\)', ' ', text).strip()
    text = re.sub('---+', ' ', text).strip()
    # remove <a href="">
    text = re.sub(r'<a href=\".*\"\s*/?>', ' ', text)
    # remove html notations
    text = re.sub(r'</?[\w\s\d]+>', ' ', text)
    text = re.sub(r'~~+', ' ', text)
    # remove links
    #URL_REGULAR = re.compile( r"\((https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(/[-a-zA-Z0-9()@:%_\+.~#?&//=]+)\)")
    URL_REGULAR = re.compile( r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(/[-a-zA-Z0-9()@:%_\+.~#?&//=]+)")
    text = URL_REGULAR.sub(" ", text)
    return text.strip()

def tokenize_stanza(text, pipeline):
    text = text.strip()
    if len(text) == 0:
        return '', '', ''
    doc = pipeline(text)
    words, pos, ner = [], [], []
    for sent in doc.sentences:
        for token in sent.tokens:
            for word in token.words:
                text = word.text.strip()
                if len(text.split()) > 1:
                    text = ''.join(text.split())
                words.append(text)
                pos.append(word.upos)
                ner.append(token.ner)
 
    featured_src = ' '.join([u"￨".join([w, p, n]) for w, p, n in zip(words, pos, ner)])
    src = ' '.join(words)
    pos = ' '.join(pos)
    ner = ' '.join(ner)
    return src, pos, ner, featured_src


import re
import numpy as np
from bisect import bisect_left
import torch

class CONSTANTS:
    PAD = '[pad]'
    UNK = '[UNK]'
    START = '<s>'
    END = '</s>'
    SPLIT = '<split>'

    PAD_ID = 0
    UNK_ID = 1
    START_ID = 2
    END_ID = 3
    SPLIT_ID = 4


GENERAL_WD = ['is', 'are', 'am', 'was', 'were', 'have', 'has', 'had', 'can', 'could',
              'shall', 'will', 'should', 'would', 'do', 'does', 'did', 'may', 'might', 'must', 'ought', 'need', 'dare']
GENERAL_WD += [x.capitalize() for x in GENERAL_WD]
GENERAL_WD = re.compile(' |'.join(GENERAL_WD))


def judge_question_type(q: str, G=GENERAL_WD) -> int:
    if q.find(' or ') >= 0:
        return 2
    elif G.match(q):
        return 1
    else:
        return 0

class WindowMean:
    def __init__(self, window_size=50):
        self.array = []
        self.sum = 0
        self.window_size = window_size

    def update(self, x):
        self.array.append(x)
        self.sum += x
        if len(self.array) > self.window_size:
            self.sum -= self.array.pop(0)
        return self.sum / len(self.array)

