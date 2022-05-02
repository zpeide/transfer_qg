import torch
import time
from transformers import *
import numpy as np
import os, json

from collections import defaultdict
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS
from sklearn import cluster, mixture
# t-sne
from sklearn import manifold
import sklearn
from sklearn.metrics import confusion_matrix
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import random

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


random.seed(21)
np.random.seed(21)

base_path_new = "../../data"

file_paths_new = {
                'NQ': os.path.join(base_path_new, 'nq/train.jsonl'),
                'RACE': os.path.join(base_path_new, 'RACE/target.jsonl'),
                'SciQ':  os.path.join(base_path_new, 'SciQ/target.jsonl'),
            }
race_ds = [json.loads(_) for _ in open(file_paths_new['RACE'])] 
sciq_ds = [json.loads(_) for _ in open(file_paths_new['SciQ'])]
nq_ds = [json.loads(_) for _ in open(file_paths_new['NQ'])]


def make_bert_train_ds(sorted_nq_idx, targets, file_path, num_nq_for_train = 4000):
    idx = 0
    f1 = open(file_path, "w")  
    selected_items = []
    selected_passages = set()
    while len(selected_items) < num_nq_for_train:
        passage = ' '.join(nq_ds[sorted_nq_idx[idx]]['text']).lower()
        idx += 1
        if passage in selected_passages:
            continue
        selected_items.append(
            {'src': passage, 'is_tgt_domain': 0})
        selected_passages.add(passage)
    for d in selected_items:
        f1.write(json.dumps(d))
        f1.write('\n')

    for d in targets:
        f1.write(json.dumps(d))
        f1.write('\n')
    
    f1.close()

targets = []
for d in race_ds:
    targets.append(
        {
            'src': ' '.join(d['text']).lower(),
            'is_tgt_domain': 1
        }
    )

nq_race = torch.load("results_NQ-RACE.pt")
method_used = 'avg'
nq_nq_center_l2_distance = nq_race[method_used]['gmm']['nq_center_l2_distance'][:89453]
sorted_nq_idx = np.argsort(nq_nq_center_l2_distance)

make_bert_train_ds(sorted_nq_idx, targets[:1600], "rc.train.jsonl")

random_nq_idx = np.arange(len(nq_ds))
np.random.shuffle(random_nq_idx)
make_bert_train_ds(random_nq_idx, targets[:1600], "rc.train.random.jsonl")

idx_for_dev = list()
selected_idx = set(sorted_nq_idx[:5000])
for idx in random_nq_idx[5000:]:
    if idx not in selected_idx:
        idx_for_dev.append(idx)

make_bert_train_ds(idx_for_dev, targets[1600:], "rc.dev.jsonl", num_nq_for_train=600)


targets = []
for d in sciq_ds:
    targets.append(
        {
            'src': ' '.join(d['text']).lower(),
            'is_tgt_domain': 1
        }
    )

nq_sciq = torch.load("results_NQ-SciQ.pt")
method_used = 'avg'
nq_nq_center_l2_distance = nq_sciq[method_used]['gmm']['nq_center_l2_distance'][:89453]
sorted_nq_idx = np.argsort(nq_nq_center_l2_distance)

make_bert_train_ds(sorted_nq_idx, targets, "sq.train.jsonl")

random_nq_idx = np.arange(len(nq_ds))
np.random.shuffle(random_nq_idx)
make_bert_train_ds(random_nq_idx, targets[:1600], "sq.train.random.jsonl")

idx_for_dev = list()
selected_idx = set(sorted_nq_idx[:5000])
for idx in random_nq_idx[5000:]:
    if idx not in selected_idx:
        idx_for_dev.append(idx)

make_bert_train_ds(idx_for_dev, targets[1600:], "sq.dev.jsonl", num_nq_for_train=600)

