## This is a script for fluency service. Because model trained here cannot apply to QG. because of the version issue. 
from argparse import ArgumentParser
import socket
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import AutoModelWithLMHead
import json

parser = ArgumentParser()
parser.add_argument("--fluency_model_path", default=None, type=str, required=True,
                    help="path to the fluency model.")
# parser.add_argument("--fluency_model_path", default=None, type=str, required=True,
#                     help="path to the fluency model.")
parser.add_argument("--port", default=8800, type=int, help="port for the server.")

args = parser.parse_args()

device=0

fluency_model = AutoModelWithLMHead.from_pretrained(args.fluency_model_path).to(device).eval()
bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


host = '127.0.0.1'        # Symbolic name meaning all available interfaces
port = args.port 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(1)
while True:
    conn, addr = s.accept()
    print('Connected by', addr)
    while True:
        data = (conn.recv(1024))
        if not data: break
        try:
            en = data.decode()
            sent = bert_tokenizer(en, return_tensors='pt').to(device)
            sent_out = fluency_model(**sent, labels=sent.input_ids)
            fluency = sent_out.loss.cpu().tolist()
            print(f"\nReceived:\t{en},\tFluency:\t{fluency}")
            
        except UnicodeDecodeError:
            print("Unicode Decode error")
            y="UnicodeDecodeError" # a very high number
        conn.sendall((json.dumps({'fluency': fluency}) + "\n").encode('utf8'))
    conn.close()
