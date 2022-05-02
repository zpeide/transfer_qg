#!/usr/bin/env python
__author__ = 'xinya'

from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from collections import defaultdict
from argparse import ArgumentParser
import codecs

from pdb import set_trace
import sys
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')
class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self):
        output = []
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            # (Cider(), "CIDEr")
        ]

        bleus = []

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)

            # set_trace()

            if type(method) == list:
                bleus = scores
                np.savez("bleus.npz", bleus=bleus)
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.5f"%(m, sc))
                    output.append(sc)
            else:
                print( "%s: %0.5f"%(method, score))
                output.append(score)
        return output


def eval(out_file, src_file, tgt_file, isDIn = False, num_pairs = 500):
    """
        Given a filename, calculate the metric scores for that prediction file

        isDin: boolean value to check whether input file is DirectIn.txt
    """

    pairs = []
    infile = codecs.open(src_file, 'r', encoding='utf-8').readlines() 
    infile = [_.strip() for _ in infile]

    tgts = codecs.open(tgt_file, "r" , encoding='utf-8').readlines() 
    tgts = [_.strip() for _ in tgts]
    output = []
    output = codecs.open(out_file, 'r', encoding='utf-8').readlines() 
    output = [_.strip() for _ in output]
    aclen = min(len(infile), len(tgts), len(output))
    output = output[:aclen]
    for cnt, (inline, tgt) in enumerate(zip(infile[:aclen], tgts[:aclen])):
        pair = {}
        pair['tokenized_sentence'] = inline
        pair['tokenized_question'] = tgt
        pairs.append(pair)

 
    print(len(pairs), len(output))
    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]


    ## eval
    from eval import QGEvalCap
    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])

    #set_trace()

    for pair in pairs[:]:
        key = pair['tokenized_sentence']
        res[key] = [pair['prediction']]

        ## gts 
        gts[key].append(pair['tokenized_question'])

    #set_trace()

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-out", "--out_file", dest="out_file", default="./output/pred.txt", help="output file to compare")
    parser.add_argument("-src", "--src_file", dest="src_file", default="../data/processed/src-test.txt", help="src file")
    parser.add_argument("-tgt", "--tgt_file", dest="tgt_file", default="../data/processed/tgt-test.txt", help="target file")
    args = parser.parse_args()

    print("scores: \n")
    eval(args.out_file, args.src_file, args.tgt_file)


