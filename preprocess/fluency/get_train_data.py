import json
import random

train_file = '/ssd/data/Google_Natural_Question/v1.0-simplified_simplified-nq-train.jsonl'
f = open(train_file)
lines = f.readlines()
f.close()

questions = [json.loads(_)['question_text'].strip().lower() + ' ?' for _ in lines]

quora_file = "./quora.ppl.train.jsonl"
f = open(quora_file)
quora = [json.loads(_) for _ in f]
for q in quora:
    questions.append(q['question1'].lower())
    if q['is_duplicate'] == 0:
        questions.append(q['question2'].lower())
f.close()

dev_file = '/ssd/data/Google_Natural_Question/v1.0-simplified_nq-dev-all.jsonl'
f = open(dev_file)
lines = f.readlines()
f.close()

dev_questions = [json.loads(_)['question_text'].strip().lower() + ' ?' for _ in lines]

f = open("tgt.train.txt", "w")
for q in questions:
    f.write(q)
    f.write('\n')
f.close()

f = open("tgt.dev.txt", "w")
for q in dev_questions:
    f.write(q)
    f.write('\n')
f.close()