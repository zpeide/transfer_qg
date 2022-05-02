from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import json
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import tqdm

from s2s_ft.modeling_uda import BertForSequenceToSequence
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import \
    RobertaConfig, BertConfig, \
    BertTokenizer, RobertaTokenizer, \
    XLMRobertaConfig, XLMRobertaTokenizer
from s2s_ft.configuration_unilm import UnilmConfig
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.configuration_minilm import MinilmConfig
from s2s_ft.tokenization_minilm import MinilmTokenizer

from s2s_ft import utils
from s2s_ft.config import BertForSeq2SeqConfig

from s2s_ft.modeling_decoding import BertForSeq2SeqDecoder, BertConfig as BertDecoderConfig
from s2s_ft import s2s_loader as seq2seq_loader
import math

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
    'minilm': (MinilmConfig, MinilmTokenizer),
    'roberta': (RobertaConfig, RobertaTokenizer),
    'xlm-roberta': (XLMRobertaConfig, XLMRobertaTokenizer),
    'unilm': (UnilmConfig, UnilmTokenizer),
}



def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

def get_model_and_tokenizer(args):
    config_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    config = BertForSeq2SeqConfig.from_exist_config(
        config=model_config, label_smoothing=args.label_smoothing,
        max_position_embeddings=args.max_source_seq_length + args.max_target_seq_length)

    logger.info("Model config for seq2seq: %s", str(config))

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None)

    model = BertForSequenceToSequence.from_pretrained(
        args.model_name_or_path, config=config, model_type=args.model_type,
        reuse_position_embedding=True,
        cache_dir=args.cache_dir if args.cache_dir else None)

    decoder_model, decoder_tokenize = None, None
    decoder_requirement = {}
    if args.local_rank in [-1, 0]:
        config_file = os.path.join(args.recover_path, "config.json")
        decoder_config = BertDecoderConfig.from_json_file(config_file)
        vocab = tokenizer.vocab
        args.max_seq_length = args.max_source_seq_length + args.max_target_seq_length
        decoder_tokenize = seq2seq_loader.Preprocess4Seq2seqDecoder(
            list(vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
        max_tgt_length=args.max_target_seq_length, pos_shift=args.pos_shift,
        source_type_id=config.source_type_id, target_type_id=config.target_type_id, 
        cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token, pad_token=tokenizer.pad_token)
        
        mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.mask_token, tokenizer.sep_token, tokenizer.sep_token])
        forbid_ignore_set = None
        if args.forbid_ignore_word:
            w_list = []
            for w in args.forbid_ignore_word.split('|'):
                if w.startswith('[') and w.endswith(']'):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
        
        decoder_model = BertForSeq2SeqDecoder.from_pretrained(
            args.recover_path, config=decoder_config, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
            length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
            forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
            ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
            max_position_embeddings=args.max_seq_length, pos_shift=args.pos_shift, 
        ).eval()
        # decoder_model.to(args.pred_device_num)
        decoder_requirement = {
            'config':decoder_config, 
            'mask_word_id': mask_word_id, 
            'search_beam_size': args.beam_size,
            'length_penalty': args.length_penalty, 
            'eos_id': eos_word_ids, 
            'sos_id': sos_word_id,
            'forbid_duplicate_ngrams': args.forbid_duplicate_ngrams, 
            'forbid_ignore_set': forbid_ignore_set,
            'ngram_size': args.ngram_size, 
            'min_len': args.min_len, 
            'mode': args.mode,
            'max_position_embeddings': args.max_seq_length, 
            'pos_shift': args.pos_shift, 
        }

    return model, tokenizer, decoder_model, decoder_tokenize, decoder_requirement


def prepare_for_training(args, model, checkpoint_state_dict):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if checkpoint_state_dict:
        model.load_state_dict(checkpoint_state_dict['model'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(0)

    return model, optimizer


def make_prediction(args, tokenizer, pred_model, pred_features, decoder_tokenize):
    pred_model.to(0)
    if args.n_gpu > 1:
        pred_model = torch.nn.DataParallel(pred_model)
        
    output_lines = [""] * len(pred_features)
    output_line_ids = [0] * len(pred_features)
    output_scores = [0] * len(pred_features)
    score_trace_list = [None] * len(pred_features)
    total_batch = math.ceil(len(pred_features) / args.pred_batch_size)
    next_i = 0
    device = args.pred_device_num
    with tqdm.tqdm(total=total_batch) as pbar:
        batch_count = 0
        first_batch = True
        while next_i < len(pred_features):
            _chunk = pred_features[next_i:next_i + args.pred_batch_size]
            buf_id = [x[0] for x in _chunk]
            buf = [x[1] for x in _chunk]
            next_i += args.pred_batch_size
            batch_count += 1
            max_a_len = max([len(x) for x in buf])
            instances = []
            for instance in [(x, max_a_len) for x in buf]:
                instances.append(decoder_tokenize(instance))
            with torch.no_grad():
                batch = seq2seq_loader.batch_list_to_batch_tensors(
                    instances)
                input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                traces = pred_model(input_ids, token_type_ids,
                                position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                if args.beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                else:
                    output_ids = traces.tolist()
                for i in range(len(buf)):
                    w_ids = output_ids[i]
                    out_ids = []
                    for t in w_ids:
                        if t in (tokenizer.sep_token_id, tokenizer.pad_token_id):
                            break
                        out_ids.append(t)
                    output_line_ids[buf_id[i]] = out_ids
                    output_scores[buf_id[i]] = traces['pred_seq_scores'][i][:len(out_ids)]
                    for _tok_idx in range(1, len(output_scores[buf_id[i]])):
                        # each seq score is accumulated by the sequence, so token score should minus the previous sequence score.
                        output_scores[buf_id[i]][_tok_idx] = output_scores[buf_id[i]][_tok_idx] - output_scores[buf_id[i]][_tok_idx - 1]

                    output_tokens = tokenizer.convert_ids_to_tokens(out_ids)

                    if args.model_type == "roberta":
                        output_sequence = tokenizer.convert_tokens_to_string(output_tokens)
                    else:
                        output_sequence = ' '.join(detokenize(output_tokens))
                    if '\n' in output_sequence:
                        output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                    output_lines[buf_id[i]] = output_sequence
                    if first_batch or batch_count % 50 == 0:
                        logger.info("{} = {}".format(buf_id[i], output_sequence))
                    if args.need_score_traces:
                        score_trace_list[buf_id[i]] = {
                            'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}
            pbar.update(1)
            first_batch = False
    module = pred_model.module if hasattr(pred_model, "module") else pred_model
    module = module.cpu()
    pred_model = None
    return output_line_ids, output_lines, output_scores


def train(args, selected_features, model, tokenizer, decoder_model, decoder_tokenize, unlabeled_features, 
            unlabeled_pred_features, decoder_requirement):
    """ Train the model """
    if args.local_rank in [-1, 0] and args.log_dir:
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        tb_writer = None
    
    checkpoint_state_dict = {}
    model_recover_checkpoint = os.path.join(args.recover_path, "pytorch_model.bin")
    logger.info(" ** Recover model checkpoint in %s ** ", model_recover_checkpoint)
    model_state_dict = torch.load(model_recover_checkpoint, map_location='cpu')
    checkpoint_state_dict['model'] = model_state_dict

    model, optimizer = prepare_for_training(args, model, checkpoint_state_dict)

    if args.n_gpu == 0 or args.no_cuda:
        per_node_train_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps
    else:
        per_node_train_batch_size = args.per_gpu_train_batch_size * args.n_gpu * args.gradient_accumulation_steps
        
    train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    num_training_steps = args.num_training_epochs * (len(selected_features) + len(unlabeled_features)) // train_batch_size
    args.train_batch_size = train_batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps, last_epoch=-1)

    args.pred_batch_size = args.pred_per_gpu_batch_size * args.n_gpu if args.n_gpu >= 1 else args.pred_per_gpu_batch_size

    # Train!
    logger.info("  ***** Running training *****  *")
    logger.info("  Num examples = %d", (len(selected_features) + len(unlabeled_features)))
    logger.info("  Num Epochs = %.2f", args.num_training_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Batch size per node = %d", per_node_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    # connect to fluency model.
    import socket
    host = '127.0.0.1'        # Symbolic name meaning all available interfaces
    port = args.fluency_port 
    fluency_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fluency_socket.connect((host, port))

    global_step =  0
    model.train()
    model.zero_grad()
    save_path = args.recover_path
    fluency_threshold = args.fluency_threshold  #13.5
    perplexity_threshold = args.perplexity_threshold
    selected_fluency_scores = {}
    selected_perplexity_scores = {}
    selected_questions = {}
    selected_question_ids = {}
    selected_sequence_socre = {}
    for epoch in tqdm.trange(1, args.num_training_epochs + 1, desc="Epoch", disable=args.local_rank not in [-1, 0]):
        # make soft labels for unlabeled samples
        pred_line_ids, pred_lines, pred_scores = make_prediction(args, tokenizer, decoder_model, unlabeled_pred_features, decoder_tokenize)

        # we calculate fluency score no matter use it for filtering or not.
        fluency_score = []
        for p in tqdm.tqdm(pred_lines, desc='fluency'):
            fluency_socket.sendall(p.encode('utf8'))
            score = fluency_socket.recv(1024)
            score = json.loads(score.decode())
            fluency_score.append(score['fluency'])
        
        perplexity_score = [np.exp(-np.mean(_)) for _ in pred_scores]
        
        # log all generation
        gf = open( os.path.join(args.output_dir, f"gen_epoch_{epoch}.jsonl"), "w")
        # log selected generation
        sf = open( os.path.join(args.output_dir, f"selected_gen_epoch_{epoch}.jsonl"), "w")
        
        selected_unlabeled_items = []
        fluency_selected_ids = set()
        perplexity_selected_ids = set()
        for i in range(len(unlabeled_pred_features)):
            unlabeled_features[i]['seq_score'] = pred_scores[i]
            unlabeled_features[i]['target_ids'] = pred_line_ids[i]            
            unlabeled_features[i]['perplexity'] = perplexity_score[i]
            unlabeled_features[i]['fluency'] = fluency_score[i]
            gf.write(json.dumps({'gen': pred_lines[i], 'score': pred_scores[i], 'fluency': fluency_score[i], 'perplexity': perplexity_score[i]}))
            gf.write('\n')            
            if args.use_fluency:
                if fluency_score[i] > fluency_threshold:
                    if args.only_better:
                        if i in selected_fluency_scores:
                            if fluency_score[i] < selected_fluency_scores[i]:
                                unlabeled_features[i]['target_ids'] = selected_question_ids[i]
                                unlabeled_features[i]['fluency'] = selected_fluency_scores[i]
                                unlabeled_features[i]['seq_score'] = selected_sequence_socre[i]
                                pred_lines[i] = selected_questions[i]
                    fluency_selected_ids.add(i)
            if args.use_perplexity:
                if perplexity_score[i] <= perplexity_threshold:
                    if args.only_better:
                        if i in selected_perplexity_scores:
                            if perplexity_score[i] > selected_perplexity_scores[i]:
                                unlabeled_features[i]['target_ids'] = selected_question_ids[i]
                                unlabeled_features[i]['perplexity'] = selected_perplexity_scores[i]
                                unlabeled_features[i]['seq_score'] = selected_sequence_socre[i]
                                pred_lines[i] = selected_questions[i]
                    perplexity_selected_ids.add(i)

        if args.use_fluency and args.use_perplexity:            
            if args.join_selection:
                selected_ids = fluency_selected_ids.union(perplexity_selected_ids)
            else:
                selected_ids = fluency_selected_ids.intersection(perplexity_selected_ids)
        else:
            if args.use_fluency:
                selected_ids = fluency_selected_ids
            elif args.use_perplexity:
                selected_ids = perplexity_selected_ids
            else:
                selected_ids = np.arange(len(unlabeled_pred_features))
        for i in selected_ids:            
            selected_unlabeled_items.append(unlabeled_features[i])
            selected_fluency_scores[i] = unlabeled_features[i]['fluency']
            selected_perplexity_scores[i] = unlabeled_features[i]['perplexity']            
            selected_question_ids[i] = unlabeled_features[i]['target_ids']
            selected_sequence_socre[i] = unlabeled_features[i]['seq_score']
            selected_questions[i] = pred_lines[i]
            sf.write(json.dumps({ 'gen': pred_lines[i], 'idx': int(i), 
                        'fluency': float(selected_fluency_scores[i]), 'perplexity':float(selected_perplexity_scores[i])}))
            sf.write('\n')
        gf.close()
        sf.close()
        # only select fluency perplexity < \epsilon

        training_features = selected_features + selected_unlabeled_items # unlabeled_features
        random.shuffle(training_features)
        # change every epoch
        train_dataset = utils.Seq2seqDatasetForBert(
            features=training_features, max_source_len=args.max_source_seq_length,
            max_target_len=args.max_target_seq_length, vocab_size=tokenizer.vocab_size,
            cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id,
            mask_id=tokenizer.mask_token_id, random_prob=args.random_prob, keep_prob=args.keep_prob,
            num_training_instances=len(training_features), seq_score=True, offset=0,
        )

        # The training features are shuffled
        train_sampler = SequentialSampler(train_dataset) \
            if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=per_node_train_batch_size // args.gradient_accumulation_steps,
            collate_fn=utils.batch_list_to_batch_tensors)

        train_iterator = tqdm.tqdm(
            train_dataloader, 
            desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=args.local_rank not in [-1, 0])

        tr_loss, logging_loss = 0.0, 0.0

        for step, batch in enumerate(train_iterator):
            if args.no_normalization:
                # No score normalization.
                batch[5] = None
            inputs = {'source_ids': batch[0],
                      'target_ids': batch[1],
                      'pseudo_ids': batch[2],
                      'num_source_tokens': batch[3],
                      'num_target_tokens': batch[4], 
                      'scores': batch[5],
                    }

            loss = model(**inputs)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            logging_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("")
                    logger.info(" Step [%d ~ %d]: %.2f", global_step - args.logging_steps, global_step, logging_loss)
                    logging_loss = 0.0

                if args.local_rank in [-1, 0] and args.save_steps > 0 and \
                        (global_step % args.save_steps == 0 or global_step == num_training_steps):

                    save_path = os.path.join(args.output_dir, "ckpt-%d" % global_step)
                    os.makedirs(save_path, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(save_path)
                    
                    logger.info("Saving model checkpoint %d into %s", global_step, save_path)
        module = model.module if hasattr(model, "module") else model
        module = module.cpu()
        decoder_model = BertForSeq2SeqDecoder.from_pretrained(
            "", config=decoder_requirement['config'], mask_word_id=decoder_requirement['mask_word_id'], 
            search_beam_size=args.beam_size,
            length_penalty=args.length_penalty, eos_id=decoder_requirement['eos_id'], sos_id=decoder_requirement['sos_id'],
            forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=decoder_requirement['forbid_ignore_set'],
            ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
            max_position_embeddings=args.max_seq_length, pos_shift=args.pos_shift, state_dict=module.state_dict()
        ).eval()

        if args.local_rank in [-1, 0] :
            save_path = os.path.join(args.output_dir, "ckpt-epoch-%d" % epoch)
            os.makedirs(save_path, exist_ok=True)
            model_to_save = module
            model_to_save.save_pretrained(save_path)
            
            logger.info("Saving model checkpoint of epoch %d into %s", epoch, save_path)

        model = torch.nn.DataParallel(module)
        model = model.to(0).train()

    if args.local_rank in [-1, 0] and tb_writer:
        tb_writer.close()



def prepare(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="Training data (json format) for training. Keys: source")
    parser.add_argument("--selected_file", default=None, type=str, required=True,
                        help="selected out-domain source data (json format) for training. Keys: source and target")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    parser.add_argument("--recover_path", default=None, type=str, required=True,
                        help="Path to pre-trained model on source dataset.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_source_seq_length", default=464, type=int,
                        help="The maximum total source sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_target_seq_length", default=48, type=int,
                        help="The maximum total target sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--cached_train_features_file", default=None, type=str,
                        help="Cached training features file")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_training_epochs", default=10, type=int,
                        help="set total number of training epochs to perform (--num_training_steps has higher priority)")
    parser.add_argument("--num_warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--random_prob", default=0.1, type=float,
                        help="prob to random replace a masked token")
    parser.add_argument("--keep_prob", default=0.1, type=float,
                        help="prob to keep no change for a masked token")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument("--pred_device_num", default=0, type=int,
                        help="gpu device used for inferencing.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")

    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument("--pred_per_gpu_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for prediction.")
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")

    parser.add_argument("--fluency_port", default=8800, type=int,
                        help="port for the fluency server.")
                
    # parser.add_argument("--no_fluency", default=False, action='store_true',
    #                     help="don't use fluency filter.")
    parser.add_argument("--use_fluency", default=False, action='store_true',
                        help="use fluency filter when set.")

    parser.add_argument("--use_perplexity", default=False, action='store_true',
                        help="use perplexity as confidence filter when set.")

    parser.add_argument("--fluency_threshold", default=12, type=float,
                        help="the fluency threshold used to filter the generated questions.")

    parser.add_argument("--perplexity_threshold", default=6, type=float,
                        help="the perplexity threshold used to filter the generated questions.")

    parser.add_argument("--join_selection", default=False, action='store_true',
                        help="combine all items selected by different filters. else use the cross set.")

    parser.add_argument("--only_better", default=False, action='store_true',
                        help="when get selected, compare to the previous state, choose the better one.")
    parser.add_argument("--only_in_domain", default=False, action='store_true',
                        help="only use unlabeled data.")

    parser.add_argument("--no_normalization", default=False, action='store_true',
                        help="No normalization with prediction scores.")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    prepare(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab
    # Load pretrained model and tokenizer
    model, tokenizer, decoder_model, decoder_tokenize, decoder_requirement = get_model_and_tokenizer(args)

    if args.cached_train_features_file is None:
        args.cached_train_features_file = os.path.join(args.output_dir, "uda_cached_features_for_training.pt")
        args.cached_selected_features_file = os.path.join(args.output_dir, "selected_cached_features_for_training.pt")
    unlabeled_features = utils.load_and_cache_examples(
        example_file=args.train_file, tokenizer=tokenizer, local_rank=args.local_rank,
        cached_features_file=args.cached_train_features_file, shuffle=False,
    )

    if args.only_in_domain:
        selected_features = []
    else:
        selected_features = utils.load_and_cache_examples(
            example_file=args.selected_file, tokenizer=tokenizer, local_rank=args.local_rank,
            cached_features_file=args.cached_selected_features_file, shuffle=False,
        )
    
        
    # transform training features for prediction.
    unlabeled_pred_features = []
    for line in unlabeled_features:
        unlabeled_pred_features.append(tokenizer.convert_ids_to_tokens(line["source_ids"])[:args.max_source_seq_length-2])

    unlabeled_pred_features = sorted(list(enumerate(unlabeled_pred_features)),
                            key=lambda x: -len(x[1]))
    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    # selected_features + unlabeled_features == examples for UDA.
    train(args, selected_features, model, tokenizer, decoder_model, decoder_tokenize, 
            unlabeled_features, unlabeled_pred_features, decoder_requirement)


if __name__ == "__main__":
    main()
