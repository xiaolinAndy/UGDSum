# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import math
import json
import logging
import pdb
import random
import jieba
import re
from argparse import ArgumentParser
from pprint import pformat
from itertools import chain
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel, BertTokenizer, GPT2Tokenizer

import nltk
from nltk.corpus import stopwords
from string import punctuation

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]"]


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


# def build_input_from_segments(history, resposne, label, tokenizer, dataset, with_eos=True):
#     """ Build a sequence of input from 3 segments: persona, history and last reply """
#
#     if dataset == 'samsum':
#         bos, eos = tokenizer.convert_tokens_to_ids(['<|endoftext|>', '<|endoftext|>'])
#     else:
#         bos, eos, pad, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
#     resposne = resposne[:510]
#     total_length = len(resposne)
#     index = len(history) - 1
#     while index >= 0:
#         total_length += len(history[index]) + 1
#         if total_length > 509:
#             break
#         index -= 1
#     history = history[index + 1:]
#     sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
#     label = label + ([eos] if with_eos else [])
#     instance = {}
#     instance["input_ids"] = list(chain(*sequence))
#     if dataset == 'CSDS':
#         instance["token_type_ids"] = [bos]
#         for i in range(len(history)):
#             if tokenizer.convert_tokens_to_ids(['客'])[0] == history[i][0]:
#                 instance["token_type_ids"] += [1] * len(history[i])
#             else:
#                 instance["token_type_ids"] += [0] * len(history[i])
#
#         if tokenizer.convert_tokens_to_ids(['客'])[0] == resposne[0]:
#             instance["token_type_ids"] += [1] * len(resposne)
#         else:
#             instance["token_type_ids"] += [0] * len(resposne)
#         if with_eos:
#             instance["token_type_ids"] += [instance["token_type_ids"][-1]]
#
#         #instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1, -1, -1] + sequence[-1][3:]
#         instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1, -1, -1] + label[3:]
#     else:
#         instance["token_type_ids"] = [0] * len(instance["input_ids"])
#
#         instance["lm_labels"] = [-1] * sum(len(s) for s in sequence[:-1])
#         index = 0
#         while index < len(sequence[-1]) and sequence[-1][index] != 1058:
#             instance["lm_labels"].append(-1)
#             index += 1
#         if index != len(sequence[-1]):
#             instance["lm_labels"].append(-1)
#             index += 1
#             #instance["lm_labels"] += sequence[-1][index:]
#             instance["lm_labels"] += label[index:]
#
#     assert len(instance["input_ids"]) == len(instance["token_type_ids"])
#     assert len(instance["input_ids"]) == len(instance["lm_labels"])
#     if len(instance["input_ids"]) > 512:
#         print(instance['input_ids'], history, resposne)
#     assert len(instance["input_ids"]) <= 512
#
#     return instance

def build_input_from_segments(history, resposne, weight, tokenizer, dataset, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """

    if dataset == 'samsum':
        bos, eos = tokenizer.convert_tokens_to_ids(['<|endoftext|>', '<|endoftext|>'])
    else:
        bos, eos, pad, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    resposne = resposne[:510]
    weight = weight[:510]
    total_length = len(resposne)
    index = len(history) - 1
    while index >= 0:
        total_length += len(history[index]) + 1
        if total_length > 509:
            break
        index -= 1
    history = history[index + 1:]
    sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    if dataset == 'CSDS':
        instance["token_type_ids"] = [bos]
        for i in range(len(history)):
            if tokenizer.convert_tokens_to_ids(['客'])[0] == history[i][0]:
                instance["token_type_ids"] += [1] * len(history[i])
            else:
                instance["token_type_ids"] += [0] * len(history[i])

        if tokenizer.convert_tokens_to_ids(['客'])[0] == resposne[0]:
            instance["token_type_ids"] += [1] * len(resposne)
        else:
            instance["token_type_ids"] += [0] * len(resposne)
        if with_eos:
            instance["token_type_ids"] += [instance["token_type_ids"][-1]]

        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1, -1, -1] + sequence[-1][3:]
        instance['weight'] = ([0] * sum(len(s) for s in sequence[:-1])) + [0, 0, 0] + weight[3:] + ([0] if with_eos else [])
        #instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1, -1, -1] + label[3:]
    else:
        instance["token_type_ids"] = [0] * len(instance["input_ids"])

        instance["lm_labels"] = [-1] * sum(len(s) for s in sequence[:-1])
        instance['weight'] = [0] * sum(len(s) for s in sequence[:-1])
        index = 0
        while index < len(sequence[-1]) and sequence[-1][index] != 1058:
            instance["lm_labels"].append(-1)
            instance['weight'].append(0)
            index += 1
        if index != len(sequence[-1]):
            instance["lm_labels"].append(-1)
            instance['weight'].append(0)
            index += 1
            instance["lm_labels"] += sequence[-1][index:]
            instance['weight'] += weight[index:] + ([0] if with_eos else [])
            #instance["lm_labels"] += label[index:]
    try:
        assert len(instance["input_ids"]) == len(instance["token_type_ids"])
        assert len(instance["input_ids"]) == len(instance["lm_labels"])
        assert len(instance["input_ids"]) == len(instance["weight"])
    except:
        print(instance['input_ids'], instance["lm_labels"], instance['weight'])
    if len(instance["input_ids"]) > 512:
        print(instance['input_ids'], history, resposne)
    assert len(instance["input_ids"]) <= 512

    return instance


def test_data(args):
    with open(args.datapath, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
    if isinstance(dataset, dict):
        dataset = dataset["test"]
    return dataset


# def sample_sequence(history, response, label, tokenizer, model, args):
#     output = []
#
#     special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
#
#     instance = build_input_from_segments(history, response, label, tokenizer, args.dataset, with_eos=False)
#     input_ids = torch.tensor(instance["input_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
#     token_type_ids = torch.tensor(instance["token_type_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
#     lm_labels = torch.tensor(instance["lm_labels"], dtype=torch.long, device=args.device).unsqueeze(0)
#
#     if args.dataset == 'CSDS':
#         #lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
#         res = model(input_ids, token_type_ids=token_type_ids)
#         lm_logits = res.logits
#     else:
#         res = model(input_ids, token_type_ids=token_type_ids)
#         lm_logits = res.logits
#     lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
#     lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
#     nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(lm_logits_flat_shifted, lm_labels_flat_shifted)
#     output.append(nll_loss.item())
#
#     return output

def sample_sequence(history, response, weight, tokenizer, model, args, use_idf=False):
    output = []

    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    instance = build_input_from_segments(history, response, weight, tokenizer, args.dataset, with_eos=False)
    input_ids = torch.tensor(instance["input_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
    token_type_ids = torch.tensor(instance["token_type_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
    lm_labels = torch.tensor(instance["lm_labels"], dtype=torch.long, device=args.device).unsqueeze(0)
    weights = torch.tensor(instance["weight"], dtype=torch.float, device=args.device).unsqueeze(0)

    if args.dataset == 'CSDS':
        lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
        #res = model(input_ids, token_type_ids=token_type_ids)
       # lm_logits = res.logits
    else:
        #res = model(input_ids, token_type_ids=token_type_ids)
        lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
        #lm_logits = res.logits
    lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
    lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
    weights = weights[..., 1:].contiguous().view(-1)
    #print(input_ids, lm_labels, weights)
    #nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(lm_logits_flat_shifted, lm_labels_flat_shifted)
    nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')(lm_logits_flat_shifted, lm_labels_flat_shifted)
    if use_idf:
        nll_loss = nll_loss * weights
    valid_len = torch.sum(lm_labels_flat_shifted != -1)
    nll_loss = torch.sum(nll_loss, dim=-1) / valid_len.float()
    output.append(nll_loss.item())

    return output


def main():
    parser = ArgumentParser()
    parser.add_argument('--gpt2', action='store_true', help="use gpt2")
    parser.add_argument("--datapath", type=str, default="", help="Path of the dataset.")
    parser.add_argument("--out_path", type=str, default="", help="Path of response generated.")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=60, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.0,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--dataset", type=str, default="samsum")
    parser.add_argument("--idf_path", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        logging.error("Checkpoint needed!")
        return

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    if args.dataset == 'CSDS':
        tokenizer_class = BertTokenizer
    else:
        tokenizer_class = GPT2Tokenizer
    model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=True)
    model = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    model.eval()

    # def tokenize(obj, words):
    #     if isinstance(obj, str):
    #         tokens = tokenizer.tokenize(obj)
    #         labels = []
    #         ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    #         for i, s in enumerate(tokens):
    #             if s in words:
    #                 labels.append(-1)
    #             else:
    #                 labels.append(ids[i])
    #         return (ids, labels)
    #     if isinstance(obj, dict):
    #         return dict((n, tokenize(o, words)[0]) for n, o in obj.items()), None
    #     return list(tokenize(o, words)[0] for o in obj), None
    #
    #
    # def tokenize_chinese(obj, words):
    #     if isinstance(obj, str):
    #         tokens = tokenizer.tokenize(obj)
    #         labels = []
    #         ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    #         split = jieba.lcut(re.sub(' ', '', obj))
    #         for s in split:
    #             if s in words:
    #                 labels += [-1] * len(s)
    #             else:
    #                 labels += [1] * len(s)
    #         if len(ids) != len(labels):
    #             labels = [1] * len(ids)
    #         for i in range(len(ids)):
    #             if labels[i] == 1:
    #                 labels[i] = ids[i]
    #         return (ids, labels)
    #     if isinstance(obj, dict):
    #         return dict((n, tokenize_chinese(o, words)[0]) for n, o in obj.items()), None
    #     return list(tokenize_chinese(o, words)[0] for o in obj), None

    def tokenize_idf(obj, idf_score):
        if isinstance(obj, str):
            tokens = tokenizer.tokenize(obj)
            weights = [idf_score[w] if w in idf_score.keys() else 10.0 for w in tokens]
            ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            return (ids, weights)
        if isinstance(obj, dict):
            return dict((n, tokenize_idf(o, idf_score)[0]) for n, o in obj.items()), None
        return list(tokenize_idf(o, idf_score)[0] for o in obj), None

    dataset = test_data(args)

    predictions = []
    if args.dataset == 'CSDS':
        words = []
        with open('hit_stopwords.txt', 'r') as f:
            for line in f.readlines():
                words.append(line.strip())

    else:
        words = stopwords.words('english')
        words += list(punctuation)
        new_words = ['Ġ' + w for w in words]
        words += new_words
        words += ['âĢ', 'Ļ']

    with open(args.idf_path, 'r') as f:
        idf_score = json.load(f)
    # for instance in tqdm(dataset, mininterval=1):
    #     if args.dataset == 'CSDS':
    #         history, _ = tokenize_chinese(instance[:-1], words)
    #         response, label = tokenize_chinese(instance[-1], words)
    #     else:
    #         history, _ = tokenize(instance[:-1], words)
    #         response, label = tokenize(instance[-1], words)
    #     with torch.no_grad():
    #         output = sample_sequence(history, response, label, tokenizer, model, args)
    #         predictions += output

    for instance in tqdm(dataset, mininterval=1):
        history, _ = tokenize_idf(instance[:-1], idf_score)
        response, weight = tokenize_idf(instance[-1], idf_score)
        with torch.no_grad():
            #output = sample_sequence(history, response, weight, tokenizer, model, args, use_idf=True)
            output = sample_sequence(history, response, weight, tokenizer, model, args, use_idf=False)
            predictions += output


    with open(args.out_path, 'w', encoding="UTF-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
