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
import numpy as np

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel, BertTokenizer, GPT2Tokenizer

import nltk
from nltk.corpus import stopwords
from string import punctuation

def train_data(args):
    with open(args.datapath, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
    if isinstance(dataset, dict):
        dataset = dataset["train"]
    return dataset

def main():
    parser = ArgumentParser()
    parser.add_argument('--gpt2', action='store_true', help="use gpt2")
    parser.add_argument("--datapath", type=str, default="", help="Path of the dataset.")
    parser.add_argument("--out_path", type=str, default="", help="Path of response generated.")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--dataset", type=str, default="samsum")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.dataset == 'CSDS':
        tokenizer_class = BertTokenizer
    else:
        tokenizer_class = GPT2Tokenizer
    model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=True)
    model = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    model.eval()

    dataset = train_data(args)
    result = {}
    doc_num = 0
    for instance in tqdm(dataset, mininterval=1):
        for s in instance:
            doc_num += 1
            tokens = tokenizer.tokenize(s)
            for w in set(tokens):
                if w in result.keys():
                    result[w] += 1
                else:
                    result[w] = 1
    for w in result.keys():
        result[w] = np.log(doc_num / result[w])

    with open(args.out_path, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()