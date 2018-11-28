# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import collections
import logging
import json
import re

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import _DataLoaderIter
from torch.utils.data.distributed import DistributedSampler

from src import tokenization


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

class BiInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_lan_ids):
        self.unique_id = unique_id
        self.tokens_a, self.tokens_b = tokens
        self.input_ids_a, self.input_ids_b = input_ids
        self.input_mask_a, self.input_mask_b = input_mask
        self.input_lan_ids_a, self.input_lan_ids_b = input_lan_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            print ("\r%d" % ex_index, end="")
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]
        if len(tokens_b) > seq_length - 2:
            tokens_b = tokens_b[0:(seq_length - 2)]

        tokens_a_ = []
        input_lan_ids_a_ = []
        tokens_a_.append("[CLS]")
        input_lan_ids_a_.append(0)
        for token in tokens_a:
            tokens_a_.append(token)
            input_lan_ids_a_.append(0)
        tokens_a_.append("[SEP]")
        input_lan_ids_a_.append(0)

        tokens_b_ = []
        input_lan_ids_b_ = []
        tokens_b_.append("[CLS]")
        input_lan_ids_b_.append(1)
        for token in tokens_b:
            tokens_b_.append(token)
            input_lan_ids_b_.append(1)
        tokens_b_.append("[SEP]")
        input_lan_ids_b_.append(1)

        input_ids_a_ = tokenizer.convert_tokens_to_ids(tokens_a_)
        input_ids_b_ = tokenizer.convert_tokens_to_ids(tokens_b_)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask_a_ = [1] * len(input_ids_a_)
        input_mask_b_ = [1] * len(input_ids_b_)

        # Zero-pad up to the sequence length.
        while len(input_ids_a_) < seq_length:
            input_ids_a_.append(0)
            input_mask_a_.append(0)
            input_lan_ids_a_.append(0)
        while len(input_ids_b_) < seq_length:
            input_ids_b_.append(0)
            input_mask_b_.append(0)
            input_lan_ids_b_.append(0)

        assert len(input_ids_a_) == seq_length
        assert len(input_mask_a_) == seq_length
        assert len(input_lan_ids_a_) == seq_length
        assert len(input_ids_b_) == seq_length
        assert len(input_mask_b_) == seq_length
        assert len(input_lan_ids_b_) == seq_length

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("lan0 tokens: %s" % " ".join([str(x) for x in tokens_a_]))
            logger.info("lan0 input_ids: %s" % " ".join([str(x) for x in input_ids_a_]))
            logger.info("lan0 input_mask: %s" % " ".join([str(x) for x in input_mask_a_]))
            logger.info("lan0 input_type_ids: %s" % " ".join([str(x) for x in input_lan_ids_a_]))
            logger.info("lan1 tokens: %s" % " ".join([str(x) for x in tokens_b_]))
            logger.info("lan1 input_ids: %s" % " ".join([str(x) for x in input_ids_b_]))
            logger.info("lan1 input_mask: %s" % " ".join([str(x) for x in input_mask_b_]))
            logger.info("lan1 input_type_ids: %s" % " ".join([str(x) for x in input_lan_ids_b_]))

        features.append(
            BiInputFeatures(
                unique_id=example.unique_id,
                tokens=[tokens_a_, tokens_b_],
                input_ids=[input_ids_a_, input_ids_b_],
                input_mask=[input_mask_a_, input_mask_b_],
                input_lan_ids=[input_lan_ids_a_, input_lan_ids_b_]))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples

def load(vocab_file, input_file, batch_size=32, do_lower_case=True, 
            max_seq_length=128, local_rank=-1):

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    examples = read_examples(input_file)

    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    all_input_ids_a = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
    all_input_ids_b = torch.tensor([f.input_ids_b for f in features], dtype=torch.long)
    all_input_mask_a = torch.tensor([f.input_mask_a for f in features], dtype=torch.long)
    all_input_mask_b = torch.tensor([f.input_mask_b for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids_a.size(0), dtype=torch.long)

    dataset = TensorDataset(all_input_ids_a, all_input_mask_a, 
                        all_input_ids_b, all_input_mask_b, all_example_index)
    #if local_rank == -1:
    #    sampler = SequentialSampler(dataset)
        #sampler = RandomSampler(dataset)
    #else:
    #    sampler = DistributedSampler(dataset)
    #dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataset, unique_id_to_feature, features

import sys
if __name__ == "__main__":
    vocab = sys.argv[1]
    input = sys.argv[2]
    dataset, _ = load(vocab, input)
    loader = DataLoader(data, batch_size=batch_size)
    loader_ = _DataLoaderIter(loader)
    a = next(loader_, None)
    print ("map:",a[-1])
    a = next(loader_, None)
    print ("map:",a[-1])
    loader_ = _DataLoaderIter(loader)
    a = next(loader_, None)
    print ("map:",a[-1])
    n = 0
    for a,b,c,d,id in loader:
        n+=1
        print (id)
        if n == 2:
            print ("map:",loader_.next()[-1])
