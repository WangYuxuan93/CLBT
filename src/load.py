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

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_lan_ids, 
                    align=[None,None], align_mask=None):
        self.unique_id = unique_id
        self.tokens_a, self.tokens_b = tokens
        self.input_ids_a, self.input_ids_b = input_ids
        self.input_mask_a, self.input_mask_b = input_mask
        self.input_lan_ids_a, self.input_lan_ids_b = input_lan_ids
        self.align_ids_a, self.align_ids_b = align
        self.align_mask = align_mask

def convert_examples_to_features(examples, seq_length, tokenizer, tokenizer1=None, aligns=None):
    """Loads a data file into a list of `InputBatch`s."""

    if aligns:
        lens = [len(a) for a in aligns]
        max_align = max(lens)
        logger.info("aligned word number: %s" % (sum(lens)))
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            print ("\r%d" % ex_index, end="")
        tokens_a = tokenizer.tokenize(example.text_a)
        if tokenizer1:
            tokens_b = tokenizer1.tokenize(example.text_b)
        else:
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
        if tokenizer1:
            input_ids_b_ = tokenizer1.convert_tokens_to_ids(tokens_b_)
        else:
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

        # for alignment
        if aligns:
            align_ids_a, align_ids_b = aligns[ex_index]
            assert len(align_ids_a) == len(align_ids_b)
            align_mask = [1] * len(align_ids_a)
            while len(align_ids_a) < max_align:
                align_ids_a.append(0)
                align_ids_b.append(0)
                align_mask.append(0)

        assert len(align_ids_a) == max_align
        assert len(align_ids_b) == max_align
        assert len(align_mask) == max_align

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
            if aligns:
                logger.info("align_ids_a: %s" % " ".join([str(x) for x in align_ids_a]))
                logger.info("align_ids_b: %s" % " ".join([str(x) for x in align_ids_b]))
                logger.info("align_mask: %s" % " ".join([str(x) for x in align_mask]))
        if aligns:
            features.append(
                BiInputFeatures(
                    unique_id=example.unique_id,
                    tokens=[tokens_a_, tokens_b_],
                    input_ids=[input_ids_a_, input_ids_b_],
                    input_mask=[input_mask_a_, input_mask_b_],
                    input_lan_ids=[input_lan_ids_a_, input_lan_ids_b_],
                    align=[align_ids_a, align_ids_b],
                    align_mask=align_mask))
        else:
            features.append(
                BiInputFeatures(
                    unique_id=example.unique_id,
                    tokens=[tokens_a_, tokens_b_],
                    input_ids=[input_ids_a_, input_ids_b_],
                    input_mask=[input_mask_a_, input_mask_b_],
                    input_lan_ids=[input_lan_ids_a_, input_lan_ids_b_]))
    return features

def convert_examples_to_features_single(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
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

def load_aligns(file):
    """Load the word-level alignment, only one-to-one word pairs are kept"""
    #maps, rev_maps = [], []
    aligns = []
    with open(file, 'r') as fi:
        line = fi.readline()
        while line:
            pairs = [pair.split('-') for pair in line.strip().split()]
            # remove the one-to-many and many-to-one cases
            left = Counter([pair[0] for pair in pairs])
            right = Counter([pair[1] for pair in pairs])
            rm_left = []
            rm_right = []
            for l in left:
                if left[l] > 1:
                    rm_left.append(l)
            for r in right:
                if right[r] > 1:
                    rm_right.append(r)
            align = []
            for pair in pairs:
                if pair[0] not in rm_left and pair[1] not in rm_right:
                    align.append(pair)
            src_ids, tgt_ids = [], []
            for pair in align:
                src, tgt = [int(n) for n in pair]
                src_ids.append(src)
                tgt_ids.append(tgt)
            aligns.append((src_ids, tgt_ids))
            """
            map, rev_map = {}, {}
            for pair in align:
                src, tgt = [int(n) for n in pair]
                map[src] = tgt
            for key in map:
                rev_map[map[key]] = key
            maps.append(map)
            rev_maps.append(rev_map)
            """
            line = fi.readline()
    return aligns

def load(vocab_file, input_file, batch_size=32, do_lower_case=True, 
            max_seq_length=128, local_rank=-1, vocab_file1=None,
            align_file=None):

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    tokenizer1 = tokenization.FullTokenizer(
        vocab_file=vocab_file1, do_lower_case=do_lower_case)

    examples = read_examples(input_file)

    aligns = None
    if align_file:
        aligns = load_aligns(align_file)
        try:
            assert len(examples) == len(aligns)
        except:
            raise ValueError("Number of examples({}) and alignments({}) mismatch!".format(len(examples),len(aligns)))

    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_length, tokenizer=tokenizer, 
        tokenizer1=tokenizer1, aligns=aligns)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    all_input_ids_a = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
    all_input_ids_b = torch.tensor([f.input_ids_b for f in features], dtype=torch.long)
    all_input_mask_a = torch.tensor([f.input_mask_a for f in features], dtype=torch.long)
    all_input_mask_b = torch.tensor([f.input_mask_b for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids_a.size(0), dtype=torch.long)

    if align_file:
        all_align_ids_a = torch.tensor([f.align_ids_a for f in features], dtype=torch.long)
        all_align_ids_b = torch.tensor([f.align_ids_b for f in features], dtype=torch.long)
        all_align_mask = torch.tensor([f.align_mask for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids_a, all_input_mask_a, 
                        all_input_ids_b, all_input_mask_b, all_align_ids_a, all_align_ids_b,
                        all_align_mask, all_example_index)
    else:
        dataset = TensorDataset(all_input_ids_a, all_input_mask_a, 
                        all_input_ids_b, all_input_mask_b, all_example_index)
    #if local_rank == -1:
    #    sampler = SequentialSampler(dataset)
        #sampler = RandomSampler(dataset)
    #else:
    #    sampler = DistributedSampler(dataset)
    #dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataset, unique_id_to_feature, features

def load_single(vocab_file, input_file, batch_size=32, do_lower_case=True, 
            max_seq_length=128, local_rank=-1):

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    examples = read_examples(input_file)

    features = convert_examples_to_features_single(
        examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_example_index)

    return dataset, unique_id_to_feature, features

def convert_sents_to_features(sents, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    examples = []
    unique_id = 0
    for sent in sents:
        sent = [w.lower() for w in sent]
        examples.append(InputExample(unique_id=unique_id, text_a=sent, text_b=None))
        unique_id += 1

    features = []
    for (ex_index, example) in enumerate(examples):
        #tokens_a = tokenizer.tokenize(example.text_a)
        tokens_a = []
        for token in example.text_a:
            for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                tokens_a.append(sub_token)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

def convert(vocab_file, sents, batch_size=32, do_lower_case=True, 
            max_seq_length=128, local_rank=-1):

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    features = convert_sents_to_features(
        sents=sents, seq_length=max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_example_index)

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
