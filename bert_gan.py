# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
import string

from src.utils import bool_flag, initialize_exp
from src.load import load, load_single, convert
from src.build_model import build_model
from src.bert_trainer import BertTrainer, reload_model
from src.bert_evaluator import BertEvaluator
from src.bert_evaluator import load_stop_words, rm_stop_words, cos_sim, get_overlaps, get_overlap_sim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def main():
    # main
    parser = argparse.ArgumentParser(description='Unsupervised training')
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--model_path", type=str, default=None, help="Where to store experiment logs and models")
    # data
    parser.add_argument("--src_lang", type=str, default='en', help="Source language")
    parser.add_argument("--tgt_lang", type=str, default='de', help="Target language")

    # self.mapping
    parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the self.mapping as an identity matrix")
    parser.add_argument("--map_beta", type=float, default=0.01, help="Beta for orthogonalization")
    parser.add_argument("--map_clip_weights", type=float, default=5, help="Clip self.mapping weights")
    # self.discriminator
    parser.add_argument("--dis_layers", type=int, default=3, help="self.discriminator layers")
    parser.add_argument("--dis_hid_dim", type=int, default=2048, help="self.discriminator hidden layer dimensions")
    parser.add_argument("--dis_dropout", type=float, default=0., help="self.discriminator dropout")
    parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="self.discriminator input dropout")
    parser.add_argument("--dis_steps", type=int, default=5, help="self.discriminator steps")
    parser.add_argument("--dis_lambda", type=float, default=1, help="self.discriminator loss feedback coefficient")
    parser.add_argument("--dis_smooth", type=float, default=0.2, help="self.discriminator smooth predictions")
    parser.add_argument("--dis_clip_weights", type=float, default=5, help="Clip self.discriminator weights")
    parser.add_argument("--dis_lr_decay", type=float, default=0.98, help="Discriminator learning rate decay (SGD only)")
    # training adversarial
    parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="self.mapping optimizer")
    parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="self.discriminator optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.95, help="Learning rate decay (SGD only)") 
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
    parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
    # training refinement
    parser.add_argument("--n_refinement", type=int, default=0, help="Number of refinement iterations (0 to disable the refinement procedure)")
    # for bert
    parser.add_argument("--input_file", default=None, type=str)
    parser.add_argument("--vocab_file", default=None, type=str, required=True, 
                            help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--bert_config_file", default=None, type=str, required=True,
                            help="The config json file corresponding to the pre-trained BERT model. "
                                "This specifies the model architecture.")
    parser.add_argument("--init_checkpoint", default=None, type=str, required=True, 
                            help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--vocab_file1", default=None, type=str, required=True, 
                            help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--bert_config_file1", default=None, type=str, required=True,
                            help="The config json file corresponding to the pre-trained BERT model. "
                                "This specifies the model architecture.")
    parser.add_argument("--init_checkpoint1", default=None, type=str, required=True, 
                            help="Initial checkpoint (usually from a pre-trained BERT model).")
    # Other parameters
    parser.add_argument("--bert_layer", default=-1, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_lower_case", default=True, action='store_true', 
                        help="Whether to lower case the input text. Should be True for uncased "
                            "models and False for cased models.")
    parser.add_argument("--dev_sent_num", default=1000, type=int, help="Number of sentence pair for development(sentence similarity).")
    parser.add_argument("--print_every_dis_steps", default=100, type=int, help="Print every ? self.discriminator steps.")
    parser.add_argument("--save_every_dis_steps", default=1000, type=int, help="Save every ? self.discriminator steps.")
    parser.add_argument("--local_rank",type=int, default=-1, help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--rm_stop_words", default=False, action='store_true', help="Whether to remove stop words while evaluating(sentence similarity)")
    parser.add_argument("--rm_punc", default=False, action='store_true', help="Whether to remove punctuation while evaluating(sentence similarity)")
    parser.add_argument("--stop_words_src", type=str, default="", help="Stop word file for source language")
    parser.add_argument("--stop_words_tgt", type=str, default="", help="Stop word file for target language")
    parser.add_argument("--save_dis", default=False, action='store_true', help="Whether to save self.discriminator")
    parser.add_argument("--eval_non_parallel", default=False, action='store_true', help="Whether to add disorder sentence while evaluating(sentence similarity)")
    # For predict
    parser.add_argument("--pred", type=bool_flag, default=False, help="Map source bert to target space")
    parser.add_argument("--src_file", default=None, type=str, help="The source input file")
    parser.add_argument("--output_file", default=None, type=str, help="The output file of mapped source language embeddings")
    parser.add_argument("--cal_sent_sim", type=bool_flag, default=False, help="Calculate sentence similarity?")
    parser.add_argument("--sim_with_map", default=False, action='store_true', help="Calculate similarity with mapping?")
    parser.add_argument("--overlap_sim", default=False, action='store_true', help="Calculate similarity of overlap words?")
    parser.add_argument("--base_embed", default=False, action='store_true', help="Use base embeddings of BERT?")
    parser.add_argument("--map_input", default=False, action='store_true', help="Apply mapping to the BERT input embeddings?")
    parser.add_argument("--sim_file", type=str, default="", help="output similarity file")
    # For non-linear mapping
    parser.add_argument("--non_linear", action='store_true', default=False, help="Use non-linear mapping")
    parser.add_argument("--activation", type=str, default='leaky_relu', help="learky_relu,tanh")
    parser.add_argument("--n_layers", type=int, default=1, help="mapping layer")
    parser.add_argument("--hidden_size", type=int, default=768, help="mapping hidden layer size")
    # parse parameters
    args = parser.parse_args()

    advbert = AdvBert(args)

    if not args.pred and args.adversarial:
        advbert.train_adv()

    if args.n_refinement > 0:
        advbert.refine()

    if args.pred:
        advbert.pred()

    if args.cal_sent_sim:
        advbert.calculate_sim()

class Args(object):

    def __init__(self, model_path, vocab_file, bert_config_file, init_checkpoint, 
                output_file, max_seq_length=128, bert_layer=-1, map_input=False,
                vocab_file1=None, bert_config_file1=None, init_checkpoint1=None,
                non_linear=False, activation="leaky_relu", n_layers=2, hidden_size=768,
                emb_dim=768):

        self.adversarial = False
        self.pred = True
        self.no_cuda = False
        self.cal_sent_sim = False
        self.local_rank = -1
        self.batch_size = 32
        self.do_lower_case = True
        self.map_input = map_input
        self.transformer = None
        self.fine_tune = False

        self.vocab_file = vocab_file
        self.bert_config_file = bert_config_file
        self.init_checkpoint = init_checkpoint
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.output_file = output_file
        self.bert_layer = bert_layer

        self.vocab_file1 = vocab_file1
        self.bert_config_file1 = bert_config_file1
        self.init_checkpoint1 = init_checkpoint1

        self.non_linear = non_linear
        self.activation = activation
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim

class AdvBert(object):

    def __init__(self, args):

        self.args = args
        # check parameters
        if self.args.adversarial:
            assert 0 <= self.args.dis_dropout < 1
            assert 0 <= self.args.dis_input_dropout < 1
            assert 0 <= self.args.dis_smooth < 0.5
            assert self.args.dis_lambda > 0 and self.args.dis_steps > 0
            assert 0 < self.args.lr_shrink <= 1
            assert self.args.model_path is not None
        self.dataset = None
        # build model / trainer / evaluator
        if not self.args.pred and not self.args.cal_sent_sim:
            self.logger = initialize_exp(self.args)
        if self.args.adversarial or self.args.cal_sent_sim:
            assert os.path.isfile(self.args.input_file)
            self.dataset, unique_id_to_feature, self.features = load(self.args.vocab_file, 
                    self.args.input_file, batch_size=self.args.batch_size, 
                    do_lower_case=self.args.do_lower_case, max_seq_length=self.args.max_seq_length, 
                    local_rank=self.args.local_rank, vocab_file1=self.args.vocab_file1)
        self.bert_model, self.mapping, self.discriminator, self.bert_model1 = build_model(self.args, True)

        if self.args.adversarial or self.args.pred:
            self.trainer = BertTrainer(self.bert_model, self.dataset, self.mapping, self.discriminator, 
                                    self.args, bert_model1=self.bert_model1)

        if self.args.adversarial or self.args.cal_sent_sim:
            self.evaluator = BertEvaluator(self.bert_model, self.dataset, self.mapping, self.discriminator, 
                                    self.args, self.features, bert_model1=self.bert_model1)

        if self.args.local_rank == -1 or self.args.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        else:
            self.device = torch.device("cuda", self.args.local_rank)

    def train_adv(self):
        """
        Learning loop for Adversarial Training
        """

        self.logger.info('----> ADVERSARIAL TRAINING <----\n\n')

        sampler = RandomSampler(self.dataset)
        train_loader = DataLoader(self.dataset, sampler=sampler, batch_size=self.args.batch_size)
        # training loop
        for n_epoch in range(self.args.n_epochs):
            path = os.path.join(self.args.model_path, "epoch-{}".format(n_epoch))

            self.logger.info('Starting adversarial training epoch %i...' % n_epoch)
            tic = time.time()
            n_words_proc = 0
            n_dis_step = 0
            n_map_step = 0
            stats = {'DIS_COSTS': []}
            for input_ids_a, input_mask_a, input_ids_b, input_mask_b, example_indices in train_loader:
                input_ids_a = input_ids_a.to(self.device)
                input_mask_a = input_mask_a.to(self.device)
                input_ids_b = input_ids_b.to(self.device)
                input_mask_b = input_mask_b.to(self.device)

                src_bert = self.trainer.get_bert(input_ids_a, input_mask_a, 
                                                bert_layer=self.args.bert_layer, model_id=0)
                tgt_bert = self.trainer.get_bert(input_ids_b, input_mask_b, 
                                                bert_layer=self.args.bert_layer, model_id=1)
                self.trainer.dis_step(src_bert, tgt_bert, stats)

                n_dis_step += 1
                if n_dis_step % self.args.dis_steps == 0:
                    n_words_proc += self.trainer.mapping_step(stats)
                    n_map_step += 1

                # log stats
                if n_dis_step % self.args.print_every_dis_steps == 0:
                    stats_str = [('DIS_COSTS', 'Discriminator loss')]
                    stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                                 for k, v in stats_str if len(stats[k]) > 0]
                    stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                    self.logger.info(('Epoch: {:0>3d} | Step: {:0>6d} - '.format(n_epoch, n_dis_step)) + ' | '.join(stats_log))

                    # reset
                    tic = time.time()
                    n_words_proc = 0
                    for k, _ in stats_str:
                        del stats[k][:]
                if n_dis_step % self.args.save_every_dis_steps == 0:
                    metric = self.evaluator.eval_sim()
                    self.evaluator.eval_dev_dis()
                    self.trainer.save_best(metric["para_sim"], path=path)

            # embeddings / self.discriminator evaluation
            #to_log = OrderedDict({'n_epoch': n_epoch})
            metric = self.evaluator.eval_sim()
            self.evaluator.eval_dev_dis()
            self.trainer.save_best(metric["para_sim"], path=path)

            # JSON log / save best model / end of epoch
            #self.logger.info("__log__:%s" % json.dumps(to_log))
            if not os.path.exists(path):
                self.trainer.save_epoch(path, n_epoch)
            self.logger.info('End of epoch %i.\n\n' % n_epoch)

            # update the learning rate (stop if too small)
            self.trainer.update_lr(metric["para_sim"])
            #self.trainer.update_dis_lr(metric["para_sim"])
            if self.trainer.map_optimizer.param_groups[0]['lr'] < self.args.min_lr:
                self.logger.info('Learning rate < 1e-6. BREAK.')
                break

    def refine(self):
        """
        Learning loop for Procrustes Iterative Refinement
        """

        # Get the best self.mapping according to VALIDATION_METRIC
        self.logger.info('----> ITERATIVE PROCRUSTES REFINEMENT <----\n\n')
        self.trainer.load_best()

        # training loop
        for n_iter in range(self.args.n_refinement):

            self.logger.info('Starting refinement iteration %i...' % n_iter)

            # apply the Procrustes solution
            self.trainer.procrustes()

            metric = self.evaluator.eval_sim()

            # JSON log / save best model / end of epoch
            #self.logger.info("__log__:%s" % json.dumps(to_log))
            self.trainer.save_best(metric["para_sim"])
            self.logger.info('End of refinement iteration %i.\n\n' % n_iter)

    def pred(self):
        """
        Map bert of source language to target space
        """
        assert self.args.src_file is not None
        assert self.args.output_file is not None

        self.trainer.load_best()
        pred_dataset, unique_id_to_feature, features = load_single(self.args.vocab_file, 
                        self.args.src_file, batch_size=self.args.batch_size, 
                        do_lower_case=self.args.do_lower_case, 
                        max_seq_length=self.args.max_seq_length, 
                        local_rank=self.args.local_rank)
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=self.args.batch_size)
        self.bert_model.eval()
        with open(self.args.output_file, "w", encoding='utf-8') as writer:
            for input_ids, input_mask, example_indices in pred_dataloader:
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)

                if self.args.map_input:
                    all_encoder_layers, _ = self.bert_model(input_ids, token_type_ids=None, 
                                                attention_mask=input_mask, input_mapping=self.trainer.mapping)
                    target_layer = all_encoder_layers[self.args.bert_layer]
                else:
                    all_encoder_layers, _ = self.bert_model(input_ids, token_type_ids=None,
                                                attention_mask=input_mask)
                    src_encoder_layer = all_encoder_layers[self.args.bert_layer]
                    target_layer = self.trainer.mapping(src_encoder_layer)

                for b, example_index in enumerate(example_indices):
                    feature = features[example_index.item()]
                    unique_id = int(feature.unique_id)
                    # feature = unique_id_to_feature[unique_id]
                    output_json = OrderedDict()
                    output_json["linex_index"] = unique_id
                    all_out_features = []
                    for (i, token) in enumerate(feature.tokens):
                        all_layers = []
                        layer_output = target_layer.detach().cpu().numpy()
                        layer_output = layer_output[b]
                        layers = OrderedDict()
                        layers["index"] = self.args.bert_layer
                        layers["values"] = [
                            round(x.item(), 6) for x in layer_output[i]
                        ]
                        all_layers.append(layers)
                        out_features = OrderedDict()
                        out_features["token"] = token
                        out_features["layers"] = all_layers
                        all_out_features.append(out_features)
                    output_json["features"] = all_out_features
                    writer.write(json.dumps(output_json) + "\n")

    def list2bert(self, sents):
        """
        Map bert of source language to target space
        """
        assert self.args.output_file is not None

        self.trainer.load_best()
        pred_dataset, unique_id_to_feature, features = convert(self.args.vocab_file, 
                        sents, batch_size=self.args.batch_size, 
                        do_lower_case=self.args.do_lower_case, 
                        max_seq_length=self.args.max_seq_length, 
                        local_rank=self.args.local_rank)
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=self.args.batch_size)
        self.bert_model.eval()
        self.trainer.mapping.eval()
        with open(self.args.output_file, "w", encoding='utf-8') as writer:
            for input_ids, input_mask, example_indices in pred_dataloader:
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)

                if self.args.map_input:
                    all_encoder_layers, _ = self.bert_model(input_ids, token_type_ids=None, 
                                                attention_mask=input_mask, input_mapping=self.trainer.mapping)
                    target_layer = all_encoder_layers[self.args.bert_layer]
                else:
                    all_encoder_layers, _ = self.bert_model(input_ids, token_type_ids=None,
                                                attention_mask=input_mask)
                    src_encoder_layer = all_encoder_layers[self.args.bert_layer]
                    target_layer = self.trainer.mapping(src_encoder_layer)

                for b, example_index in enumerate(example_indices):
                    feature = features[example_index.item()]
                    unique_id = int(feature.unique_id)
                    # feature = unique_id_to_feature[unique_id]
                    output_json = OrderedDict()
                    output_json["linex_index"] = unique_id
                    all_out_features = []
                    for (i, token) in enumerate(feature.tokens):
                        all_layers = []
                        layer_output = target_layer.detach().cpu().numpy()
                        layer_output = layer_output[b]
                        layers = OrderedDict()
                        layers["index"] = self.args.bert_layer
                        layers["values"] = [
                            round(x.item(), 6) for x in layer_output[i]
                        ]
                        all_layers.append(layers)
                        out_features = OrderedDict()
                        out_features["token"] = token
                        out_features["layers"] = all_layers
                        all_out_features.append(out_features)
                    output_json["features"] = all_out_features
                    writer.write(json.dumps(output_json) + "\n")

    def get_bert(self, input_ids, input_mask, bert_layer=-1, model_id=0):
        """
        Get BERT
        """
        with torch.no_grad():
            if model_id == 0 or self.bert_model1 is None:
                self.bert_model.eval()
                all_encoder_layers, _ = self.bert_model(input_ids, token_type_ids=None, attention_mask=input_mask)
            else:
                self.bert_model1.eval()
                all_encoder_layers, _ = self.bert_model1(input_ids, token_type_ids=None, attention_mask=input_mask)
            encoder_layer = all_encoder_layers[bert_layer]
        
        # [batch_size, seq_len, output_dim]
        return encoder_layer

    def calculate_sim(self):
        """
        Learning loop for Adversarial Training
        """

        print ('----> Calculate Sentence Similarity <----\n\n')

        sampler = SequentialSampler(self.dataset)
        loader = DataLoader(self.dataset, sampler=sampler, batch_size=self.args.batch_size)
        self.evaluator.calculate_sim(loader)

if __name__ == "__main__":
  main()
