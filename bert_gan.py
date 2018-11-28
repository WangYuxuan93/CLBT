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

from src.utils import bool_flag, initialize_exp
from src.load import load
from src.build_model import build_model
from src.bert_self.trainer import Bertself.trainer
from src.evaluation import Bertself.evaluator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def main():
    # main
    parser = argparse.ArgumentParser(description='Unsupervised training')
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--model_path", type=str, default="", help="Where to store experiment logs and models")
    # data
    parser.add_argument("--src_lang", type=str, default='en', help="Source language")
    parser.add_argument("--tgt_lang", type=str, default='de', help="Target language")

    # self.mapping
    parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the self.mapping as an identity matrix")
    parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
    parser.add_argument("--map_clip_weights", type=float, default=5, help="Clip self.mapping weights")
    # self.discriminator
    parser.add_argument("--dis_layers", type=int, default=2, help="self.discriminator layers")
    parser.add_argument("--dis_hid_dim", type=int, default=2048, help="self.discriminator hidden layer dimensions")
    parser.add_argument("--dis_dropout", type=float, default=0., help="self.discriminator dropout")
    parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="self.discriminator input dropout")
    parser.add_argument("--dis_steps", type=int, default=5, help="self.discriminator steps")
    parser.add_argument("--dis_lambda", type=float, default=1, help="self.discriminator loss feedback coefficient")
    parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
    parser.add_argument("--dis_smooth", type=float, default=0.1, help="self.discriminator smooth predictions")
    parser.add_argument("--dis_clip_weights", type=float, default=5, help="Clip self.discriminator weights")
    # training adversarial
    parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="self.mapping optimizer")
    parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="self.discriminator optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
    parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
    # training refinement
    parser.add_argument("--n_refinement", type=int, default=0, help="Number of refinement iterations (0 to disable the refinement procedure)")
    # for bert
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--vocab_file", default=None, type=str, required=True, 
                            help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--bert_config_file", default=None, type=str, required=True,
                            help="The config json file corresponding to the pre-trained BERT model. "
                                "This specifies the model architecture.")
    parser.add_argument("--init_checkpoint", default=None, type=str, required=True, 
                            help="Initial checkpoint (usually from a pre-trained BERT model).")
    ## Other parameters
    parser.add_argument("--bert_layer", default=-1, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_lower_case", default=True, action='store_true', 
                        help="Whether to lower case the input text. Should be True for uncased "
                            "models and False for cased models.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--dev_sent_num", default=10000, type=int, help="Number of sentence pair for development(sentence similarity).")
    parser.add_argument("--print_every_map_steps", default=100, type=int, help="Print every ? self.mapping steps.")
    parser.add_argument("--local_rank",type=int, default=-1, help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--rm_stop_words", default=True, action='store_true', help="Whether to remove stop words while evaluating(sentence similarity)")
    parser.add_argument("--stop_words_src", type=str, default="", help="Stop word file for source language")
    parser.add_argument("--stop_words_tgt", type=str, default="", help="Stop word file for target language")
    parser.add_argument("--save_dis", default=True, action='store_true', help="Whether to save self.discriminator")
    # parse parameters
    args = parser.parse_args()

    if args.adversarial:
        train_adv()

    if args.n_refinement > 0:
        refine()

class AdvBert(object):

    def __init__(self, args):

        self.args = args
        # check parameters
        assert 0 <= self.args.dis_dropout < 1
        assert 0 <= self.args.dis_input_dropout < 1
        assert 0 <= self.args.dis_smooth < 0.5
        assert self.args.dis_lambda > 0 and self.args.dis_steps > 0
        assert 0 < self.args.lr_shrink <= 1
        # build model / trainer / evaluator
        self.logger = initialize_exp(self.args)

        self.dataset, unique_id_to_feature, self.features = load(self.args.vocab_file, self.args.input_file, batch_size=self.args.batch_size, 
                                        do_lower_case=self.args.do_lower_case, max_seq_length=self.args.max_seq_length, 
                                        local_rank=self.args.local_rank)
        self.bert_model, self.mapping, self.discriminator = build_model(self.args, True)
        self.trainer = Bertself.trainer(self.bert_model, self.dataset, self.mapping, self.discriminator, self.args)
        self.evaluator = Bertself.evaluator(self.trainer, self.features)

    def train_adv(self):
        """
        Learning loop for Adversarial Training
        """

        if self.args.local_rank == -1 or self.args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        else:
            device = torch.device("cuda", self.args.local_rank)

        self.logger.info('----> ADVERSARIAL TRAINING <----\n\n')

        sampler = RandomSampler(self.dataset)
        train_loader = DataLoader(self.dataset, sampler=sampler, batch_size=self.args.batch_size)
        # training loop
        for n_epoch in range(self.args.n_epochs):

            self.logger.info('Starting adversarial training epoch %i...' % n_epoch)
            tic = time.time()
            n_words_proc = 0
            n_dis_step = 0
            n_map_step = 0
            stats = {'DIS_COSTS': []}
            for input_ids_a, input_mask_a, input_ids_b, input_mask_b, example_indices in train_loader:
                input_ids_a = input_ids_a.to(device)
                input_mask_a = input_mask_a.to(device)
                input_ids_b = input_ids_b.to(device)
                input_mask_b = input_mask_b.to(device)

                src_bert = self.trainer.get_bert(input_ids_a, input_mask_a, bert_layer=self.args.bert_layer)
                tgt_bert = self.trainer.get_bert(input_ids_b, input_mask_b, bert_layer=self.args.bert_layer)
                self.trainer.dis_step(src_bert, tgt_bert, stats)

                n_dis_step += 1
                if n_dis_step % self.args.dis_steps == 0:
                    n_words_proc += self.trainer.self.mapping_step(stats)
                    n_map_step += 1

                # log stats
                if n_map_step % self.args.print_every_map_steps == 0:
                    stats_str = [('DIS_COSTS', 'self.discriminator loss')]
                    stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                                 for k, v in stats_str if len(stats[k]) > 0]
                    stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                    self.logger.info(('%06i - ' % n_dis_step) + ' - '.join(stats_log))

                    # reset
                    tic = time.time()
                    n_words_proc = 0
                    for k, _ in stats_str:
                        del stats[k][:]

            # embeddings / self.discriminator evaluation
            to_log = OrderedDict({'n_epoch': n_epoch})
            sent_sim = self.evaluator.sent_sim(rm_stop_words=self.args.rm_stop_words)
            self.evaluator.eval_dis()

            # JSON log / save best model / end of epoch
            #self.logger.info("__log__:%s" % json.dumps(to_log))
            self.trainer.save_best(sent_sim)
            self.logger.info('End of epoch %i.\n\n' % n_epoch)

            # update the learning rate (stop if too small)
            self.trainer.update_lr(sent_sim)
            if self.trainer.map_optimizer.param_groups[0]['lr'] < self.args.min_lr:
                self.logger.info('Learning rate < 1e-6. BREAK.')
                break

    def refine(self):
        """
        Learning loop for Procrustes Iterative Refinement
        """

        # Get the best self.mapping according to VALIDATION_METRIC
        self.logger.info('----> ITERATIVE PROCRUSTES REFINEMENT <----\n\n')
        self.trainer.reload_best()

        # training loop
        for n_iter in range(self.args.n_refinement):

            self.logger.info('Starting refinement iteration %i...' % n_iter)

            # apply the Procrustes solution
            self.trainer.procrustes()

            sent_sim = self.evaluator.sent_sim(rm_stop_words=self.args.rm_stop_words)

            # JSON log / save best model / end of epoch
            #self.logger.info("__log__:%s" % json.dumps(to_log))
            self.trainer.save_best(sent_sim)
            self.logger.info('End of refinement iteration %i.\n\n' % n_iter)

    def pred(self):
        """
        """

        self.trainer.reload_best()


