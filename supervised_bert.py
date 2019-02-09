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
from src.load import load, load_single, convert, load_from_bert
from src.build_model import build_model
from src.supervised_bert_trainer import SupervisedBertTrainer
from src.bert_evaluator import BertEvaluator
from src.bert_evaluator import load_stop_words, rm_stop_words, cos_sim, get_overlaps, get_overlap_sim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import _DataLoaderIter

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
    # training
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="self.mapping optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.95, help="Learning rate decay (SGD only)") 
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
    parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
    parser.add_argument("--decay_step", type=int, default=100, help="Learning rate decay step (SGD only)")
    parser.add_argument("--save_all", default=False, action='store_true', help="Save every model?")
    parser.add_argument("--quit_after_n_epochs_without_improvement", type=int, default=500, help="Quit after n epochs without improvement")
    parser.add_argument("--normalize_embed", type=bool_flag, default=False, help="Normalize embeddings? (should be false with l2_dist loss)")
    parser.add_argument("--loss", type=str, default="l2_dist", help="loss type (cos_sim, max_margin_top-k, l2_dist)")
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
    # For supervised learning
    parser.add_argument("--adversarial", default=False, action='store_true', help="Adversarial training?")
    parser.add_argument("--align_file", default=None, type=str, help="The alignment file of paralleled sentences")
    parser.add_argument("--map_type", type=str, default='linear', help="svd|linear|nonlinear|self_attention|attention|linear_self_attention|nonlinear_self_attention|fine_tune")
    parser.add_argument("--emb_dim", type=int, default=768, help="BERT embedding dimension")
    # For non-linear mapping
    #parser.add_argument("--non_linear", action='store_true', default=False, help="Use non-linear mapping")
    parser.add_argument("--activation", type=str, default='leaky_relu', help="learky_relu,tanh")
    parser.add_argument("--n_layers", type=int, default=1, help="mapping layer")
    parser.add_argument("--hidden_size", type=int, default=768, help="mapping hidden layer size")
    # For attention-based mapping
    #parser.add_argument("--transformer", type=str, default=None, help="self_attention|attention")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="attention head number")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="attention probability dropout rate")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="attention hidden layer dropout rate")
    parser.add_argument("--load_pred_bert", action='store_true', default=False, help="Directly load predicted BERT")
    parser.add_argument("--bert_file0", default=None, type=str, help="Input predicted BERT file for language 0")
    parser.add_argument("--bert_file1", default=None, type=str, help="Input predicted BERT file for language 1")
    parser.add_argument("--n_max_sent", type=int, default=None, help="Maximum BERT sentence number")
    # Fine-tuning
    #parser.add_argument("--fine_tune", action='store_true', default=False, help="Fine tune on src BERT model")
    parser.add_argument("--save_sim", type=bool_flag, default=True, help="Save model by cosine similarity?")
    # parse parameters
    args = parser.parse_args()

    bert_super = SupervisedBert(args)

    if not (args.pred or args.map_type=='svd'):
        bert_super.train()

    if args.map_type=='svd':
        bert_super.svd()

    if args.pred:
        bert_super.pred()

    if args.cal_sent_sim:
        bert_super.calculate_sim()

class Args(object):

    def __init__(self, model_path, vocab_file, bert_config_file, init_checkpoint, 
                output_file, max_seq_length=128, bert_layer=-1, map_input=False,
                vocab_file1=None, bert_config_file1=None, init_checkpoint1=None,
                map_type='linear', activation="leaky_relu", n_layers=2, hidden_size=768,
                emb_dim=768, num_attention_heads=12, attention_probs_dropout_prob=0, 
                hidden_dropout_prob=0):

        self.adversarial = False
        self.pred = True
        self.no_cuda = False
        self.cal_sent_sim = False
        self.load_pred_bert = False
        self.local_rank = -1
        self.batch_size = 32
        self.do_lower_case = True
        self.map_input = map_input
        self.map_type = map_type
        #self.fine_tune = fine_tune

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

        #self.non_linear = non_linear
        self.activation = activation
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim

        #self.transformer = transformer
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob

class SupervisedBert(object):

    def __init__(self, args):

        self.args = args
        # check parameters
        if not self.args.pred:
            assert 0 < self.args.lr_shrink <= 1
            assert self.args.model_path is not None
        self.dataset = None
        # build model / trainer / evaluator
        if not self.args.pred:
            self.logger = initialize_exp(self.args)

        self.bert_model, self.mapping, self.discriminator, self.bert_model1 = build_model(self.args, True)

        if self.args.local_rank == -1 or self.args.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        else:
            self.device = torch.device("cuda", self.args.local_rank)
        self.transformer_types = ['self_attention','attention','linear_self_attention','nonlinear_self_attention']

    def train(self):
        """
        """

        if self.args.load_pred_bert:
            assert self.args.bert_file0 is not None
            assert self.args.bert_file1 is not None
            self.dataset, unique_id_to_feature, self.features = load_from_bert(self.args.vocab_file, self.args.bert_file0,
                self.args.bert_file1, do_lower_case=self.args.do_lower_case, 
                max_seq_length=self.args.max_seq_length, n_max_sent=self.args.n_max_sent,
                vocab_file1=self.args.vocab_file1, align_file=self.args.align_file)
        else:
            self.dataset, unique_id_to_feature, self.features = load(self.args.vocab_file, self.args.input_file,
                batch_size=self.args.batch_size, do_lower_case=self.args.do_lower_case, 
                max_seq_length=self.args.max_seq_length, local_rank=self.args.local_rank, 
                vocab_file1=self.args.vocab_file1, align_file=self.args.align_file)
        self.trainer = SupervisedBertTrainer(self.bert_model, self.mapping, self.discriminator, 
                                    self.args, bert_model1=self.bert_model1, trans_types=self.transformer_types)

        sampler = RandomSampler(self.dataset)
        train_loader = DataLoader(self.dataset, sampler=sampler, batch_size=self.args.batch_size)

        n_without_improvement = 0
        min_loss = 1e6
        path4loss = self.args.model_path + '/model4loss'
        if not os.path.exists(path4loss):
            os.makedirs(path4loss)
        if self.args.save_all:
            model_log = open(path4loss+'/model.log', 'w')

        # training loop
        for n_epoch in range(self.args.n_epochs):
            #self.logger.info('Starting epoch %i...' % n_epoch)
            if (n_epoch+1) % self.args.decay_step == 0:
                self.trainer.decay_map_lr()
            n_inst = 0
            n_batch = 0
            to_log = {"avg_cosine_similarity": 0, "loss": 0}

            if self.args.load_pred_bert:
                for input_embs_a, input_mask_a, input_embs_b, input_mask_b, align_ids_a, align_ids_b, align_mask, example_indices in train_loader:
                    n_batch += 1
                    with torch.no_grad():
                        input_embs_a = input_embs_a.to(self.device)
                        input_mask_a = input_mask_a.to(self.device)
                        input_embs_b = input_embs_b.to(self.device)
                    align_ids_a = align_ids_a.to(self.device)
                    align_ids_b = align_ids_b.to(self.device)
                    align_mask = align_mask.to(self.device)
                    #print (align_ids_a, align_ids_b, align_mask)
                    src_bert = self.trainer.get_indexed_mapped_bert_from_bert(
                                    input_embs_a, input_mask_a, align_ids_a, align_mask, 
                                    bert_layer=self.args.bert_layer)
                    tgt_bert = self.trainer.get_indexed_bert_from_bert(
                                    input_embs_b, align_ids_b, align_mask,
                                    bert_layer=self.args.bert_layer)

                    avg_cos_sim, loss = self.trainer.supervised_mapping_step(src_bert, tgt_bert)
                    n_inst += src_bert.size()[0]
                    cos_sim = avg_cos_sim.cpu().detach().numpy()
                    loss_ = loss.cpu().detach().numpy()

                    to_log["avg_cosine_similarity"] += cos_sim
                    to_log["loss"] += loss_
            else:
                for input_ids_a, input_mask_a, input_ids_b, input_mask_b, align_ids_a, align_ids_b, align_mask, example_indices in train_loader:
                    n_batch += 1
                    input_ids_a = input_ids_a.to(self.device)
                    input_mask_a = input_mask_a.to(self.device)
                    input_ids_b = input_ids_b.to(self.device)
                    input_mask_b = input_mask_b.to(self.device)
                    align_ids_a = align_ids_a.to(self.device)
                    align_ids_b = align_ids_b.to(self.device)
                    align_mask = align_mask.to(self.device)
                    #print (align_ids_a, align_ids_b, align_mask)
                    src_bert = self.trainer.get_indexed_mapped_bert(
                                    input_ids_a, input_mask_a, align_ids_a, align_mask, 
                                    bert_layer=self.args.bert_layer, model_id=0)
                    tgt_bert = self.trainer.get_indexed_bert(
                                    input_ids_b, input_mask_b, align_ids_b, align_mask,
                                    bert_layer=self.args.bert_layer, model_id=1)

                    avg_cos_sim, loss = self.trainer.supervised_mapping_step(src_bert, tgt_bert)
                    n_inst += src_bert.size()[0]
                    cos_sim = avg_cos_sim.cpu().detach().numpy()
                    loss_ = loss.cpu().detach().numpy()

                    to_log["avg_cosine_similarity"] += cos_sim
                    to_log["loss"] += loss_
            to_log["avg_cosine_similarity"] /= n_batch
            to_log["loss"] /= n_batch
            self.logger.info("Epoch:{}, avg cos sim:{:.6f}, avg loss:{:.6f}, instances:{}".format(n_epoch, to_log["avg_cosine_similarity"], to_log["loss"], n_inst))

            if to_log["avg_cosine_similarity"] <= self.trainer.best_valid_metric and to_log["loss"] >= min_loss:
                n_without_improvement += 1
            else:
                n_without_improvement = 0
            if to_log["loss"] < min_loss:
                self.logger.info(" Minimum loss : {:.6f}".format(to_log["loss"]))
                if self.args.save_all:
                    save_path = path4loss+'/epoch-'+str(n_epoch)
                    model_log.write("Epoch:{}, avg cos sim:{:.6f}, avg loss:{:.6f}\n".format(n_epoch, 
                                    to_log["avg_cosine_similarity"], to_log["loss"]))
                else:
                    save_path = path4loss
                self.trainer.save_model(save_path+'/best_mapping.pkl')
                min_loss = to_log["loss"]
            if self.args.save_sim:
                self.trainer.save_best(to_log, "avg_cosine_similarity")
            else:
                if to_log["avg_cosine_similarity"] > self.trainer.best_valid_metric:
                    self.trainer.best_valid_metric = to_log["avg_cosine_similarity"]
            self.logger.info("Max avg cos sim:{:.6f}, Min avg loss:{:.6f}".format(self.trainer.best_valid_metric, min_loss))
            #self.logger.info('End of epoch %i.\n\n' % n_epoch)
            if n_without_improvement >= self.args.quit_after_n_epochs_without_improvement:
                self.logger.info('After {} epochs without improvement, quiting!'.format(n_without_improvement))
                break

    def svd(self):
        """
        """
        if self.args.load_pred_bert:
            assert self.args.bert_file0 is not None
            assert self.args.bert_file1 is not None
            self.dataset, unique_id_to_feature, self.features = load_from_bert(self.args.vocab_file, self.args.bert_file0,
                self.args.bert_file1, do_lower_case=self.args.do_lower_case, 
                max_seq_length=self.args.max_seq_length, n_max_sent=self.args.n_max_sent,
                vocab_file1=self.args.vocab_file1, align_file=self.args.align_file)

        self.trainer = SupervisedBertTrainer(self.bert_model, self.mapping, self.discriminator, 
                                    self.args, bert_model1=self.bert_model1, trans_types=self.transformer_types)

        sampler = SequentialSampler(self.dataset)
        train_loader = DataLoader(self.dataset, sampler=sampler, batch_size=len(self.dataset))

        self.trainer.args.loss = 'l2_dist'
        for input_embs_a, input_mask_a, input_embs_b, input_mask_b, align_ids_a, align_ids_b, align_mask, example_indices in train_loader:
            self.logger.info("Applying SVD")
            with torch.no_grad():
                input_embs_a = input_embs_a.to(self.device)
                input_mask_a = input_mask_a.to(self.device)
                input_embs_b = input_embs_b.to(self.device)
            align_ids_a = align_ids_a.to(self.device)
            align_ids_b = align_ids_b.to(self.device)
            align_mask = align_mask.to(self.device)
            #print (align_ids_a, align_ids_b, align_mask)
            src_bert = self.trainer.get_indexed_bert_from_bert(
                            input_embs_a, align_ids_a, align_mask, 
                            bert_layer=self.args.bert_layer)
            tgt_bert = self.trainer.get_indexed_bert_from_bert(
                            input_embs_b, align_ids_b, align_mask,
                            bert_layer=self.args.bert_layer)
            avg_cos_sim, loss = self.trainer.supervised_mapping_step(src_bert, tgt_bert, eval_only=True)
            avg_cos_sim_0 = avg_cos_sim.cpu().detach().numpy()
            loss_0 = loss.cpu().detach().numpy()
            self.logger.info("Before mapping: avg cos sim:{:.6f}, avg l2 distance:{:.6f}".format(avg_cos_sim_0, loss_0))
            
            self.trainer.procrustes(src_bert, tgt_bert)
            mapped_src_bert = self.trainer.get_indexed_mapped_bert_from_bert(
                                    input_embs_a, input_mask_a, align_ids_a, align_mask, 
                                    bert_layer=self.args.bert_layer)
            avg_cos_sim, loss = self.trainer.supervised_mapping_step(mapped_src_bert, tgt_bert, eval_only=True)
            avg_cos_sim_1 = avg_cos_sim.cpu().detach().numpy()
            loss_1 = loss.cpu().detach().numpy()
            self.logger.info("After mapping: avg cos sim:{:.6f}, avg l2 distance:{:.6f}".format(avg_cos_sim_1, loss_1))
        self.trainer.save_model(self.args.model_path+'/best_mapping.pkl')

    def list2bert(self, sents):
        """
        Map bert of source language to target space
        """
        assert self.args.output_file is not None

        self.trainer = SupervisedBertTrainer(self.bert_model, self.mapping, self.discriminator, 
                                        self.args, trans_types=self.transformer_types)
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
                    if self.args.map_type in self.transformer_types:
                        target_layer = self.trainer.mapping(src_encoder_layer, input_mask)
                    elif self.args.map_type == 'fine_tune':
                        target_layer = src_encoder_layer
                    else:
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

if __name__ == "__main__":
  main()
