# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import deepcopy
import numpy as np
from numpy.linalg import norm
import os
from torch.autograd import Variable
from torch import Tensor as torch_tensor
from torch.utils.data import Sampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = getLogger()

class SubsetSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class BertEvaluator(object):

    def __init__(self, trainer, features):
        """
        Initialize evaluator.
        """
        self.bert_model = trainer.bert_model
        self.mapping = trainer.mapping
        self.discriminator = trainer.discriminator
        self.params = trainer.params
        self.dataset = trainer.dataset
        self.features = features
        self.dev_sent_num = self.params.dev_sent_num
        assert self.dev_sent_num <= len(self.dataset)
        dev_sampler = SubsetSampler(range(self.dev_sent_num))
        self.dev_loader = DataLoader(self.dataset, sampler=dev_sampler, batch_size=self.params.batch_size)
        logger.info("### Development sentence number: {} ###".format(len(self.dev_sampler)))
        dis_sampler = SequentialSampler(self.dataset)
        self.dis_loader = DataLoader(self.dataset, sampler=dis_sampler, batch_size=self.params.batch_size)

        if self.params.local_rank == -1 or self.params.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.params.no_cuda else "cpu")
        else:
            self.device = torch.device("cuda", self.params.local_rank)

        self.stop_words_a = self.load_stop_words(self.params.stop_words_src)
        self.stop_words_b = self.load_stop_words(self.params.stop_words_tgt)

    def get_bert(self, input_ids, input_mask, bert_layer=-1):
        """
        Get BERT
        """
        self.bert_model.eval()
        with torch.no_grad():
            all_encoder_layers, _ = self.bert_model(input_ids, token_type_ids=None, attention_mask=input_mask)
            encoder_layer = all_encoder_layers[bert_layer]
        
        # [batch_size, seq_len, output_dim]
        return encoder_layer

    def select(self, embed, mask):
        """
        Select all unmasked embed in this batch 
        """
        batch_size, seq_len, emb_dim = list(embed.size())
        return embed.masked_select(mask.view(batch_size, seq_len, 1).expand(-1, -1, emb_dim)).view(-1,emb_dim)

    def load_stop_words(self, file):
        """
        Load stop words
        """
        if os.path.exists(file):
            with open(file, 'r') as fi:
                return fi.read().strip().split('\n')
        else:
            return None

    def rm_stop_words(self, tokens, embs, stop_words):
        """
        Remove stop words
        """
        assert len(tokens) == len(embs)
        if stop_words is None:
            return tokens, embs
        new_toks = []
        new_embs = []
        for tok, emb in zip(tokens, embs):
            if tok not in stop_words:
                new_toks.append(tok)
                new_embs.append(emb)
        return new_toks, new_embs

    def cos_sim(self, a, b):
        return np.inner(a, b)/(norm(a)*norm(b))

    def sent_sim(self, rm_stop_words=True):
        """
        Run all evaluations.
        """
        similarities = []
        for input_ids_a, input_mask_a, input_ids_b, input_mask_b, example_indices in self.dev_loader:

            src_bert = self.get_bert(input_ids_a.to(self.device), input_mask_a.to(self.device), 
                                    bert_layer=self.params.bert_layer).data.cpu().numpy()
            tgt_bert = self.get_bert(input_ids_b.to(self.device), input_mask_b.to(self.device), 
                                    bert_layer=self.params.bert_layer).data.cpu().numpy()
            
            for i, example_index in enumerate(example_indices):
                feature = self.features[example_index.item()]
                seq_len_a = np.sum(input_mask_a[i])
                seq_len_b = np.sum(input_mask_b[i])
                # [seq_len, output_dim]
                src_emb = src_bert[i][:seq_len_a]
                tgt_emb = tgt_bert[i][:seq_len_b]
                if rm_stop_words:
                    src_emb, src_toks = self.rm_stop_words(feature.tokens_a, src_emb, self.stop_words_a)
                    tgt_emb, tgt_toks = self.rm_stop_words(feature.tokens_b, tgt_emb, self.stop_words_b)
                similarities.append(self.cos_sim(np.mean(src_emb, 0), np.mean(tgt_emb, 0)))
        sim_mean = np.mean(similarities)
        logger.info("### Mean sentence similarity: {:.2f}% ###".format(sim_mean*100))

        return sim_mean

    def eval_dis(self):
        """
        Evaluate discriminator predictions and accuracy.
        """
        src_preds = []
        tgt_preds = []

        self.discriminator.eval()

        for input_ids_a, input_mask_a, input_ids_b, input_mask_b, example_indices in self.dis_loader:

            src_bert = self.get_bert(input_ids_a.to(self.device), input_mask_a.to(self.device), 
                                    bert_layer=self.params.bert_layer)
            tgt_bert = self.get_bert(input_ids_b.to(self.device), input_mask_b.to(self.device), 
                                    bert_layer=self.params.bert_layer)
            src_preds.extend(self.discriminator(self.mapping(self.select(src_bert, input_mask_a))).data.cpu().tolist())
            tgt_preds.extend(self.discriminator(self.select(tgt_bert, input_mask_b)).data.cpu().tolist())

        src_pred = np.mean(src_preds)
        tgt_pred = np.mean(tgt_preds)
        logger.info("Discriminator source / target predictions: {:.2f}% / {.2f}%".format(src_pred*100, tgt_pred*100))

        src_acc = np.mean([x >= 0.5 for x in src_preds])
        tgt_acc = np.mean([x < 0.5 for x in tgt_preds])
        dis_acc = ((src_acc * len(src_preds) + tgt_acc * len(tgt_preds)) /
                    (len(src_preds) + len(tgt_preds)))
        logger.info("Discriminator source / target / global accuracy: {:.2f}% / {:.2f}% / {:.2f}%".format(src_acc*100, tgt_acc*100, dis_acc*100))

        #to_log['dis_acc'] = dis_acc
        #to_log['dis_src_pred'] = src_pred
        #to_log['dis_tgt_pred'] = tgt_pred