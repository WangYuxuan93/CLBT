# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch import Tensor as torch_tensor

logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.bert_model = trainer.bert_model
        self.mapping = trainer.mapping
        self.discriminator = trainer.discriminator
        self.params = trainer.params
        self.dataset = trainer.dataset

    def sent_sim(self, to_log):
        """
        Run all evaluations.
        """
        return

    def eval_dis(self, to_log):
        """
        Evaluate discriminator predictions and accuracy.
        """
        bs = 128
        src_preds = []
        tgt_preds = []

        self.discriminator.eval()

        for i in range(0, self.src_emb.num_embeddings, bs):
            emb = Variable(self.src_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator(self.mapping(emb))
            src_preds.extend(preds.data.cpu().tolist())

        for i in range(0, self.tgt_emb.num_embeddings, bs):
            emb = Variable(self.tgt_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator(emb)
            tgt_preds.extend(preds.data.cpu().tolist())

        src_pred = np.mean(src_preds)
        tgt_pred = np.mean(tgt_preds)
        logger.info("Discriminator source / target predictions: %.5f / %.5f"
                    % (src_pred, tgt_pred))

        src_accu = np.mean([x >= 0.5 for x in src_preds])
        tgt_accu = np.mean([x < 0.5 for x in tgt_preds])
        dis_accu = ((src_accu * self.src_emb.num_embeddings + tgt_accu * self.tgt_emb.num_embeddings) /
                    (self.src_emb.num_embeddings + self.tgt_emb.num_embeddings))
        logger.info("Discriminator source / target / global accuracy: %.5f / %.5f / %.5f"
                    % (src_accu, tgt_accu, dis_accu))

        to_log['dis_accu'] = dis_accu
        to_log['dis_src_pred'] = src_pred
        to_log['dis_tgt_pred'] = tgt_pred