# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import random

from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary


logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params

        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            print ("Map optimizer parameters: ",optim_params)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        else:
            assert discriminator is None

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False
        self.decrease_dis_lr = False

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf == 0 else mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        with torch.no_grad():
            src_emb = self.src_emb(Variable(src_ids))
            tgt_emb = self.tgt_emb(Variable(tgt_ids))
        if volatile:
            with torch.no_grad():
                src_emb = self.mapping(Variable(src_emb.data))
                tgt_emb = Variable(tgt_emb.data)
        else:
            src_emb = self.mapping(Variable(src_emb.data))
            tgt_emb = Variable(tgt_emb.data)

        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = Variable(y.cuda() if self.params.cuda else y)

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.item())

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()

        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        self.orthogonalize()

        return 2 * self.params.batch_size

    def get_aligned_id_batchs(self, shuffle=True):
        """
        Get aligned ids by batch
        """
        batches = []
        # ids for aligned embeddings
        ids = np.arange(self.dico.shape[0])
        if shuffle:
            random.shuffle(ids)
        bs = self.params.batch_size
        assert bs <= min(len(self.src_dico), len(self.tgt_dico))
        for offset in range(0, len(ids), bs):
            if offset+bs <= len(ids):
                src_ids = ids[offset:offset+bs]
                tgt_ids = ids[offset:offset+bs]
            else:
                src_ids = ids[offset:]
                tgt_ids = ids[offset:]
            batches.append((src_ids, tgt_ids))
        return batches

    def get_aligned_embs(self, src_ids, tgt_ids):
        """
        Get aligned embeddings.
        """

        src_ids = self.dico[src_ids, 0]
        tgt_ids = self.dico[tgt_ids, 1]
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        with torch.no_grad():
            src_emb = self.src_emb(src_ids)
            tgt_emb = self.tgt_emb(tgt_ids)
        
        src_emb = self.mapping(src_emb)

        return src_emb, tgt_emb

    def supervised_mapping_step(self, src_ids, tgt_ids, margin=1):
        """
        Fooling discriminator training step.
        """

        # loss
        src_emb, tgt_emb = self.get_aligned_embs(src_ids, tgt_ids)

        # normalization
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        scores = src_emb.mm(tgt_emb.transpose(0, 1))

        rang = torch.arange(scores.shape[0], out=torch.LongTensor())
        # (n)
        gold_scores = scores[rang, rang]
        avg_cos_sim = gold_scores.mean()

        if self.params.loss == 'cos_sim':
            # maximize cosine similarities
            loss = - gold_scores.mean()
        elif self.params.loss.startswith('max_margin_top'):
            # max margin with top k elements
            k = int(self.params.loss.split('-')[1])
            # (n, k)
            top_vals, top_ids = scores.topk(k, 1, True)
            # (n) => (n, k)
            gold_vals = gold_scores.unsqueeze(1).expand_as(top_vals)
            # (n, k)
            margins = torch.ones_like(top_vals) * margin
            # (n, k)
            losses = margins - gold_vals + top_vals
            # mask out less than 0
            losses = torch.where(losses>0, losses, torch.zeros_like(losses))
            # ()
            loss = losses.mean()

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        #self.orthogonalize()

        return avg_cos_sim, loss

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        if isinstance(self.mapping, torch.nn.DataParallel):
            W = self.mapping.module.weight.data
        else:
            W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))
        logger.info("Finished Procrustes.")

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            if isinstance(self.mapping, torch.nn.DataParallel):
                W = self.mapping.module.weight.data
            else:
                W = self.mapping.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def decay_map_lr(self):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing Mapping learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def update_dis_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.dis_optimizer:
            return
        old_lr = self.dis_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing discriminator learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.dis_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_dis_lr:
                    old_lr = self.dis_optimizer.param_groups[0]['lr']
                    self.dis_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the discriminator learning rate: %.5f -> %.5f"
                                % (old_lr, self.dis_optimizer.param_groups[0]['lr']))
                self.decrease_dis_lr = True

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            if isinstance(self.mapping, torch.nn.DataParallel):
                W = self.mapping.module.weight.data.cpu().numpy()
            else:
                W = self.mapping.weight.data.cpu().numpy()
            path = os.path.join(self.params.model_path, 'best_mapping.pkl')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(W, path)

    def save_iter(self, iter):
        """
        Save the current model.
        """
        # best mapping for the given validation criterion
        # save the mapping
        if isinstance(self.mapping, torch.nn.DataParallel):
            W = self.mapping.module.weight.data.cpu().numpy()
        else:
            W = self.mapping.weight.data.cpu().numpy()
        path = os.path.join(self.params.model_path, 'iter-'+str(iter) , 'best_mapping.pkl')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        logger.info('* Saving the mapping to %s ...' % path)
        torch.save(W, path)

    def save_model(self, path):
        """
        Save the current model.
        """
        # best mapping for the given validation criterion
        # save the mapping
        if isinstance(self.mapping, torch.nn.DataParallel):
            W = self.mapping.module.weight.data.cpu().numpy()
        else:
            W = self.mapping.weight.data.cpu().numpy()
        path = os.path.join(path, 'best_mapping.pkl')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        logger.info('* Saving the mapping to %s ...' % path)
        torch.save(W, path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.model_path, 'best_mapping.pkl')
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        if isinstance(self.mapping, torch.nn.DataParallel):
            W = self.mapping.module.weight.data
        else:
            W = self.mapping.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        # map source embeddings to the target space
        bs = 4096
        logger.info("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(src_emb), bs)):
            with torch.no_grad():
                x = Variable(src_emb[k:k + bs])
            src_emb[k:k + bs] = self.mapping(x.cuda() if params.cuda else x).data.cpu()

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)
