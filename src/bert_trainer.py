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

from src.utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from src.utils import clip_parameters
from src.evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import _DataLoaderIter

logger = getLogger()


class BertTrainer(object):

    def __init__(self, bert_model, dataset, mapping, discriminator, args):
        """
        Initialize trainer script.
        """
        self.bert_model = bert_model
        if args.adversarial:
            self.dataset = dataset
            #sampler = SequentialSampler(dataset)
            sampler = RandomSampler(dataset)
            self.dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
            self.iter_loader = _DataLoaderIter(self.dataloader)
        self.mapping = mapping
        self.discriminator = discriminator
        self.args = args

        if self.args.local_rank == -1 or self.args.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        else:
            self.device = torch.device("cuda", self.args.local_rank)

        # optimizers
        if hasattr(args, 'map_optimizer'):
            optim_fn, optim_args = get_optimizer(args.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_args)
        if hasattr(args, 'dis_optimizer'):
            optim_fn, optim_args = get_optimizer(args.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_args)
        else:
            assert discriminator is None

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False
        self.decrease_dis_lr = False

    def get_mapping_xy(self):
        """
        Get x/y for mapping step
        """
        items = next(self.iter_loader, None)
        if items is None:
            self.iter_loader = _DataLoaderIter(self.dataloader)
            items = next(self.iter_loader, None)
        input_ids_a, input_mask_a, input_ids_b, input_mask_b, example_indices = items
        src_emb = self.get_bert(input_ids_a.to(self.device), input_mask_a.to(self.device), bert_layer=self.args.bert_layer)
        tgt_emb = self.get_bert(input_ids_b.to(self.device), input_mask_b.to(self.device), bert_layer=self.args.bert_layer)
        src_emb = self.mapping(src_emb)
        src_len = src_emb.size(0)
        tgt_len = tgt_emb.size(0)

        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(src_len+tgt_len).zero_()
        y[:src_len] = 1 - self.args.dis_smooth
        y[src_len:] = self.args.dis_smooth

        return x,y

    def get_bert(self, input_ids, input_mask, bert_layer=-1):
        """
        Get BERT
        """
        self.bert_model.eval()
        with torch.no_grad():
            all_encoder_layers, _ = self.bert_model(input_ids, token_type_ids=None, attention_mask=input_mask)
            encoder_layer = all_encoder_layers[bert_layer]
        
        # [batch_size, seq_len, output_dim] => [unmasked_len, output_dim]
        return self.select(encoder_layer, input_mask)

    def select(self, embed, mask):
        """
        Select all unmasked embed in this batch 
        """
        batch_size, seq_len, emb_dim = list(embed.size())
        return embed.masked_select(mask.byte().view(batch_size, seq_len, 1).expand(-1, -1, emb_dim)).view(-1,emb_dim)

    def dis_step(self, src_emb, tgt_emb, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()
        src_len = src_emb.size(0)
        tgt_len = tgt_emb.size(0)

        with torch.no_grad():
            src_emb = self.mapping(src_emb)

        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(src_len+tgt_len).zero_()
        y[:src_len] = 1 - self.args.dis_smooth
        y[src_len:] = self.args.dis_smooth

        # loss
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y.to(self.device))
        stats['DIS_COSTS'].append(loss.item())

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.args.dis_clip_weights)
        self.dis_optimizer.step()
        
        #clip_parameters(self.discriminator, self.args.dis_clip_weights)

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.args.dis_lambda == 0:
            return 0

        self.discriminator.eval()

        # loss
        x, y = self.get_mapping_xy()
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y.to(self.device))
        loss = self.args.dis_lambda * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mapping.parameters(), self.args.map_clip_weights)
        self.map_optimizer.step()
        self.orthogonalize()

        return x.size(0)

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.args.map_beta > 0:
            if isinstance(self.mapping, torch.nn.DataParallel):
                W = self.mapping.module.weight.data
            else:
                W = self.mapping.weight.data
            beta = self.args.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, sent_sim):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.args.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.args.min_lr, old_lr * self.args.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing map learning rate: {:.8f} -> {:.8f}".format(old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.args.lr_shrink < 1 and sent_sim >= -1e7:
            if sent_sim < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: {:.5f} vs {:.5f}".format(sent_sim, self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.args.lr_shrink
                    logger.info("Shrinking map learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def update_dis_lr(self, sent_sim):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.args.dis_optimizer:
            return
        old_lr = self.dis_optimizer.param_groups[0]['lr']
        new_lr = max(self.args.min_lr, old_lr * self.args.dis_lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing discriminator learning rate: {:.8f} -> {:.8f}".format(old_lr, new_lr))
            self.dis_optimizer.param_groups[0]['lr'] = new_lr

        #if self.args.lr_shrink < 1 and sent_sim >= -1e7:
        #    if sent_sim < self.best_valid_metric:
                #logger.info("Validation metric is smaller than the best: {:.5f} vs {:.5f}".format(sent_sim, self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
        #        if self.decrease_dis_lr:
        #            old_lr = self.dis_optimizer.param_groups[0]['lr']
        #            self.dis_optimizer.param_groups[0]['lr'] *= self.args.lr_shrink
        #            logger.info("Shrinking discriminator learning rate: %.5f -> %.5f"
        #                        % (old_lr, self.dis_optimizer.param_groups[0]['lr']))
        #        self.decrease_dis_lr = True

    def save_best(self, sent_sim):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if sent_sim > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = sent_sim
            logger.info('### New record (sentence similarity): {:.2f}% ###'.format(sent_sim*100))
            # save the mapping
            if isinstance(self.mapping, torch.nn.DataParallel):
                W = self.mapping.module.weight.data.cpu().numpy()
            else:
                W = self.mapping.weight.data.cpu().numpy()
            path = os.path.join(self.args.model_path, 'best_mapping.pkl')
            logger.info('### Saving the mapping to %s ... ###' % path)
            torch.save(W, path)
            if self.args.save_dis:
                torch.save(self.discriminator.state_dict(), os.path.join(self.args.model_path, 'discriminator.pkl'))

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.args.model_path, 'best_mapping.pkl')
        logger.info('### Reloading the best model from %s ... ###' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))
