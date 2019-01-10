# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import argparse
from collections import OrderedDict
import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_supervised_model
from src.trainer import Trainer
from src.evaluation import Evaluator


VALIDATION_METRIC_SUP = 'precision_at_1-csls_knn_10'
VALIDATION_METRIC_UNSUP = 'mean_cosine-csls_knn_10-S2T-10000'

def main():
    # main
    parser = argparse.ArgumentParser(description='Supervised training')
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--model_path", type=str, default="", help="Where to store experiment logs and models")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
    parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
    # supervised sgd learning
    parser.add_argument("--n_epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1,weight_decay=0.01", help="self.mapping optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.95, help="Learning rate decay (SGD only)") 
    parser.add_argument("--decay_step", type=int, default=100, help="Learning rate decay step (SGD only)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
    parser.add_argument("--quit_after_n_epochs_without_improvement", type=int, default=500, help="Quit after n epochs without improvement")
    parser.add_argument("--normalize_embed", type=bool_flag, default=False, help="Normalize embeddings? (should be false with l2_dist loss)")
    parser.add_argument("--loss", type=str, default="cos_sim", help="loss type (cos_sim, max_margin_top-k, l2_dist)")
    # data
    parser.add_argument("--src_lang", type=str, default='en', help="Source language")
    parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
    # training refinement
    parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
    # dictionary creation parameters (for refinement)
    parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default: use identical character strings)")
    parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
    parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
    parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
    parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
    parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
    parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
    parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
    # reload pre-trained embeddings
    parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
    parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
    parser.add_argument("--test", action='store_true', default=False, help="Predict cosine similarity & L2 distance with input model")
    parser.add_argument("--save_all", action='store_true', default=False, help="Save every model")
    parser.add_argument("--map_beta", type=float, default=0.01, help="Beta for orthogonalization")
    parser.add_argument("--ortho", action='store_true', default=False, help="Apply orthogonalize after each update")
    parser.add_argument("--non_linear", action='store_true', default=False, help="Use non-linear mapping")
    parser.add_argument("--activation", type=str, default='leaky_relu', help="learky_relu,tanh")
    parser.add_argument("--n_layers", type=int, default=1, help="mapping layer")
    parser.add_argument("--hidden_size", type=int, default=768, help="mapping hidden layer size")
    # parse parameters
    params = parser.parse_args()

    supmap = SupervisedMap(params)

    if params.test:
        supmap.test()
        exit()

    supmap.train()

class SupervisedMap(object):

    def __init__(self, params):

        self.params = params
        # check parameters
        assert not params.cuda or torch.cuda.is_available()
        assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
        assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
        assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
        assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
        assert os.path.isfile(params.src_emb)
        assert os.path.isfile(params.tgt_emb)
        assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
        assert params.export in ["", "txt", "pth"]

        # build self.logger / model / self.trainer / evaluator
        self.logger = initialize_exp(params)
        src_emb, tgt_emb, mapping, _ = build_supervised_model(params, False)
        self.trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
        evaluator = Evaluator(self.trainer)

        # load a training dictionary. if a dictionary path is not provided, use a default
        # one ("default") or create one based on identical character strings ("identical_char")
        self.trainer.load_training_dico(params.dico_train)

        # define the validation metric
        #VALIDATION_METRIC = VALIDATION_METRIC_UNSUP if params.dico_train == 'identical_char' else VALIDATION_METRIC_SUP
        #self.logger.info("Validation metric: %s" % VALIDATION_METRIC)

    def train(self):
        """
        Learning loop for Procrustes Iterative Learning
        """
        n_without_improvement = 0
        min_loss = 1e6
        path4loss = self.params.model_path + '/model4loss'
        if not os.path.exists(path4loss):
            os.makedirs(path4loss)
        if self.params.save_all:
            model_log = open(path4loss+'/model.log', 'w')
        for n_epoch in range(self.params.n_epochs):

            #self.logger.info('Starting epoch %i...' % n_epoch)
            if (n_epoch+1) % self.params.decay_step == 0:
                self.trainer.decay_map_lr()
            batches = self.trainer.get_aligned_id_batchs()
            n_inst = 0
            to_log = {"avg_cosine_similarity": 0, "loss": 0}
            for i, (src_ids, tgt_ids) in enumerate(batches):
                #print (src_ids, tgt_ids)
                avg_cos_sim, loss = self.trainer.supervised_mapping_step(src_ids, tgt_ids)
                n_inst += len(src_ids)
                cos_sim = avg_cos_sim.cpu().detach().numpy()
                loss_ = loss.cpu().detach().numpy()
                #self.logger.info("Step:{}, Total Instances:{}, Cosine Similarity:{:.6f}, Loss:{:.6f}".format(i, 
                #            n_inst, cos_sim, loss_))
                to_log["avg_cosine_similarity"] += cos_sim
                to_log["loss"] += loss_
            to_log["avg_cosine_similarity"] /= len(batches)
            to_log["loss"] /= len(batches)
            self.logger.info("Epoch:{}, avg cos sim:{:.6f}, avg loss:{:.6f}, instances:{}".format(n_epoch, to_log["avg_cosine_similarity"], to_log["loss"], n_inst))

            if to_log["avg_cosine_similarity"] <= self.trainer.best_valid_metric and to_log["loss"] >= min_loss:
                n_without_improvement += 1
            else:
                n_without_improvement = 0
            if to_log["loss"] < min_loss:
                self.logger.info(" Minimum loss : {:.6f}".format(to_log["loss"]))
                if self.params.save_all:
                    save_path = path4loss+'/epoch-'+str(n_epoch)
                    model_log.write("Epoch:{}, avg cos sim:{:.6f}, avg loss:{:.6f}\n".format(n_epoch, 
                                    to_log["avg_cosine_similarity"], to_log["loss"]))
                else:
                    save_path = path4loss
                self.trainer.save_model(save_path+'/best_mapping.pkl')
                min_loss = to_log["loss"]
            self.trainer.save_best(to_log, "avg_cosine_similarity") 
            self.logger.info("Max avg cos sim:{:.6f}, Min avg loss:{:.6f}".format(self.trainer.best_valid_metric, min_loss))
            #self.logger.info('End of epoch %i.\n\n' % n_epoch)
            if n_without_improvement >= self.params.quit_after_n_epochs_without_improvement:
                self.logger.info('After {} epochs without improvement, quiting!'.format(n_without_improvement))
                break

        # export embeddings
        if self.params.export:
            self.trainer.reload_best()
            self.trainer.export()

    def test(self):
        #self.trainer.reload_best()
        self.trainer.load_best()
        batches = self.trainer.get_aligned_id_batchs(shuffle=False)
        n_inst = 0
        to_log = {"avg_cosine_similarity": 0, "loss": 0}
        for i, (src_ids, tgt_ids) in enumerate(batches):
            #print (src_ids, tgt_ids)
            with torch.no_grad():
                avg_cos_sim, loss = self.trainer.supervised_mapping_step(src_ids, tgt_ids)
            n_inst += len(src_ids)
            cos_sim = avg_cos_sim.cpu().detach().numpy()
            loss_ = loss.cpu().detach().numpy()
            self.logger.info("Step:{}, Total Instances:{}, Cosine Similarity:{:.6f}, Loss:{:.6f}".format(i, 
                        n_inst, cos_sim, loss_))
            to_log["avg_cosine_similarity"] += cos_sim
            to_log["loss"] += loss_
        to_log["avg_cosine_similarity"] /= len(batches)
        to_log["loss"] /= len(batches)

        self.logger.info("Average Cosine Similarity:{:.6f}, Average Loss:{:.6f}".format(to_log["avg_cosine_similarity"], to_log["loss"]))

if __name__ == '__main__':
    main()
