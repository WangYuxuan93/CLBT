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
import string
import os
import torch
from torch.autograd import Variable
from torch import Tensor as torch_tensor
from torch.utils.data import TensorDataset, Sampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from collections import Counter
from src.bert_trainer import reload_model

logger = getLogger()

class SubsetSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def load_stop_words(file):
    """
    Load stop words
    """
    if os.path.exists(file):
        with open(file, 'r') as fi:
            return fi.read().strip().split('\n')
    else:
        logger.info("### Stop word file {} does not exist! ###".format(file))
        return ""

def rm_stop_words(tokens, embs, stop_words, puncs):
    """
    Remove stop words
    """
    assert len(tokens) == len(embs)
    if stop_words is None and puncs is None:
        return tokens, embs
    new_toks = []
    new_embs = []
    for tok, emb in zip(tokens, embs):
        if tok not in stop_words and tok not in puncs:
            new_toks.append(tok)
            new_embs.append(emb)
    return new_toks, np.array(new_embs)

def get_overlaps(src_toks, src_embs, tgt_toks, tgt_embs):
    """
    Get overlaps
    """
    assert len(src_toks) == len(src_embs)
    assert len(tgt_toks) == len(tgt_embs)
    src_cnt = Counter(src_toks)
    tgt_cnt = Counter(tgt_toks)

    overlaps = []
    for src_id, (tok, src_emb) in enumerate(zip(src_toks, src_embs)):
        # only return overlap tokens that are unique in both source and target
        if not tok == '[UNK]' and src_cnt[tok] == 1 and tok in tgt_toks and tgt_cnt[tok] == 1:
            tgt_id = tgt_toks.index(tok)
            overlap = {'src_id':src_id, 'tgt_id':tgt_id, 'token':tok,
                        'src_emb':src_emb, 'tgt_emb':tgt_embs[tgt_id]}
            overlaps.append(overlap)
    return overlaps

def get_overlap_sim(overlaps):
    """
    Calculate overlap token similarities
    """
    similarities = []
    infos = []
    for pair in overlaps:
        tok_sim = cos_sim(pair['src_emb'], pair['tgt_emb'])
        similarities.append({'src_id':pair['src_id'], 'tgt_id':pair['tgt_id'],
                            'sim':tok_sim})
        infos.append('token:\"{}\" (src_id:{}, tgt_id:{}), sim:{:.4f}'.format(pair['token'], pair['src_id'], pair['tgt_id'], tok_sim))
    return similarities, infos

def cos_sim(a, b):
    return np.inner(a, b)/(norm(a)*norm(b))

class BertEvaluator(object):

    def __init__(self, bert_model, dataset, mapping, discriminator, 
                    args, features, bert_model1=None ):
        """
        Initialize evaluator.
        """
        self.bert_model = bert_model
        self.bert_model1 = bert_model1
        self.mapping = mapping
        self.discriminator = discriminator
        self.args = args
        self.dataset = dataset
        self.features = features
        self.dev_sent_num = self.args.dev_sent_num
        if self.args.adversarial:
            assert self.dev_sent_num <= len(self.dataset)
            dev_sampler = SubsetSampler(range(self.dev_sent_num))
            self.dev_loader = DataLoader(self.dataset, sampler=dev_sampler, batch_size=self.args.batch_size)
            logger.info("### Development sentence number: {} ###".format(len(dev_sampler)))
            if self.args.eval_non_parallel:
                self.nonpara_loader = self.get_nonpara_loader()
        #dis_sampler = SequentialSampler(self.dataset)
        #self.dis_loader = DataLoader(self.dataset, sampler=dis_sampler, batch_size=self.args.batch_size)

        if self.args.local_rank == -1 or self.args.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        else:
            self.device = torch.device("cuda", self.args.local_rank)
	# "!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        self.punc = ""
        self.stop_words_a = ""
        self.stop_words_b = ""
        if self.args.rm_punc:
            self.punc = string.punctuation
        if self.args.rm_stop_words:
            self.stop_words_a = load_stop_words(self.args.stop_words_src)
            self.stop_words_b = load_stop_words(self.args.stop_words_tgt)

    def get_nonpara_loader(self):
        """
        Get nonparaed loader for penalty sent sim
        """
        num = self.dev_sent_num
        all_input_ids_a = torch.tensor([f.input_ids_a for f in self.features[:num]], dtype=torch.long)
        all_input_ids_b = torch.tensor([f.input_ids_b for f in self.features[:num]], dtype=torch.long)
        all_input_mask_a = torch.tensor([f.input_mask_a for f in self.features[:num]], dtype=torch.long)
        all_input_mask_b = torch.tensor([f.input_mask_b for f in self.features[:num]], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids_a.size(0), dtype=torch.long)

        # move each b sentence to the previous a, move the first b to the last
        ids_b_0 = all_input_ids_b[:1]
        all_input_ids_b_ = all_input_ids_b[1:]
        all_input_ids_b = torch.cat([all_input_ids_b_, ids_b_0], 0)
        mask_b_0 = all_input_mask_b[:1]
        all_input_mask_b_ = all_input_mask_b[1:]
        all_input_mask_b = torch.cat([all_input_mask_b_, mask_b_0], 0)

        nonpara_dataset = TensorDataset(all_input_ids_a, all_input_mask_a, 
                        all_input_ids_b, all_input_mask_b, all_example_index)
        nonpara_sampler = SequentialSampler(nonpara_dataset)
        nonpara_loader = DataLoader(nonpara_dataset, sampler=nonpara_sampler, 
                                        batch_size=self.args.batch_size)
        return nonpara_loader

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

    def select(self, embed, mask):
        """
        Select all unmasked embed in this batch 
        """
        batch_size, seq_len, emb_dim = list(embed.size())
        return embed.masked_select(mask.byte().view(batch_size, seq_len, 1).expand(-1, -1, emb_dim)).view(-1,emb_dim)

    def eval_sim(self):

        metrics = {"para_sim":self.parallel_sim()}
        if self.args.eval_non_parallel:
            metrics["nonpara_sim"] = self.nonpara_sim()
        return metrics

    def parallel_sim(self):

        return self.sent_sim(self.dev_loader, "parallel")

    def nonpara_sim(self):

        return self.sent_sim(self.nonpara_loader, "non_parallel")

    def sent_sim(self, loader, type):
        """
        Run all evaluations.
        """
        similarities = []
        fo = open(self.args.model_path+'/'+type+"_similarities.txt" ,'w')
        for input_ids_a, input_mask_a, input_ids_b, input_mask_b, example_indices in loader:

            src_bert = self.get_bert(input_ids_a.to(self.device), input_mask_a.to(self.device), 
                                    bert_layer=self.args.bert_layer, model_id=0)#.data.cpu().numpy()
            tgt_bert = self.get_bert(input_ids_b.to(self.device), input_mask_b.to(self.device), 
                                    bert_layer=self.args.bert_layer, model_id=1).data.cpu().numpy()
            src_bert = self.mapping(src_bert).data.cpu().numpy()
            for i, example_index in enumerate(example_indices):
                feature = self.features[example_index.item()]
                tokens_a = feature.tokens_a
                if type == "parallel":
                    tokens_b = feature.tokens_b
                else:
                    real_id = example_index.item()+1 if example_index.item()+1 < self.dev_sent_num else 0
                    tokens_b = self.features[real_id].tokens_b
                seq_len_a = np.sum(input_mask_a[i].data.cpu().numpy())
                seq_len_b = np.sum(input_mask_b[i].data.cpu().numpy())
                # [seq_len, output_dim]
                src_emb = src_bert[i][1:seq_len_a-1]
                tgt_emb = tgt_bert[i][1:seq_len_b-1]
                if self.args.rm_stop_words or self.args.rm_punc:
                    src_toks, src_emb = rm_stop_words(tokens_a[1:-1], src_emb, self.stop_words_a, self.punc)
                    tgt_toks, tgt_emb = rm_stop_words(tokens_b[1:-1], tgt_emb, self.stop_words_b, self.punc)
                if len(src_emb) == 0 or len(tgt_emb) == 0:
                    continue
                similarities.append(cos_sim(np.mean(src_emb, 0), np.mean(tgt_emb, 0)))
                fo.write('sim:'+str(similarities[-1])+'\n'+' '.join(tokens_a)+' ||| '+' '.join(tokens_b)+'\n')
        #print ("sent sim:", similarities)
        sim_mean = np.mean(similarities)
        logger.info("Mean {} sentence similarity: {:.2f}% ".format(type, sim_mean*100))
        
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
                                    bert_layer=self.args.bert_layer, model_id=0)
            tgt_bert = self.get_bert(input_ids_b.to(self.device), input_mask_b.to(self.device), 
                                    bert_layer=self.args.bert_layer, model_id=1)
            src_preds.extend(self.discriminator(self.mapping(self.select(src_bert, input_mask_a.to(self.device)))).data.cpu().tolist())
            tgt_preds.extend(self.discriminator(self.select(tgt_bert, input_mask_b.to(self.device))).data.cpu().tolist())

        src_pred = np.mean(src_preds)
        tgt_pred = np.mean(tgt_preds)
        logger.info("Discriminator source / target predictions: {:.2f}% / {:.2f}%".format(src_pred*100, tgt_pred*100))

        src_acc = np.mean([x >= 0.5 for x in src_preds])
        tgt_acc = np.mean([x < 0.5 for x in tgt_preds])
        dis_acc = ((src_acc * len(src_preds) + tgt_acc * len(tgt_preds)) /
                    (len(src_preds) + len(tgt_preds)))
        logger.info("Discriminator source / target / global accuracy: {:.2f}% / {:.2f}% / {:.2f}%".format(src_acc*100, tgt_acc*100, dis_acc*100))

        #to_log['dis_acc'] = dis_acc
        #to_log['dis_src_pred'] = src_pred
        #to_log['dis_tgt_pred'] = tgt_pred

    def eval_dev_dis(self):
        """
        Evaluate discriminator predictions and accuracy.
        """
        src_preds = []
        tgt_preds = []

        self.discriminator.eval()
        for input_ids_a, input_mask_a, input_ids_b, input_mask_b, example_indices in self.dev_loader:
            src_bert = self.get_bert(input_ids_a.to(self.device), input_mask_a.to(self.device), 
                                    bert_layer=self.args.bert_layer, model_id=0)
            tgt_bert = self.get_bert(input_ids_b.to(self.device), input_mask_b.to(self.device), 
                                    bert_layer=self.args.bert_layer, model_id=1)
            src_preds.extend(self.discriminator(self.mapping(self.select(src_bert, input_mask_a.to(self.device)))).data.cpu().tolist())
            tgt_preds.extend(self.discriminator(self.select(tgt_bert, input_mask_b.to(self.device))).data.cpu().tolist())

        src_pred = np.mean(src_preds)
        tgt_pred = np.mean(tgt_preds)
        logger.info("Discriminator source / target predictions: {:.2f}% / {:.2f}%".format(src_pred*100, tgt_pred*100))

        src_acc = np.mean([x >= 0.5 for x in src_preds])
        tgt_acc = np.mean([x < 0.5 for x in tgt_preds])
        dis_acc = ((src_acc * len(src_preds) + tgt_acc * len(tgt_preds)) /
                    (len(src_preds) + len(tgt_preds)))
        logger.info("Discriminator source / target / global accuracy: {:.2f}% / {:.2f}% / {:.2f}%".format(src_acc*100, tgt_acc*100, dis_acc*100))

    def calculate_sim(self, loader):
        """
        Calculate similarities
        """

        #print ('----> Calculate Sentence Similarity <----\n\n')

        n_sent = 0
        if self.args.sim_with_map:
            reload_model(self.mapping, self.args.model_path)
        outfile = self.args.sim_file if self.args.sim_file else 'similarities.txt'
        similarities = []
        with open(outfile ,'w') as fo:
            for input_ids_a, input_mask_a, input_ids_b, input_mask_b, example_indices in loader:
                input_ids_a = input_ids_a.to(self.device)
                input_mask_a = input_mask_a.to(self.device)
                input_ids_b = input_ids_b.to(self.device)
                input_mask_b = input_mask_b.to(self.device)

                if self.args.base_embed:
                    src_bert = self.bert_model.module.embeddings(input_ids_a, None)
                    tgt_bert = self.bert_model1.module.embeddings(input_ids_b, None).data.cpu().numpy()
                else:
                    src_bert = self.get_bert(input_ids_a, input_mask_a, 
                                        bert_layer=self.args.bert_layer, model_id=0)
                    tgt_bert = self.get_bert(input_ids_b, input_mask_b, 
                                        bert_layer=self.args.bert_layer, model_id=1).data.cpu().numpy()
                if self.args.sim_with_map:
                    src_bert = self.mapping(src_bert)
                src_bert = src_bert.data.cpu().numpy()

                for i, example_index in enumerate(example_indices):
                    n_sent += 1
                    if n_sent % 1000 == 0:
                        print ("\r{}".format(n_sent),end="")
                    feature = self.features[example_index.item()]
                    seq_len_a = np.sum(input_mask_a[i].data.cpu().numpy())
                    seq_len_b = np.sum(input_mask_b[i].data.cpu().numpy())
                    # [seq_len, output_dim]
                    src_emb = src_bert[i][1:seq_len_a-1]
                    tgt_emb = tgt_bert[i][1:seq_len_b-1]
                    if self.args.rm_stop_words or self.args.rm_punc:
                        src_toks, src_emb = rm_stop_words(feature.tokens_a[1:-1], src_emb, self.stop_words_a, self.punc)
                        tgt_toks, tgt_emb = rm_stop_words(feature.tokens_b[1:-1], tgt_emb, self.stop_words_b, self.punc)
                    else:
                        src_toks = feature.tokens_a[1:-1]
                        tgt_toks = feature.tokens_b[1:-1]
                    # calculate overlap token sim
                    if self.args.overlap_sim:
                        overlaps = get_overlaps(src_toks, src_emb, tgt_toks, tgt_emb)
                        if not overlaps:
                            continue
                        sims, infos = get_overlap_sim(overlaps)
                        similarities.extend([s['sim'] for s in sims])
                        fo.write(' | '.join(infos)+'\n'+' '.join(src_toks)+' ||| '+' '.join(tgt_toks)+'\n')
                    # calculate sent sim
                    else:
                        if len(src_emb) == 0 or len(tgt_emb) == 0:
                            continue
                        similarities.append(cos_sim(np.mean(src_emb, 0), np.mean(tgt_emb, 0)))
                        fo.write('sim:'+str(similarities[-1])+'\n'+' '.join(feature.tokens_a)+' ||| '+' '.join(feature.tokens_b)+'\n')
            sim_mean = np.mean(similarities)
            fo.write("Mean similarity: {:.2f}% , Number: {}".format(sim_mean*100, len(similarities)))
        print("Mean similarity: {:.2f}% , Number: {} ".format(sim_mean*100, len(similarities)))
        return sim_mean

    def dist_mean_cosine(self, src_emb, tgt_emb, dico_method='nn', dico_build='S2T', dico_max_size=10000):
        """
        Mean-cosine model selection criterion.
        """
        # get normalized embeddings
        src_emb = self.mapping(src_emb).data
        tgt_emb = tgt_emb.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        # build dictionary
        # temp params / dictionary generation
        _args = deepcopy(self.args)
        _args.dico_method = dico_method
        _args.dico_build = dico_build
        _args.dico_threshold = 0
        _args.dico_max_rank = 10000
        _args.dico_min_size = 0
        _args.dico_max_size = dico_max_size
        s2t_candidates, s2t_scores = get_candidates(src_emb, tgt_emb, _args)
        t2s_candidates, t2s_scores = get_candidates(tgt_emb, src_emb, _args)
        #print ('S2T:\n',s2t_candidates, s2t_scores)
        #print ('T2S:\n',t2s_candidates, t2s_scores)
        dico = build_dictionary(src_emb, tgt_emb, _args, s2t_candidates, t2s_candidates)
        #print ('Dico:\n', dico)
        # mean cosine
        if dico is None:
            mean_cosine = -1e9
        else:
            mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
        mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
        logger.info("Mean cosine (%s method, %s build, %i max size): %.5f"
                    % (dico_method, _args.dico_build, dico_max_size, mean_cosine))
        to_log['mean_cosine-%s-%s-%i' % (dico_method, _args.dico_build, dico_max_size)] = mean_cosine
