# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

from src.utils import load_embeddings, normalize_embeddings
from src.bert_modeling import BertConfig, BertModel

class Discriminator(nn.Module):

    def __init__(self, params, bert_hidden_size):
        super(Discriminator, self).__init__()

        self.emb_dim = bert_hidden_size
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)


def build_model(params, with_dis):
    """
    Build all components of the model.
    """

    if params.local_rank == -1 or params.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not params.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", params.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device", device, "n_gpu", n_gpu, "distributed training", bool(params.local_rank != -1))

    bert_config = BertConfig.from_json_file(params.bert_config_file)
    model = BertModel(bert_config)
    if params.init_checkpoint is not None:
        model.load_state_dict(torch.load(params.init_checkpoint, map_location='cpu'))
    model.to(device)

    # mapping
    #mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
    mapping = nn.Linear(bert_config.hidden_size, bert_config.hidden_size, bias=False)
    if getattr(params, 'map_id_init', True):
        mapping.weight.data.copy_(torch.diag(torch.ones(bert_config.hidden_size)))
    mapping.to(device)

    # discriminator
    discriminator = Discriminator(params, bert_config.hidden_size) if with_dis else None
    discriminator.to(device)

    if params.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.local_rank],
                                                          output_device=params.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        mapping = torch.nn.DataParallel(mapping)
        discriminator = torch.nn.DataParallel(discriminator)

    return model, mapping, discriminator
