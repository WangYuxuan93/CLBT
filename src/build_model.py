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
from src.maps import NonLinearMap, SelfAttentionMap, AttentionMap

transformer_types = ['self_attention','attention']

class Discriminator(nn.Module):

    def __init__(self, args, bert_hidden_size):
        super(Discriminator, self).__init__()

        self.emb_dim = bert_hidden_size
        self.dis_layers = args.dis_layers
        self.dis_hid_dim = args.dis_hid_dim
        self.dis_dropout = args.dis_dropout
        self.dis_input_dropout = args.dis_input_dropout

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


def build_model(args, with_dis):
    """
    Build all components of the model.
    """

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    print("device", device, "n_gpu", n_gpu, "distributed training", bool(args.local_rank != -1))

    bert_config = BertConfig.from_json_file(args.bert_config_file)
    model = BertModel(bert_config)
    if args.init_checkpoint is not None:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    model.to(device)

    model1 = None
    if args.bert_config_file1 and args.init_checkpoint1:
        bert_config1 = BertConfig.from_json_file(args.bert_config_file1)
        model1 = BertModel(bert_config1)
        if args.init_checkpoint is not None:
            model1.load_state_dict(torch.load(args.init_checkpoint1, map_location='cpu'))
        model1.to(device)
        assert bert_config.hidden_size == bert_config1.hidden_size

    # mapping
    if args.non_linear:
        args.emb_dim = bert_config.hidden_size
        mapping = NonLinearMap(args)
    elif args.transformer:
        if args.transformer not in transformer_types:
            raise ValueError("{} not in transformer types {}".format(args.transformer, '|'.join(transformer_types)))
            exit(1)
        if args.transformer == 'self_attention':
            mapping = SelfAttentionMap(args)
        elif args.transformer == 'attention':
            mapping = AttentionMap(args)
    elif args.fine_tune:
        mapping = None
    elif not args.bert_config_file1:
        mapping = nn.Linear(bert_config.hidden_size, bert_config.hidden_size, bias=False)
    else:
        mapping = nn.Linear(bert_config.hidden_size, bert_config1.hidden_size, bias=False)
    if not args.non_linear and not args.transformer:
        if getattr(args, 'map_id_init', True):
            mapping.weight.data.copy_(torch.diag(torch.ones(bert_config.hidden_size)))
    mapping.to(device)

    # discriminator
    discriminator = None
    if args.adversarial:
        discriminator = Discriminator(args, bert_config.hidden_size) if with_dis else None
        discriminator.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        mapping = torch.nn.DataParallel(mapping)
        if model1:
            model1 = torch.nn.DataParallel(model1)
        if args.adversarial:
            discriminator = torch.nn.DataParallel(discriminator)

    return model, mapping, discriminator, model1
