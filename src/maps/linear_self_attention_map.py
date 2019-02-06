import torch
import logging
from torch import nn
from src.maps import NonLinearMap, SelfAttentionMap, AttentionMap

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class LinearSelfAttentionMap(nn.Module):

    def __init__(self, args):
        """
        The output has the same size as the input (hidden_size)
        """
        super(LinearSelfAttentionMap, self).__init__()
        self.emb_dim = args.emb_dim

        logger.info("Linear + SelfAttention Mapping")
        logger.info("Linear mapping:\nEmbedding Dimension:{}".format(args.emb_dim))
        self.linear_map = nn.Linear(args.emb_dim, args.emb_dim, bias=False)
        if getattr(args, 'map_id_init', True):
            self.linear_map.weight.data.copy_(torch.diag(torch.ones(args.emb_dim)))
        self.self_map = SelfAttentionMap(args)

    def forward(self, input_tensor, attention_mask=None):
        """
        Input: 
            input_tensor: [batch_size, seq_len, emb_dim]
        """

        linear_output = self.linear_map(input_tensor)
        self_output = self.self_map(linear_output, attention_mask)
        return self_output
