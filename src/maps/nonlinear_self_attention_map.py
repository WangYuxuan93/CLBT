import torch
import logging
from torch import nn
from src.maps import NonLinearMap, SelfAttentionMap, AttentionMap

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class NonLinearSelfAttentionMap(nn.Module):

    def __init__(self, args):
        """
        The output has the same size as the input (hidden_size)
        """
        super(NonLinearSelfAttentionMap, self).__init__()
        self.emb_dim = args.emb_dim

        logger.info("NonLinear + SelfAttention Mapping")
        nonlinear_map = NonLinearMap(args)
        self_map = SelfAttentionMap(args)

    def forward(self, input_tensor, attention_mask=None):
        """
        Input: 
            input_tensor: [batch_size, seq_len, emb_dim]
        """

        nonlinear_output = self.nonlinear_map(input_tensor)
        self_output = self.self_map(linear_output, attention_mask)
        return self_output
