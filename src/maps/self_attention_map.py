import torch
import logging
from torch import nn
from src.bert_modeling import BertConfig, BERTSelfAttention, BERTAttention

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class SelfAttentionMap(nn.Module):

    def __init__(self, args):
        """
        The output has the same size as the input (hidden_size)
        """
        super(SelfAttentionMap, self).__init__()

        self.config = BertConfig.from_json_file(args.bert_config_file)
        self.config.num_attention_heads = args.num_attention_heads
        self.config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        self.self = BERTSelfAttention(self.config)
        logger.info("Self Attention Mapping:\nHidden Size:{}\nAttention Heads:{}\nAttention Dropout:{}".format(self.config.hidden_size, 
                self.config.num_attention_heads, self.config.attention_probs_dropout_prob))

    def forward(self, input_tensor, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        self_output = self.self(input_tensor, extended_attention_mask)
        return self_output
