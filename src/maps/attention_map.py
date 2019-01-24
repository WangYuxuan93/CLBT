import torch
from torch import nn
from src.bert_modeling import BertConfig, BERTSelfAttention, BERTAttention

class AttentionMap(nn.Module):

    def __init__(self, args):
        """
        The output has the same size as the input (hidden_size)
        """
        super(AttentionMap, self).__init__()

        self.config = BertConfig.from_json_file(args.bert_config_file)
        self.config.num_attention_heads = args.num_attention_heads
        self.config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        self.config.hidden_dropout_prob = args.hidden_dropout_prob
        self.attention = BERTAttention(self.config)
        print ("Self Attention Mapping:\nHidden Size:{}\nAttention Heads:{}\nAttention Dropout:{}\nHidden Dropout:{}".format(self.config.hidden_size, 
                self.config.num_attention_heads, self.config.attention_probs_dropout_prob,
                self.config.hidden_dropout_prob))

    def forward(self, input_tensor, attention_mask):
        attention_output = self.attention(input_tensor, attention_mask)
        return attention_output