from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import codecs
import re
import numpy as np
import json
import argparse
from supervised_bert import *

def load_conllu(file):
  with codecs.open(file, 'rb') as f:
    reader = codecs.getreader('utf-8')(f)
    buff = []
    for line in reader:
      line = line.strip()
      if line and not line.startswith('#'):
        if not re.match('[0-9]+[-.][0-9]+', line):
          buff.append(line.split('\t')[1])
      elif buff:
        yield buff
        buff = []
    if buff:
      yield buff

def list_to_bert(sents, bert_file, layer, map_model, bert_model, max_seq=256, batch_size=8, 
                  map_input=False, map_type='linear', activation="leaky_relu", n_layers=2, 
                  hidden_size=768, num_attention_heads=12):
  model_path = map_model
  bert_config_file = bert_model+'/bert_config.json'
  vocab_file = bert_model+'/vocab.txt'
  init_checkpoint = bert_model+'/bert_model'
  output_file = bert_file
  bert_layer = layer
  max_seq_length = max_seq
  flags = Args(model_path, vocab_file, bert_config_file, init_checkpoint, output_file, 
                max_seq_length, bert_layer, map_type=map_type, activation=activation, 
                n_layers=n_layers, hidden_size=hidden_size)
  flags.batch_size = batch_size
  flags.map_input = map_input
  flags.num_attention_heads = num_attention_heads

  sup_bert = SupervisedBert(flags)
  sup_bert.list2bert(sents)
  
def merge(bert_file, merge_file, sents, merge_type='sum'):
  merge_file = merge_file+'.'+merge_type
  n = 0
  n_unk = 0
  n_tok = 0
  fo = codecs.open(merge_file, 'w')
  print ("Merge Type: {}".format(merge_type))
  with codecs.open(bert_file, 'r') as fin:
    line = fin.readline()
    while line:
      if n % 100 == 0:
        print ("\r%d" % n, end='')
      bert = json.loads(line)
      tokens = []
      merged = {"linex_index": bert["linex_index"], "features":[]}
      i = 0
      while i < len(bert["features"]):
        item = bert["features"][i]
        if item["token"]=="[CLS]" or item["token"]=="[SEP]":
          merged["features"].append(item)
        elif item["token"].startswith("##") and not (len(merged["features"])-1<len(sents[n]) and item["token"] == sents[n][len(merged["features"])-1]):
          tmp_layers = []
          for j, layer in enumerate(merged["features"][-1]["layers"]):
            #merged["features"][-1]["layers"][j]["values"] = list(np.array(layer["values"]) + np.array(item["layers"][j]["values"]))
            # j-th layer
            tmp_layers.append([np.array(layer["values"])])
            tmp_layers[j].append(np.array(item["layers"][j]["values"]))

          item = bert["features"][i+1]
          while item["token"].startswith("##") and not (len(merged["features"])-1<len(sents[n]) and item["token"] == sents[n][len(merged["features"])-1]):
            for j, layer in enumerate(merged["features"][-1]["layers"]):
              # j-th layer
              tmp_layers[j].append(np.array(item["layers"][j]["values"]))
            i += 1
            item = bert["features"][i+1]
          for j, layer in enumerate(merged["features"][-1]["layers"]):
            if merge_type == 'sum':
              merged["features"][-1]["layers"][j]["values"] = list(np.sum(tmp_layers[j], 0))
            elif merge_type == 'avg':
              merged["features"][-1]["layers"][j]["values"] = list(np.mean(tmp_layers[j], 0))
            elif merge_type == 'first':
              merged["features"][-1]["layers"][j]["values"] = list(tmp_layers[j][0])
            elif merge_type == 'last':
              merged["features"][-1]["layers"][j]["values"] = list(tmp_layers[j][-1])
            elif merge_type == 'mid':
              mid = int(len(tmp_layers[j]) / 2)
              merged["features"][-1]["layers"][j]["values"] = list(tmp_layers[j][mid])
          if len(sents[n]) < len(merged["features"]) - 1:
            print (sents[n], len(merged["features"]))
          else:
            merged["features"][-1]["token"] = sents[n][len(merged["features"])-2].lower()
        elif item["token"] == "[UNK]":
          n_unk += 1
          merged["features"].append(item)
          if len(sents[n]) < len(merged["features"]) - 1:
            print (sents[n], len(merged["features"]))
          else:
            merged["features"][-1]["token"] = sents[n][len(merged["features"])-2].lower()
        else:
          merged["features"].append(item)
        i += 1
      try:
        assert len(merged["features"]) == len(sents[n]) + 2
      except:
        orig = [m["token"] for m in merged["features"]]
        print ('\n',len(merged["features"]), len(sents[n]))
        print (sents[n], '\n', orig)
        print (zip(sents[n], orig[1:-1]))
        raise ValueError("Sentence-{}:{}".format(n, ' '.join(sents[n])))
      for i in range(len(sents[n])):
        try:
          assert sents[n][i].lower() == merged["features"][i+1]["token"]
        except:
          print ('wrong word id:{}, word:{}'.format(i, sents[n][i]))

      n_tok += len(sents[n])
      fo.write(json.dumps(merged)+"\n")
      line = fin.readline()
      n += 1
    print ('Total tokens:{}, UNK tokens:{}'.format(n_tok, n_unk))
    info_file=os.path.dirname(merge_file) + '/README.txt'
    print (info_file)
    with open(info_file, 'a') as info:
      info.write('File:{}\nTotal tokens:{}, UNK tokens:{}\n\n'.format(merge_file, n_tok, n_unk))

parser = argparse.ArgumentParser(description='CoNLLU to BERT')
parser.add_argument("bert_model", type=str, default=None, help="bert model")
parser.add_argument("conll_file", type=str, default=None, help="input conllu file")
parser.add_argument("bert_file", type=str, default=None, help="orig bert file")
parser.add_argument("merge_file", type=str, default=None, help="merged bert file")
parser.add_argument("--mapping", type=str, default=None, help="mapping model")
parser.add_argument("--layer", type=int, default=-1, help="output bert layer")
parser.add_argument("--map_input", default=False, action='store_true', help="Apply mapping to the BERT input embeddings?")
parser.add_argument("--activation", type=str, default='leaky_relu', help="learky_relu,tanh")
parser.add_argument("--n_layers", type=int, default=1, help="mapping layer")
parser.add_argument("--hidden_size", type=int, default=768, help="mapping hidden layer size")
parser.add_argument("--map_type", type=str, default=None, help="mapping type(linear|nonlinear|attention|self_attention)")
parser.add_argument("--head_num", type=int, default=12, help="attention head number")
parser.add_argument("--merge_type", type=str, default=None, help="merge type (sum|avg|first|last|mid)")
args = parser.parse_args()

map_model = args.mapping
bert_model = args.bert_model
layer = args.layer
conll_file = args.conll_file
bert_file = args.bert_file
merge_file = args.merge_file

n = 0
sents = []
for sent in load_conllu(conll_file):
  sents.append(sent)
print ("Total {} Sentences".format(len(sents)))
list_to_bert(sents,bert_file,layer,map_model, bert_model,max_seq=512,map_input=args.map_input,
            map_type=args.map_type, activation=args.activation, n_layers=args.n_layers,
            hidden_size=args.hidden_size, num_attention_heads=args.head_num)

merge_types = args.merge_type.split(',')
for merge_type in merge_types:
  merge(bert_file, merge_file, sents, merge_type=merge_type)
