from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import codecs
import re
import numpy as np
import json
import argparse
from bert_gan import *

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

def to_raw(sents, file):
  with codecs.open(file, 'w') as fo:
    for sent in sents:
      fo.write((" ".join(sent)).encode('utf-8')+'\n')

def bert(raw_file, bert_file, gpu, layer, model, max_seq='256',batch_size='8'):
  dict = {'gpu':gpu, 'input':raw_file, 'output':bert_file, 'layer':layer, 'BERT_BASE_DIR':model, 'max_seq':max_seq, 'batch_size':batch_size}
  cmd = "source /users2/yxwang/work/env/py2.7/bin/activate ; CUDA_VISIBLE_DEVICES={gpu} python extract_features.py --input_file={input} --output_file={output} --vocab_file={BERT_BASE_DIR}/vocab.txt --bert_config_file={BERT_BASE_DIR}/bert_config.json --init_checkpoint={BERT_BASE_DIR}/bert_model.ckpt --layers={layer} --max_seq_length={max_seq} --batch_size={batch_size}".format(**dict)
  #cmd = "./scripts/extract.sh {gpu} {input} {output} {layer}".format(**dict)
  print (cmd)
  os.system(cmd)

def list_to_bert(sents, bert_file, layer, map_model, bert_model, max_seq=256, batch_size=8, map_input=False):
  model_path = map_model
  bert_config_file = bert_model+'/bert_config.json'
  vocab_file = bert_model+'/vocab.txt'
  init_checkpoint = bert_model+'/bert_model'
  output_file = bert_file
  bert_layer = layer
  max_seq_length = max_seq
  flags = Args(model_path, vocab_file, bert_config_file, init_checkpoint, output_file, max_seq_length, bert_layer)
  flags.batch_size = batch_size
  flags.map_input = map_input

  adv_bert = AdvBert(flags)
  adv_bert.list2bert(sents)
  
def merge(bert_file, merge_file, sents):
  n = 0
  n_unk = 0
  n_tok = 0
  fo = codecs.open(merge_file, 'w')
  with codecs.open(bert_file, 'r') as fin:
    line = fin.readline()
    while line:
      if n % 100 == 0:
        print ("\r%d" % n, end='')
      bert = json.loads(line)
      tokens = []
      merged = {"linex_index": bert["linex_index"], "features":[]}
      for i, item in enumerate(bert["features"]):
        if item["token"]=="[CLS]" or item["token"]=="[SEP]":
          merged["features"].append(item)
          continue
        if item["token"].startswith("##") and not (len(merged["features"])-1<len(sents[n]) and item["token"] == sents[n][len(merged["features"])-1]):
          for j, layer in enumerate(merged["features"][-1]["layers"]):
            merged["features"][-1]["layers"][j]["values"] = list(np.array(layer["values"]) + np.array(item["layers"][j]["values"]))
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

#if len(sys.argv) < 7:
#  print ("usage:%s [map_model] [bert_model] [layer(-1)] [conllu file] [output bert] [merged bert]" % sys.argv[0])
#  exit(1)

parser = argparse.ArgumentParser(description='CoNLLU to BERT')
parser.add_argument("bert_model", type=str, default=None, help="bert model")
parser.add_argument("conll_file", type=str, default=None, help="input conllu file")
parser.add_argument("bert_file", type=str, default=None, help="orig bert file")
parser.add_argument("merge_file", type=str, default=None, help="merged bert file")
parser.add_argument("--mapping", type=str, default=None, help="mapping model")
parser.add_argument("--layer", type=int, default=-1, help="output bert layer")
parser.add_argument("--map_input", default=False, action='store_true', help="Apply mapping to the BERT input embeddings?")
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
list_to_bert(sents,bert_file,layer,map_model, bert_model,max_seq=512,map_input=args.map_input)
merge(bert_file, merge_file, sents)
