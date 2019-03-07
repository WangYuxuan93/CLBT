test()
{
main=/users2/yxwang/work/codes/cl_bert/supervised_bert.py

batch=64
loss=l2_dist
n_layer=2

vocab=trial_data/multi_vocab.txt
vocab1=trial_data/en_vocab.txt

gpu=$1
input0=$2
model=$3
type=$4
output=$5

if [ -z $1 ];then
  echo "usage:./train.sh [GPU] [input] [model path] [type] [output]"
  exit
fi

source /users2/yxwang/work/env/py3.6_torch0.4.1/bin/activate
CUDA_VISIBLE_DEVICES=$gpu python $main --load_pred_bert --bert_file0 $input0 \
--model_path $model --output_file $output \
--batch_size $batch \
--n_layers $n_layer --map_type $type \
--vocab_file $vocab --pred
}

test 1 trial_data/de-en.100.de.bert saves linear de.transformed.bert
