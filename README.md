## Cross-Lingual BERT Transformation (CLBT)

This is the implementation of the paper [Cross-Lingual BERT Transformation for Zero-Shot Dependency Parsing](https://arxiv.org/abs/1909.06775). If you use this code or our results in your research, we'd appreciate you cite our paper as following:

```
@inproceedings{wang2019cross,
  author    = {Wang, Yuxuan and Che, Wanxiang and Guo, Jiang and Liu, Yijia and  Liu, Ting},
  title     = {Cross-Lingual BERT Transformation for Zero-Shot Dependency Parsing},
  booktitle = {Proc. of EMNLP},
  year      = {2019}
}
```

## Dependencies
* Python 3.6 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [PyTorch] 0.4.1 (http://pytorch.org/)
detailed requirements are in the requirements.txt file.

## Training

Training and testing scripts for fast startup are in the `scripts` file. (the `source` line is used to activate virtualenv environment, if the requirements are installed in corrent environment, just remove the line)
Please first run the `train.sh` shell to train a model and the the `test.sh` shell to test the model.

The following pre-processing is required for training, check `trial_data` file to see the format of each required input.

* Parallel sentences for source (de)/target (en) languages
* Use pytorch BERT model to predict the contextualized embeddings (-1 layer) of source language (e.g. de in the trial data) and target language (en). (For English please use the base en model, while for other languages use the multilingual model)
* Alignments between the two languages. (obtained from [fast_aling](https://github.com/clab/fast_align)) Note that this must be alignments between sentences tokenized by word piece vocabulary of corresponding BERT model.

Some options:

* --map_type: svd/linear/nonlinear. The `linear` is optimized with Adam. SVD is also a linear transformation and is faster, but `--no_cuda` option should be added when trained on big data. `linear` is recommended since it achieves the best performances in most languages.
* --n_layers: number of layers for nonlinear transformation. (don't need to add if the model is linear)
* --bert_file0: training data from source language (e.g. English).
* --bert_file1: training data from target language (e.g. German), should be parallel to bert_file0
* --align_file: the alignment obtained by `fast_align`, the parallel sentences should be tokenized with the corresponding BERT vocabulary. (i.e. use EN Base vocab for English, use Multilingual Base vocab for other languages)
* --vocab_file: bert vocab file of source language, should be the same to the vocab file of `--bert_file0`.
* --vocab_file1: bert vocab file of target language, should be the same to the vocab file of `--bert_file1`.
* --n_epochs: maximum training epochs.
* --model_path: where the model should be saved.
* --batch_size: training batch size.

## Transforming

The testing procedure requires a bert file (-1 layer) and a transformation model trained by the training process. 

Options:

* --model_path: input transformation model path.
* --bert_file0: input bert file to be transformed.
* --map_type: for svd/linear model, this should be `linear`, while for nonlinear model, this is `nonlinear`.
* --n_layers: number of layers for nonlinear transformation, this must be the same as the layer in the input model (don't need to add if the model is linear).
* --output_file: output path.
* --batch_size: predicting batch size.
* --pred: activate predicting mode.

## Using Pretrained Transformation

We have uploaded pretrained transformation matrices in the `models` file for 17 languages with 2 optimizing mechanisms (i.e., SVD and GD). 
If you would like to use these pretrained matrices, just use the `transform.sh` shell from the `scripts` file:

`./scripts/transform.sh [GPU] [input bert file] [optimizer(gd|svd)] [language pair(e.g. de-en)] [output bert file]`

For instance, to transform the German bert file in the `trial_data` with matrix trianed by SVD, use:

`./scripts/transform.sh 1 trial_data/de.trial.bert svd de-en de.trial.svd.bert`

The languages whose transformation matrices to English is provided are listed below:

* Bulgarian (bg)
* Czech (cs)
* Danish (da)
* German (de)
* Spanish (es)
* Estonian (et)
* Finnish (fi)
* French (fr)
* Italian (it)
* Latvian (lv)
* Dutch (nl)
* Polish (pl)
* Portuguese (pt)
* Romanian (ro)
* Slovak (sk)
* Slovenian (sl)
* Swedish (sv)
