## Dependencies
* Python 3.6 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [PyTorch] 0.4.1 (http://pytorch.org/)
detailed requirements are in the requirements.txt file.

## Training

Training and testing scripts for fast startup are in scripts. (the `source` line is used to activate virtualenv environment, if the requirements are installed in corrent environment, just remove the line)

The following pre-processing is required for training, check `trial_data` file to see the format of each required input.

* Parallel sentences for source (de)/target (en) languages
* Use pytorch BERT model to predict the contextualized embeddings (-1 layer) of source language (e.g. de in the trial data) and target language (en). (For English please use the base en model, while for other languages use the multilingual model)
* Alignments between the two languages. (obtained from [fast_aling](https://github.com/clab/fast_align)) Note that this must be alignments between sentences tokenized by word piece vocabulary of corresponding BERT model.

Some options:

* --map_type: svd/linear/nonlinear. The `linear` is optimized with Adam. SVD is also a linear transformation and is faster, but `--no_cuda` option should be added when trained on big data. `linear` is recommended since it achieves the best performances in most languages.
* --n_layers: number of layers for nonlinear transformation. (don't need to add if the model is linear)
* --bert_file0: training data from source language (e.g. English).
* --bert_file1: training data from target language (e.g. German), should be aligned to bert_file0
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

