## Dependencies
* Python 3.6 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [PyTorch] 0.4.1 (http://pytorch.org/)
detailed requirements are in the requirements.txt file.

##Training
Training and testing scripts for fast startup are in scripts.

The following pre-processing is required for training.

* Parallel sentences for source (de)/target (en) languages
* Use pytorch BERT model to predict the contextualized embeddings of source language (e.g. de in the trial data) and target language (en). (For English please use the base en model, while for other languages use the multilingual model)
* Alignments between the two languages. (obtained from [fast_aling](https://github.com/clab/fast_align)) Note that this must be alignments between sentences tokenized by word piece vocabulary of corresponding BERT model.

Some options:

* --map_type: svd/linear/nonlinear. The `linear` is optimized with Adam. SVD is also a linear transformation and is faster, but `--no_cuda` option should be added when trained on big data. `linear` is recommended since it achieves the best performances in most languages.
* --vocab_file: bert vocab file of source language, should be the same to the vocab file of `--bert_file0`
* --vocab_file: bert vocab file of target language, should be the same to the vocab file of `--bert_file1`
* --n_epochs: maximum training epochs
* --model_path: where the model should be saved
* --batch_size: training batch size.
